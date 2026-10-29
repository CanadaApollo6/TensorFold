const std = @import("std");
const llama = @import("llama");
const lanes = @import("lanes");
const check = @import("check.zig");
const tests = @import("llama_tests.zig");
const Hook = struct {
    m: *llama.forward.Model,
    logits: []f32,
    fn head(ptr: *anyopaque, x: u64, y: u64, rows: usize) !void {
        const h: *Hook = @ptrCast(@alignCast(ptr));
        try h.m.k.exl.decode(h.m.s, h.m.w.head, x, .bf16, rows, y, .f32);
        if (rows * h.m.w.config.vocab != h.logits.len) return error.DraftLogitShape;
        try h.m.download(y, std.mem.sliceAsBytes(h.logits));
    }
};
pub fn run(gpu: check.Gpu, model: []const u8, draft: []const u8, prompt: []const u32, expected: []const u32, index: usize, samples: usize, writer: *std.Io.Writer) !void {
    _ = samples;
    var rig: tests.Rig = undefined;
    try rig.init(gpu, model, null, prompt.len + 32);
    defer rig.deinit();
    const mode = try llama.native.kvMode();
    if (mode != rig.m.kv_mode) {
        rig.m.deinit();
        rig.m = try llama.forward.Model.initMode(gpu.gpa, gpu.d, &rig.w, &rig.k, rig.s, prompt.len + 32, mode);
    }
    for (prompt, 0..) |token, pos| try rig.m.step(token, pos, pos + 1 == prompt.len);
    var batch = try llama.forward.Model.initRows(gpu.gpa, gpu.d, &rig.w, &rig.k, rig.s, prompt.len + 32, try llama.native.kvMode(), 8);
    defer batch.deinit();
    const format = try llama.draft.storage.Format.parse(if (std.c.getenv("TF_LLAMA_DRAFT_FORMAT")) |v| std.mem.span(v) else "affine4");
    var attachment = try llama.draft.Attachment.init(gpu.gpa, gpu.io, &batch, draft, format, null);
    defer attachment.deinit();
    var adapter: llama.backend.Llama = .{ .m = &batch, .drafter = &attachment };
    const backend = adapter.backend();
    var stream = try lanes.Stream.init(gpu.gpa, .{ .id = "cached", .prompt = prompt, .max_new = 16, .eos = &.{}, .drafts = true, .sampling = .{ .seed = 0, .temperature = 0.7, .top_k = 3, .top_p = 0.8 } });
    defer stream.deinit(gpu.gpa);
    try backend.prefill(&stream);
    defer backend.vtable.release(backend.ptr, &stream);
    const vocab = rig.w.config.vocab;
    const draft_logits = try gpu.gpa.alloc(f32, (attachment.r.graph.config.block - 1) * vocab);
    defer gpu.gpa.free(draft_logits);
    var hook: Hook = .{ .m = &batch, .logits = draft_logits };
    attachment.r.target.ptr = &hook;
    attachment.r.target.head = Hook.head;
    try attachment.propose(expected[0], 7);
    if (attachment.count == 0) return error.DraftDidNotRun;
    var positions: [8]u64 = undefined;
    for (&positions, 0..) |*pos, row| pos.* = prompt.len + row + 1;
    const window: lanes.backend.Window = .{ .stream = &stream, .pending = expected[0], .tokens = expected[1..8], .positions = &positions, .held = 0, .parents = null };
    var drawn: [8]u32 = undefined;
    var verified = [_]lanes.backend.Verified{.{ .sampled = &drawn, .drafts = &.{} }};
    try backend.verify(&.{window}, &verified);
    const base = batch.logits;
    defer batch.logits = base;
    const target = try gpu.gpa.alloc(f32, vocab);
    defer gpu.gpa.free(target);
    for (0..8) |row| {
        try rig.m.step(expected[row], prompt.len + row, true);
        if (row != 0 and row != 3 and row != 6) continue;
        batch.logits = base + row * vocab * 4;
        try batch.download(batch.logits, std.mem.sliceAsBytes(target));
        for (0..4) |variant| {
            const temp = [_]f64{ 0.3, 0.7, 1, 1.3 };
            const top_k = [_]u32{ 0, 3, 20, 50 };
            const top_p = [_]f64{ 1, 0.8, 0.95, 0.9 };
            var draw: llama.sample_host.Draw = .{ .position = @intCast(prompt.len + row + 1), .sampling = .{ .seed = 0, .temperature = temp[variant], .top_k = top_k[variant], .top_p = top_p[variant] }, .penalties = if (variant == 3) .{ .presence = 0.7, .frequency = 0.3 } else .{}, .history = expected[0..row], .pending = expected[row] };
            const cpu = try llama.sample_host.Prepared.init(gpu.gpa, target, draw);
            defer cpu.deinit(gpu.gpa);
            for (0..16) |seed| {
                draw.sampling.?.seed = seed;
                const token = try batch.pick(.{ .position = draw.position, .sampling = draw.sampling, .penalties = draw.penalties, .history = draw.history, .pending = draw.pending });
                const serial = try rig.m.pick(.{ .position = draw.position, .sampling = draw.sampling, .penalties = draw.penalties, .history = draw.history, .pending = draw.pending });
                if (token != serial) return error.SerialBatchSampleMismatch;
                if (token != cpu.pick(seed, draw.position)) return error.CoupledSamplerMismatch;
                if (variant == 1 and seed == 0 and token != drawn[row]) return error.VerifySamplingMismatch;
            }
            try writer.print("{f}\n", .{std.json.fmt(.{ .prompt = index, .row = row, .variant = variant, .target = target, .draft = draft_logits[row * vocab ..][0..vocab], .sampling = draw.sampling.?, .penalties = draw.penalties, .history = draw.history, .pending = draw.pending.?, .coupled = 16 }, .{})});
        }
    }
}
