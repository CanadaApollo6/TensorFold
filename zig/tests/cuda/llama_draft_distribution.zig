const std = @import("std");
const llama = @import("llama");
const lanes = @import("lanes");
const check = @import("check.zig");
const tests = @import("llama_tests.zig");

pub fn run(gpu: check.Gpu, model: []const u8, draft: []const u8, prompt: []const u32, expected: []const u32, index: usize, samples: usize, writer: *std.Io.Writer) !void {
    var rig: tests.Rig = undefined;
    try rig.init(gpu, model, null, prompt.len + 32);
    defer rig.deinit();
    const mode = try llama.native.kvMode();
    if (mode != rig.m.kv_mode) {
        rig.m.deinit();
        rig.m = try llama.forward.Model.initMode(gpu.gpa, gpu.d, &rig.w, &rig.k, rig.s, prompt.len + 32, mode);
    }
    var batch = try llama.forward.Model.initRows(gpu.gpa, gpu.d, &rig.w, &rig.k, rig.s, prompt.len + 32, rig.m.kv_mode, 8);
    defer batch.deinit();
    const format = try llama.draft.storage.Format.parse(if (std.c.getenv("TF_LLAMA_DRAFT_FORMAT")) |v| std.mem.span(v) else "affine4");
    var attachment = try llama.draft.Attachment.init(gpu.gpa, gpu.io, &batch, draft, format, null);
    defer attachment.deinit();
    var adapter: llama.backend.Llama = .{ .m = &batch, .drafter = &attachment };
    const backend = adapter.backend();
    var stream = try lanes.Stream.init(gpu.gpa, .{ .id = "distribution", .prompt = prompt, .max_new = 16, .eos = &.{}, .drafts = true });
    defer stream.deinit(gpu.gpa);
    try backend.prefill(&stream);
    defer backend.vtable.release(backend.ptr, &stream);
    try attachment.propose(expected[0], 7);
    if (attachment.count == 0) return error.DraftDidNotRun;
    var positions: [8]u64 = undefined;
    for (&positions, 0..) |*pos, row| pos.* = prompt.len + row + 1;
    const window: lanes.backend.Window = .{ .stream = &stream, .pending = expected[0], .tokens = expected[1..8], .positions = &positions, .held = 0, .parents = null };
    var drawn: [8]u32 = undefined;
    var verified = [_]lanes.backend.Verified{.{ .sampled = &drawn, .drafts = &.{} }};
    try backend.verify(&.{window}, &verified);
    for (prompt, 0..) |token, pos| try rig.m.step(token, pos, pos + 1 == prompt.len);
    const base = batch.logits;
    defer batch.logits = base;
    for (0..8) |row| {
        try rig.m.step(expected[row], prompt.len + row, true);
        if (row != 1 and row != 4 and row != 7) continue;
        batch.logits = base + row * rig.w.config.vocab * 4;
        for (0..4) |variant| {
            const temp = [_]f32{ 0.3, 0.7, 1, 1.3 };
            const top_k = [_]u32{ 0, 3, 20, 50 };
            const top_p = [_]f32{ 1, 0.8, 0.95, 0.9 };
            var counts: [2][8]u32 = @splat(@splat(0));
            for (0..samples) |sample| for ([_]*llama.forward.Model{ &rig.m, &batch }, 0..) |m, side| {
                const token = try m.pick(.{ .position = @intCast(prompt.len + row + 1), .sampling = .{ .seed = @intCast(sample + side * 1000000), .temperature = temp[variant], .top_k = top_k[variant], .top_p = top_p[variant] }, .penalties = if (variant == 3) .{ .presence = 0.7, .frequency = 0.3 } else .{}, .history = expected[0..row], .pending = expected[row] });
                counts[side][token % 8] += 1;
            };
            try writer.print("{f}\n", .{std.json.fmt(.{ .prompt = index, .row = row, .variant = variant, .draws = samples, .counts = counts }, .{})});
        }
    }
}
