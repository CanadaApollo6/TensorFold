const std = @import("std");
const cuda = @import("cuda");
const llama = @import("llama");
const check = @import("check.zig");
const lanes = @import("lanes");
const shared = llama.draft.runtime;
const op = @import("llama").draft.operators;

pub fn run(gpu: check.Gpu, model: []const u8, draft: []const u8, taps_file: []const u8, out: []const u8, format: []const u8) !void {
    const a = gpu.gpa;
    var w = try llama.weights.load(a, gpu.io, gpu.ctx, model, 2, 32);
    defer w.deinit();
    var k = try llama.kernels.Kernels.load(gpu.ctx);
    defer k.deinit();
    var s = try cuda.Stream.init(gpu.d, false);
    defer s.deinit();
    for ([_]cuda.kv4.Mode{ .bf16, .q4 }) |mode| {
        var serial = try llama.forward.Model.initMode(a, gpu.d, &w, &k, s, 32, mode);
        defer serial.deinit();
        var batched = try llama.forward.Model.initRows(a, gpu.d, &w, &k, s, 32, mode, 8);
        defer batched.deinit();
        const ids = [_]u32{ @intCast(w.config.bos), 42, 19, 7, 109, 87, 91, 63, 29, 18, 1, 31, 51, 78, 99, 83 };
        const reference = try a.alloc(u8, ids.len * w.config.vocab * 4);
        defer a.free(reference);
        for (ids, 0..) |id, pos| {
            try serial.step(id, pos, true);
            try serial.download(serial.logits, reference[pos * w.config.vocab * 4 ..][0 .. w.config.vocab * 4]);
        }
        const read = try a.alloc(u8, 8 * w.config.vocab * 4);
        defer a.free(read);
        for ([_]usize{ 2, 3, 8 }) |width| {
            var pos: usize = 0;
            while (pos < ids.len) {
                const n = @min(width, ids.len - pos);
                if (n == 1) try batched.step(ids[pos], pos, true) else try batched.batch(ids[pos..][0..n], pos);
                try batched.download(batched.logits, read[0 .. n * w.config.vocab * 4]);
                try check.sameBytes("real two-layer batched logits", read[0 .. n * w.config.vocab * 4], reference[pos * w.config.vocab * 4 ..][0 .. n * w.config.vocab * 4]);
                pos += n;
            }
            check.pass("draft verify mode={s} width={d} rows={d} logits byte-equal", .{ @tagName(mode), width, ids.len });
        }
        try batched.batch(&.{ ids[8], ids[9], ids[10], ids[11] }, 8);
        try batched.step(ids[9], 9, true);
        try batched.download(batched.logits, read[0 .. w.config.vocab * 4]);
        try check.sameBytes("rejected suffix overwritten", read[0 .. w.config.vocab * 4], reference[9 * w.config.vocab * 4 ..][0 .. w.config.vocab * 4]);
        check.pass("draft rollback mode={s} prefix=9", .{@tagName(mode)});
    }
    var m = try llama.forward.Model.initRows(a, gpu.d, &w, &k, s, 32, .bf16, 8);
    defer m.deinit();
    const before = cuda.usage(true);
    var attachment = try llama.draft.Attachment.init(a, gpu.io, &m, draft, try llama.draft.storage.Format.parse(format));
    defer attachment.deinit();
    const loaded = cuda.usage(false);
    const raw = try std.Io.Dir.cwd().readFileAlloc(gpu.io, taps_file, a, .limited(1 << 20));
    defer a.free(raw);
    if (raw.len != attachment.r.graph.config.tapWidth() * 2) return error.BadCommittedTaps;
    try attachment.r.upload(attachment.captures.ptr, raw);
    try attachment.ingest(0, 0);
    var capture: Capture = .{ .gpu = gpu, .m = &m, .out = out };
    attachment.r.target.ptr = &capture;
    attachment.r.target.head = Capture.head;
    try attachment.propose(42, 7);
    std.debug.print("PROPOSALS format={s} tokens={any}\n", .{ format, attachment.held[0..attachment.count] });
    const used = cuda.usage(false);
    const predicted = llama.draft.plan.fixed(attachment.r.graph.config, attachment.r.w.footprint, try llama.draft.plan.scratch(a, &attachment.r.graph, attachment.r.w.format));
    try check.expect(predicted == used.device - before.device, "draft allocation plan {d} != measured {d}", .{ predicted, used.device - before.device });
    std.debug.print("FOOTPRINT format={s} native={d} codebooks_host={d} weight_device={d} stage={d} loaded_delta={d} warm_delta={d} peak_delta={d} runtime_pool={d} proposed={d}\n", .{ format, attachment.r.w.footprint.native, attachment.r.w.footprint.host_codebooks, attachment.r.w.footprint.device, attachment.r.w.footprint.stage, loaded.device - before.device, used.device - before.device, used.peak - before.device, attachment.r.bytes() - attachment.r.w.footprint.device, attachment.count });
    m.tap_ids = .{ 0, 1, 0, 1, 0 };
    attachment.r.graph.config.window = 8;
    for (0..4) |i| {
        const seed: ?u64 = if (i == 0 or i == 3) null else i;
        const plain = try generate(gpu, &m, null, seed, i >= 2);
        defer a.free(plain);
        const drafted = try generate(gpu, &m, &attachment, seed, i >= 2);
        defer a.free(drafted);
        try check.expect(std.mem.eql(u32, plain, drafted), "partial lane tokens differ sample={d}", .{i});
    }
    check.pass("partial production lane equality greedy=32 sampled=32 penalties=2 window=8 diagnostic tap mapping", .{});
    std.debug.print("SUMMARY draft partial target_layers=2 modes=2 widths=3 checked_logits=96 rollback=2 format={s}\n", .{format});
}
const Capture = struct {
    gpu: check.Gpu,
    m: *llama.forward.Model,
    out: []const u8,
    saved: bool = false,
    fn head(ptr: *anyopaque, x: u64, y: u64, rows: usize) !void {
        const c: *Capture = @ptrCast(@alignCast(ptr));
        const bytes = try c.gpu.gpa.alloc(u8, rows * c.m.w.config.hidden * 2);
        defer c.gpu.gpa.free(bytes);
        try c.m.download(x, bytes);
        if (!c.saved) try std.Io.Dir.cwd().writeFile(c.gpu.io, .{ .sub_path = c.out, .data = bytes });
        c.saved = true;
        try c.m.k.exl.decode(c.m.s, c.m.w.head, x, .bf16, rows, y, .f32);
    }
};

fn generate(gpu: check.Gpu, m: *llama.forward.Model, attachment: ?*llama.draft.Attachment, seed: ?u64, penalties: bool) ![]u32 {
    var backend: llama.backend.Llama = .{ .m = m, .drafter = attachment };
    var cfg = try lanes.Config.init(gpu.gpa, backend.facts(), if (attachment != null) 8 else 1, if (attachment != null) 7 else 0);
    defer cfg.deinit(gpu.gpa);
    var clock: lanes.backend.WallClock = .{ .io = gpu.io };
    var engine = lanes.Engine.init(gpu.gpa, &cfg, backend.backend(), clock.clock());
    defer engine.deinit();
    const prompt = [_]u32{ @intCast(m.w.config.bos), 42, 19, 7, 109, 87, 91, 63, 29, 18 };
    var stream = try lanes.Stream.init(gpu.gpa, .{ .id = "partial", .prompt = &prompt, .max_new = 16, .eos = &.{}, .drafts = attachment != null, .penalties = if (penalties) .{ .presence = 0.5, .frequency = 0.25 } else .{}, .sampling = if (seed) |n| .{ .seed = n, .temperature = 0.7 } else null });
    defer stream.deinit(gpu.gpa);
    try engine.addStream(&stream);
    while (engine.activeCount() > 0) try engine.step();
    if (stream.emitted().len != 16) return error.ShortGeneration;
    if (attachment != null and stream.drafted == 0) return error.DraftDidNotRun;
    return gpu.gpa.dupe(u32, stream.emitted());
}
