const std = @import("std");
const llama = @import("llama");
const lanes = @import("lanes");
const host = llama.sample_host;
const Case = struct {
    target: []const f32,
    draft: []const f32,
    sampling: lanes.Sampling,
    penalties: lanes.Penalties = .{},
    pending: u32,
    history: []const u32 = &.{},
    mutation: []const u8 = "none",
    draws: usize = 32768,
    seed_start: u64 = 0,
};
const Runtime = struct {
    graph: struct { config: struct { window: usize = 2048 } = .{} } = .{},
    context: struct { cache: usize = 1, begin: usize = 0, end: usize = 0 } = .{},
    pub fn reset(r: *Runtime) void { r.context = .{}; }
};
const Draft = struct {
    r: *Runtime,
    held: [7]u32,
    count: usize = 0,
    pub fn capture(_: *Draft, _: usize) void {}
    pub fn ingest(d: *Draft, _: usize, pos: usize) !void { d.r.context.end = pos + 1; }
    pub fn propose(d: *Draft, _: u32, depth: u32) !void { d.count = @min(depth, 7); }
};
const Model = struct {
    gpa: std.mem.Allocator,
    w: struct { config: struct { bos: u32 = 1, vocab: usize } },
    k: struct { draws: struct { penalize_fn: ?usize = 1 } = .{} } = .{},
    s: struct { pub fn synchronize(_: @This()) !void {} } = .{},
    context: usize = 32,
    tap_buffer: u64 = 0,
    logits: u64 = 0,
    case: Case,
    cached: ?host.Prepared = null,
    pub fn step(_: *Model, _: u32, _: usize, _: bool) !void {}
    pub fn batch(_: *Model, _: []const u32, _: usize) !void {}
    pub fn pick(m: *Model, d: host.Draw) !u32 {
        if (d.position - 2 < m.case.history.len) return m.case.history[d.position - 2];
        if (d.position == 2 + m.case.history.len) return m.case.pending;
        if (d.position != 3 + m.case.history.len) return 0;
        if (d.sampling == null) return error.SamplingPropagationLost;
        if (m.cached == null) m.cached = try prepare(m.gpa, m.case, d);
        return m.cached.?.pick(d.sampling.?.seed, d.position);
    }
};
fn prepare(a: std.mem.Allocator, c: Case, draw: host.Draw) !host.Prepared {
    var d = draw;
    if (std.mem.eql(u8, c.mutation, "temperature_twice")) d.sampling.?.temperature *= d.sampling.?.temperature;
    if (std.mem.eql(u8, c.mutation, "penalties_draft_only")) d.penalties = .{};
    var p = try host.Prepared.init(a, c.target, d);
    if (std.mem.eql(u8, c.mutation, "none") or std.mem.eql(u8, c.mutation, "temperature_twice") or std.mem.eql(u8, c.mutation, "penalties_draft_only")) return p;
    defer p.deinit(a);
    const q = try host.Prepared.init(a, c.draft, draw);
    defer q.deinit(a);
    const probs = try a.alloc(f64, c.target.len * 3);
    defer a.free(probs);
    const pp = probs[0..c.target.len];
    const qq = probs[c.target.len..][0..c.target.len];
    const bug = probs[c.target.len * 2 ..];
    p.probabilities(pp);
    q.probabilities(qq);
    if (std.mem.eql(u8, c.mutation, "draft_support")) {
        for (bug, pp, qq) |*v, pv, qv| v.* = if (qv > 0) pv else 0;
    } else if (std.mem.eql(u8, c.mutation, "within_bin_transfer")) {
        @memcpy(bug, pp);
        const top = p.entries[0].i;
        const other = (top + 8) % c.target.len;
        bug[other] += bug[top] * 0.5;
        bug[top] *= 0.5;
    } else {
        var reject: f64 = 0;
        var z: f64 = 0;
        for (pp, qq) |pv, qv| {
            reject += @max(pv - qv, 0);
            z += if (std.mem.eql(u8, c.mutation, "residual_replaced")) pv else @max(qv - pv, 0);
        }
        for (bug, pp, qq) |*v, pv, qv| v.* = @min(pv, qv) + reject * (if (std.mem.eql(u8, c.mutation, "residual_replaced")) pv else @max(qv - pv, 0)) / z;
    }
    var entries: std.ArrayList(lanes.gpu_rule.Cand) = .empty;
    for (bug, 0..) |v, i| if (v > 0) { try entries.append(a, .{ .v = @floatCast(@log(v)), .i = @intCast(i) }); };
    return .{ .entries = try entries.toOwnedSlice(a) };
}
pub fn main(init: std.process.Init) !void {
    const a = init.arena.allocator();
    const argv = try init.minimal.args.toSlice(a);
    const raw = try std.Io.Dir.cwd().readFileAlloc(init.io, argv[1], a, .limited(128 << 20));
    const parsed = try std.json.parseFromSlice([]Case, a, raw, .{});
    var results: std.Io.Writer.Allocating = .init(a);
    for (parsed.value) |c| {
        var m: Model = .{ .gpa = a, .w = .{ .config = .{ .vocab = c.target.len } }, .case = c };
        var rt: Runtime = .{};
        var draft: Draft = .{ .r = &rt, .held = @splat(lanes.gpu_rule.argmax(c.draft)) };
        var adapter: llama.backend.Adapter(Model, Draft) = .{ .m = &m, .drafter = &draft };
        var cfg = try lanes.Config.init(a, adapter.facts(), 8, 7);
        defer cfg.deinit(a);
        const counts = try a.alloc(u32, c.target.len);
        @memset(counts, 0);
        var accepted: u64 = 0;
        var proposed: u64 = 0;
        for (0..c.draws) |seed| {
            var clock: lanes.fake.FixedClock = .{};
            var engine = lanes.Engine.init(a, &cfg, adapter.backend(), clock.clock());
            defer engine.deinit();
            var sp = c.sampling;
            sp.seed = c.seed_start + seed;
            var stream = try lanes.Stream.init(a, .{ .id = "cached", .prompt = &.{1, 2}, .max_new = @intCast(c.history.len + 3), .eos = &.{}, .drafts = true, .sampling = sp, .penalties = c.penalties });
            defer stream.deinit(a);
            try engine.addStream(&stream);
            while (engine.activeCount() > 0) try engine.step();
            if (stream.emitted().len != c.history.len + 3 or stream.emitted()[c.history.len] != c.pending) return error.ReplayShape;
            if (!std.mem.eql(u32, stream.emitted()[0..c.history.len], c.history)) return error.HistoryPropagationLost;
            if (stream.emitted()[c.history.len + 2] != 0) return error.SpeculativeCommitMismatch;
            counts[stream.emitted()[c.history.len + 1]] += 1;
            accepted += stream.accepted;
            proposed += stream.drafted;
        }
        if (m.cached) |p| p.deinit(a);
        try results.writer.print("{f}\n", .{std.json.fmt(.{ .counts = counts, .accepted = accepted, .proposed = proposed, .draws = c.draws }, .{})});
    }
    try std.Io.Dir.cwd().writeFile(init.io, .{ .sub_path = argv[2], .data = results.written() });
}
