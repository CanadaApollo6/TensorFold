const std = @import("std");
const lanes = @import("lanes.zig");
const be = lanes.backend;
const a = std.testing.allocator;

const Replay = struct {
    target: lanes.fake.Fake = .{ .gpa = a },
    accepted: usize,
    now: f64 = 0,
    started: f64 = 0,
    early_calls: usize = 0,
    clock_reads: usize = 0,
    expected_accepted: usize = 0,
    plain_rounds: usize = 0,
    deep_rounds: usize = 0,

    fn self(ptr: *anyopaque) *Replay {
        const target: *lanes.fake.Fake = @ptrCast(@alignCast(ptr));
        return @fieldParentPtr("target", target);
    }

    fn backend(r: *Replay, vt: *be.Backend.VTable) be.Backend {
        var b = r.target.backend();
        vt.* = b.vtable.*;
        vt.verify = verify;
        vt.draft = draft;
        b.vtable = vt;
        return b;
    }

    fn verify(ptr: *anyopaque, windows: []const be.Window, out: []be.Verified) !void {
        const r = self(ptr);
        try r.target.backend().vtable.verify(ptr, windows, out);
        r.now += 42;
        for (windows, out) |w, o| {
            if (w.masks.len == 0) for (o.drafts, 0..) |token, i| {
                o.sampled[i] = if (i < r.accepted) token else (token + 1) % lanes.fake.vocab;
            };
            if (r.target.rounds > 872) {
                r.plain_rounds += @intFromBool(w.rows() == 1);
                r.deep_rounds += @intFromBool(w.rows() >= 7);
            }
            r.expected_accepted += @min(r.accepted, w.rows() - 1);
        }
    }

    fn draft(ptr: *anyopaque, requests: []const be.DraftRequest) !void {
        const r = self(ptr);
        for (requests) |req| {
            r.now += if (req.depth > 0) @as(f64, 54) else 0;
            r.early_calls += @intFromBool(req.early);
        }
        try r.target.backend().vtable.draft(ptr, requests);
    }

    fn clock(r: *Replay) be.Clock {
        return .{ .ptr = r, .vtable = &.{ .start = start, .elapsed_ms = elapsed } };
    }

    fn start(ptr: *anyopaque) void {
        const r: *Replay = @ptrCast(@alignCast(ptr));
        r.started = r.now;
    }

    fn elapsed(ptr: *anyopaque, _: be.Mark) f64 {
        const r: *Replay = @ptrCast(@alignCast(ptr));
        r.clock_reads += 1;
        return r.now - r.started;
    }
};

fn config(early: bool) !lanes.Config {
    var costs: [8]lanes.config.Cost = undefined;
    for (&costs, 1..) |*c, width| c.* = .{ .width = @intCast(width), .ms = 42 };
    return lanes.Config.init(a, .{ .exact_width = 8, .copy_drafts = false, .mtp = true, .speculate = true, .speculate_early = early, .drafts = 7, .plain_guard = true, .window_costs = &costs }, 8, 7);
}

fn replay(accepted: usize, early: bool) !void {
    var cfg = try config(early);
    defer cfg.deinit(a);
    var r: Replay = .{ .accepted = accepted };
    defer r.target.deinit();
    var vt: be.Backend.VTable = undefined;
    var engine = lanes.Engine.init(a, &cfg, r.backend(&vt), r.clock());
    defer engine.deinit();
    var stream = try lanes.Stream.init(a, .{ .id = "cost", .prompt = &.{ 1, 2 }, .max_new = 10000, .eos = &.{}, .drafts = true });
    defer stream.deinit(a);
    try engine.addStream(&stream);
    for (0..1000) |_| try engine.step();
    try std.testing.expectEqual(@as(u64, 1000), stream.rounds);
    try std.testing.expectEqual(@as(u64, @intCast(r.expected_accepted)), stream.accepted);
    try std.testing.expectEqual(early, r.early_calls > 0);
    std.debug.print("FAMILY_COST accepted={d} early={} final128 plain={d} deep={d}\n", .{ accepted, early, r.plain_rounds, r.deep_rounds });
    // Eight exceptions allow periodic exploration while requiring at least 93.75 percent useful choices.
    if (accepted == 1) try std.testing.expect(r.plain_rounds >= 120) else try std.testing.expect(r.deep_rounds >= 120);
}

test "family cost: a losing late drafter stops drafting" {
    try replay(1, false);
}

test "family cost: a losing early drafter stops drafting" {
    try replay(1, true);
}

test "family cost: a winning late drafter keeps drafting deep" {
    try replay(6, false);
}

test "family cost: a winning early drafter keeps drafting deep" {
    try replay(6, true);
}

test "family cost: initialization and forced rounds do not leak draft costs" {
    for ([_]bool{ false, true }) |early| {
        var cfg = try config(early);
        defer cfg.deinit(a);
        try cfg.lane_costs.put(a, 8, 42);
        var r: Replay = .{ .accepted = 6 };
        defer r.target.deinit();
        var vt: be.Backend.VTable = undefined;
        var engine = lanes.Engine.init(a, &cfg, r.backend(&vt), r.clock());
        defer engine.deinit();
        var s = try lanes.Stream.init(a, .{ .id = "transitions", .prompt = &.{ 1, 2 }, .max_new = 10000, .eos = &.{}, .drafts = true });
        defer s.deinit(a);
        try engine.addStream(&s);
        try engine.step();
        try std.testing.expect(s.depth.?.ms.?.empty());
        try engine.step();
        try std.testing.expectEqual(@as(?f64, 96), s.depth.?.ms.?.get(7));
        engine.rule.most_drafts = 0;
        s.graft_room = 0;
        s.priced = .{ .choice = 0, .ms = 96 };
        try engine.step();
        try std.testing.expectEqual(@as(?f64, 96), s.depth.?.ms.?.get(7));
        try std.testing.expectEqual(@as(?f64, 54), s.round_over);
        try std.testing.expectEqual(@as(f64, 0), s.measured[0].median());
        engine.rule.most_drafts = 7;
        try engine.step();
        try std.testing.expectEqual(@as(?f64, 42), s.depth.?.ms.?.get(0));
        engine.rule.most_drafts = 0;
        try s.force.appendSlice(a, &.{ 3, 4, 5 });
        try engine.step();
        try std.testing.expectEqual(@as(?f64, 96), s.depth.?.ms.?.get(7));
        try std.testing.expectEqual(@as(?f64, 42), s.depth.?.ms.?.get(0));
        try std.testing.expectEqual(@as(f64, 0), s.draft_ms);
        engine.rule.most_drafts = 7;
        try engine.step();
        try std.testing.expectEqual(@as(?f64, 42), s.depth.?.ms.?.get(0));
        engine.release(&s);
        try std.testing.expectEqual(@as(f64, 0), s.draft_ms);
    }
}

test "family cost: grammar rounds use late drafting and plain costs exclude the next proposal" {
    const old = lanes.grammar_fake.install();
    defer _ = lanes.grammar.swapLibrary(old);
    var compiler = lanes.grammar_fake.compiler();
    for ([_]bool{ false, true }) |early| {
        const grammar = try compiler.constraint(a, std.testing.io, .regex, "a+", null);
        defer grammar.deinit(a);
        var cfg = try config(early);
        defer cfg.deinit(a);
        var r: Replay = .{ .accepted = 6 };
        defer r.target.deinit();
        var vt: be.Backend.VTable = undefined;
        var engine = lanes.Engine.init(a, &cfg, r.backend(&vt), r.clock());
        defer engine.deinit();
        var s = try lanes.Stream.init(a, .{ .id = "grammar", .prompt = &.{ 1, 2 }, .max_new = 10000, .eos = &.{}, .drafts = true, .grammar = grammar });
        defer s.deinit(a);
        try engine.addStream(&s);
        engine.rule.most_drafts = 0;
        try engine.step();
        try engine.step();
        try std.testing.expect(!s.finished);
        try std.testing.expectEqual(@as(?f64, 42), s.depth.?.ms.?.get(0));
        engine.rule.most_drafts = 7;
        try engine.step();
        try std.testing.expectEqual(@as(?f64, 42), s.depth.?.ms.?.get(0));
        try std.testing.expectEqual(@as(usize, 0), r.early_calls);
    }
}

test "family cost: a proposal from a shared round keeps its cost when the stream runs alone" {
    var cfg = try config(false);
    defer cfg.deinit(a);
    cfg.family_streams = true;
    var r: Replay = .{ .accepted = 6 };
    defer r.target.deinit();
    var vt: be.Backend.VTable = undefined;
    var engine = lanes.Engine.init(a, &cfg, r.backend(&vt), r.clock());
    defer engine.deinit();
    var streams: [2]lanes.Stream = undefined;
    var made: usize = 0;
    defer for (streams[0..made]) |*s| s.deinit(a);
    for (&streams) |*s| {
        s.* = try lanes.Stream.init(a, .{ .id = "shared", .prompt = &.{ 1, 2 }, .max_new = 10000, .eos = &.{}, .drafts = true });
        made += 1;
        try engine.addStream(s);
    }
    try engine.step();
    streams[1].paused = true;
    try engine.step();
    const s = &streams[0];
    try std.testing.expectEqual(@as(?f64, 96), s.depth.?.ms.?.get(@intCast(r.target.last_rows - 1)));
}

test "family cost: untimed rounds never sample the clock" {
    var cfg = try config(false);
    defer cfg.deinit(a);
    cfg.timed = false;
    var r: Replay = .{ .accepted = 6 };
    defer r.target.deinit();
    var vt: be.Backend.VTable = undefined;
    var engine = lanes.Engine.init(a, &cfg, r.backend(&vt), r.clock());
    defer engine.deinit();
    var s = try lanes.Stream.init(a, .{ .id = "untimed", .prompt = &.{ 1, 2 }, .max_new = 10000, .eos = &.{}, .drafts = true });
    defer s.deinit(a);
    try engine.addStream(&s);
    for (0..8) |_| try engine.step();
    try std.testing.expectEqual(@as(usize, 0), r.clock_reads);
}
