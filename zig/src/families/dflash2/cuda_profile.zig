const std = @import("std");
const cuda = @import("cuda");

pub const Phase = enum { project, norm, embed, conv, rope, attention, swiglu, head, lattice_gpu, lattice_fallback, download, topk, logsumexp, selector };
const count = @typeInfo(Phase).@"enum".field_names.len;
const Pair = struct { start: cuda.Event, end: cuda.Event, phase: Phase = .project };
const Total = struct { calls: usize = 0, wall_ms: f64 = 0, stream_ms: f64 = 0 };
pub const Ticket = struct { at: i96, phase: Phase, event: ?usize };
pub const Kind = enum { ingest, proposal };

pub const Profile = struct {
    io: std.Io,
    pairs: [160]Pair = undefined,
    used: usize = 0,
    totals: [count]Total = @splat(.{}),
    active: bool = false,
    seen: [2]usize = .{ 0, 0 },
    started: i96 = 0,
    kind: Kind = .proposal,

    pub fn init(d: *const cuda.Driver, io: std.Io) !Profile {
        var p: Profile = .{ .io = io };
        var made: usize = 0;
        errdefer for (p.pairs[0..made]) |*pair| {
            pair.start.deinit();
            pair.end.deinit();
        };
        for (&p.pairs) |*pair| {
            var start = try cuda.Event.init(d, true);
            errdefer start.deinit();
            pair.* = .{ .start = start, .end = try cuda.Event.init(d, true) };
            made += 1;
        }
        return p;
    }
    pub fn deinit(p: *Profile) void {
        for (&p.pairs) |*pair| {
            pair.start.deinit();
            pair.end.deinit();
        }
    }
    fn now(p: *Profile) i96 {
        return std.Io.Clock.awake.now(p.io).toNanoseconds();
    }
    pub fn begin(p: *Profile, kind: Kind) void {
        p.kind = kind;
        const i = @backingInt(kind);
        p.active = p.seen[i] < (if (kind == .proposal) @as(usize, 16) else 64) and
            (kind == .proposal or p.seen[@backingInt(Kind.proposal)] > 0);
        if (!p.active) return;
        p.used = 0;
        p.totals = @splat(.{});
        p.started = p.now();
    }
    pub fn enter(p: *Profile, s: cuda.Stream, phase: Phase, gpu: bool) !?Ticket {
        if (!p.active) return null;
        const at = p.now();
        var event: ?usize = null;
        if (gpu) {
            if (p.used == p.pairs.len) return error.DraftProfileOverflow;
            event = p.used;
            p.used += 1;
            p.pairs[event.?].phase = phase;
            try p.pairs[event.?].start.record(s);
        }
        return .{ .at = at, .phase = phase, .event = event };
    }
    pub fn leave(p: *Profile, s: cuda.Stream, ticket: ?Ticket) void {
        const t = ticket orelse return;
        if (t.event) |i| p.pairs[i].end.record(s) catch {
            p.active = false;
            return;
        };
        const total = &p.totals[@backingInt(t.phase)];
        total.calls += 1;
        total.wall_ms += @as(f64, @floatFromInt(p.now() - t.at)) / 1e6;
    }
    pub fn finish(p: *Profile, s: cuda.Stream, context: u64, nodes: u32) !void {
        if (!p.active) return;
        defer p.active = false;
        try s.synchronize();
        const wall = @as(f64, @floatFromInt(p.now() - p.started)) / 1e6;
        const i = @backingInt(p.kind);
        p.seen[i] += 1;
        for (p.pairs[0..p.used]) |pair| {
            p.totals[@backingInt(pair.phase)].stream_ms += try cuda.Event.elapsedMs(pair.start, pair.end);
        }
        std.log.info("DFLASH_PROFILE kind={s} sample={d} context={d} nodes={d} wall_ms={d:.3}", .{ @tagName(p.kind), p.seen[i], context, nodes, wall });
        inline for (@typeInfo(Phase).@"enum".field_names, 0..) |field, j| {
            const t = p.totals[j];
            if (t.calls > 0 or (p.kind == .proposal and (j == @intFromEnum(Phase.topk) or j == @intFromEnum(Phase.logsumexp)))) std.log.info("DFLASH_PHASE kind={s} sample={d} phase={s} calls={d} wall_ms={d:.3} stream_span_ms={d:.3}", .{ @tagName(p.kind), p.seen[i], field, t.calls, t.wall_ms, t.stream_ms });
        }
    }
};
