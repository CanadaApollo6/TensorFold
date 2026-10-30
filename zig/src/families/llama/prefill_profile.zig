//! TF_LLAMA_PREFILL_PROFILE records prompt phase spans and run totals with CUDA events.
const std = @import("std");
const cuda = @import("cuda");

pub const Phase = enum { embed, norm, q, k, v, rope, attention, o, gate, up, swiglu, down, residual, head };
const count = @typeInfo(Phase).@"enum".field_names.len;
const gemm = [_]Phase{ .q, .k, .v, .o, .gate, .up, .down, .head };
const Pair = struct { start: cuda.Event, end: cuda.Event, phase: Phase = .embed };

const Capture = struct {
    const Call = *const fn () callconv(.c) c_int;
    d: *const cuda.Driver,
    start: Call,
    stop: Call,
    state: enum { waiting, active, complete } = .waiting,

    fn init(d: *const cuda.Driver) !?Capture {
        if (std.c.getenv("TF_LLAMA_PREFILL_NCU") == null) return null;
        var lib = d.lib;
        return .{ .d = d, .start = lib.lookup(Call, "cuProfilerStart") orelse return error.MissingSymbol, .stop = lib.lookup(Call, "cuProfilerStop") orelse return error.MissingSymbol };
    }

    fn begin(c: *Capture, run: usize, position: usize, rows: usize) !void {
        if (c.state != .waiting or run != 2 or position != 59904 or rows != 96) return;
        try c.d.check(c.start(), "cuProfilerStart");
        c.state = .active;
        std.log.info("LLAMA_PREFILL_NCU run=2 position=59904 rows=96 state=START", .{});
    }

    fn end(c: *Capture) !void {
        if (c.state != .active) return;
        try c.d.check(c.stop(), "cuProfilerStop");
        c.state = .complete;
        std.log.info("LLAMA_PREFILL_NCU run=2 position=59904 rows=96 state=STOP", .{});
    }
};

pub fn enabled() bool {
    return std.c.getenv("TF_LLAMA_PREFILL_PROFILE") != null;
}

pub const Profile = struct {
    capture: ?Capture = null,
    pairs: [1024]Pair = undefined, // a chunk records 2 + 13 a layer + 2 (52 layers: 680)
    used: usize = 0,
    span: [2]cuda.Event = undefined,
    run: usize = 0,
    position: usize = 0,
    rows: usize = 0,
    tokens: usize = 0,
    chunks: usize = 0,
    span_ms: f64 = 0,
    calls: [count]usize = @splat(0),
    ms: [count]f64 = @splat(0),

    pub fn init(d: *const cuda.Driver) !Profile {
        var p: Profile = .{ .capture = try Capture.init(d) };
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
        p.span[0] = try cuda.Event.init(d, true);
        errdefer p.span[0].deinit();
        p.span[1] = try cuda.Event.init(d, true);
        return p;
    }

    pub fn deinit(p: *Profile) void {
        if (p.capture) |*c| c.end() catch {};
        p.flush();
        for (&p.pairs) |*pair| {
            pair.start.deinit();
            pair.end.deinit();
        }
        for (&p.span) |*e| e.deinit();
    }

    pub fn begin(p: *Profile, s: cuda.Stream, position: usize, rows: usize) !void {
        if (position == 0) {
            p.flush();
            p.run += 1;
        }
        p.used = 0;
        p.position = position;
        p.rows = rows;
        if (p.capture) |*c| try c.begin(p.run, position, rows);
        try p.span[0].record(s);
    }

    /// Waits for the chunk (the next chunk's metadata upload waits for it anyway) and folds its spans into the run.
    pub fn end(p: *Profile, s: cuda.Stream, last: bool) !void {
        try p.span[1].record(s);
        try s.synchronize();
        if (p.capture) |*c| try c.end();
        var chunk: [count]f64 = @splat(0);
        for (p.pairs[0..p.used]) |pair| {
            chunk[@backingInt(pair.phase)] += try cuda.Event.elapsedMs(pair.start, pair.end);
            p.calls[@backingInt(pair.phase)] += 1;
        }
        const span: f64 = try cuda.Event.elapsedMs(p.span[0], p.span[1]);
        var busy: f64 = 0;
        for (chunk, 0..) |ms, i| {
            busy += ms;
            p.ms[i] += ms;
        }
        var matmul: f64 = 0;
        for (gemm) |phase| matmul += chunk[@backingInt(phase)];
        p.span_ms += span;
        p.tokens += p.rows;
        p.chunks += 1;
        std.log.info("LLAMA_PREFILL_CHUNK run={d} position={d} rows={d} span_ms={d:.3} gemm_ms={d:.3} attention_ms={d:.3} other_ms={d:.3} idle_ms={d:.3}", .{ p.run, p.position, p.rows, span, matmul, chunk[@backingInt(Phase.attention)], busy - matmul - chunk[@backingInt(Phase.attention)], span - busy });
        if (last) p.flush();
    }

    fn flush(p: *Profile) void {
        if (p.chunks == 0) return;
        var busy: f64 = 0;
        for (p.ms) |ms| busy += ms;
        std.log.info("LLAMA_PREFILL_RUN run={d} tokens={d} chunks={d} span_ms={d:.3} busy_ms={d:.3} idle_ms={d:.3}", .{ p.run, p.tokens, p.chunks, p.span_ms, busy, p.span_ms - busy });
        const tokens: f64 = @floatFromInt(p.tokens);
        inline for (@typeInfo(Phase).@"enum".field_names, 0..) |field, j| {
            std.log.info("LLAMA_PREFILL_PHASE run={d} phase={s} calls={d} ms={d:.3} us_per_token={d:.3}", .{ p.run, field, p.calls[j], p.ms[j], p.ms[j] * 1000 / tokens });
        }
        p.tokens = 0;
        p.chunks = 0;
        p.span_ms = 0;
        p.calls = @splat(0);
        p.ms = @splat(0);
    }
};

pub fn enter(p: ?*Profile, s: cuda.Stream, phase: Phase) !?usize {
    const q = p orelse return null;
    if (q.used == q.pairs.len) return error.PrefillProfileOverflow;
    const i = q.used;
    q.used += 1;
    q.pairs[i].phase = phase;
    try q.pairs[i].start.record(s);
    return i;
}

pub fn leave(p: ?*Profile, s: cuda.Stream, ticket: ?usize) !void {
    const q = p orelse return;
    try q.pairs[ticket orelse return].end.record(s);
}

test "Nsight range opens once at the warm 60K tail and closes once" {
    const Fake = struct {
        var starts: usize = 0;
        var stops: usize = 0;
        fn start() callconv(.c) c_int {
            starts += 1;
            return 0;
        }
        fn stop() callconv(.c) c_int {
            stops += 1;
            return 0;
        }
    };
    Fake.starts = 0;
    Fake.stops = 0;
    var d: cuda.Driver = undefined;
    var c: Capture = .{ .d = &d, .start = Fake.start, .stop = Fake.stop };
    try c.begin(1, 59904, 96);
    try c.begin(2, 59776, 128);
    try c.begin(2, 59904, 128);
    try c.end();
    try std.testing.expectEqual(@as(usize, 0), Fake.starts);
    try std.testing.expectEqual(@as(usize, 0), Fake.stops);
    try c.begin(2, 59904, 96);
    try c.begin(2, 59904, 96);
    try std.testing.expectEqual(@as(usize, 1), Fake.starts);
    try c.end();
    try c.end();
    try c.begin(3, 59904, 96);
    try std.testing.expectEqual(@as(usize, 1), Fake.starts);
    try std.testing.expectEqual(@as(usize, 1), Fake.stops);
}
