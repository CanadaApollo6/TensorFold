const std = @import("std");
const lanes = @import("lanes");
const a = std.testing.allocator;
const window = 2048;
const Config = struct {
    window: usize = window,
    pub fn tapWidth(_: Config) usize {
        return 4;
    }
};
const Driver = struct {
    api: struct { cuMemcpyDtoDAsync_v2: *const fn (u64, u64, usize, usize) u32 = copy } = .{},
    fn copy(dst: u64, src: u64, n: usize, _: usize) u32 {
        const x: [*]u8 = @ptrFromInt(dst);
        const y: [*]const u8 = @ptrFromInt(src);
        @memcpy(x[0..n], y[0..n]);
        return 0;
    }
    pub fn check(_: Driver, result: u32, _: []const u8) !void {
        try std.testing.expectEqual(@as(u32, 0), result);
    }
};
const Target = struct {
    gpa: std.mem.Allocator = a,
    w: struct { config: struct { bos: u32 = 1, vocab: usize = 97 } = .{} } = .{},
    k: struct { draws: struct { penalize_fn: ?usize = null } = .{} } = .{},
    s: struct {
        handle: usize = 0,
        pub fn synchronize(_: @This()) !void {}
    } = .{},
    d: Driver = .{},
    context: usize = 24000,
    prompt_rows: usize = 128,
    tap_buffer: u64 = 0,
    logits: u64 = 0,
    pub fn prefill(m: *Target, tokens: []const u32, pos: usize, _: bool) !void {
        const out: [*][2]u32 = @ptrFromInt(m.tap_buffer);
        for (tokens, 0..) |token, row| out[row] = .{ @intCast(pos + row), token };
    }
    pub fn step(m: *Target, token: u32, pos: usize, last: bool) !void {
        try m.prefill(&.{token}, pos, last);
    }
    pub fn batch(m: *Target, tokens: []const u32, pos: usize) !void {
        try m.prefill(tokens, pos, true);
    }
    pub const Draw = struct { position: u32, sampling: ?lanes.Sampling, penalties: lanes.Penalties, history: []const u32, pending: ?u32 };
    pub fn pick(_: *Target, _: Draw) !u32 {
        return 42;
    }
};
const Runtime = struct {
    graph: struct { config: Config = .{} } = .{},
    context: struct { cache: usize = 1, begin: usize = 0, end: usize = 0 } = .{},
    expected: []const u32 = &.{},
    ring: [window][2]u32 = @splat(.{ 0, 0 }),
    restored: usize = 0,
    pub fn reset(r: *Runtime) void {
        r.context = .{};
        r.restored = 0;
    }
    pub fn ingest(r: *Runtime, ptr: u64, pos: usize) !void {
        if (pos != r.context.end) return error.BadCommittedTaps;
        const tap: *const [2]u32 = @ptrFromInt(ptr);
        if (tap[0] != pos or tap[1] != r.expected[pos]) return error.StaleTap;
        r.ring[pos % window] = tap.*;
        r.context.end = pos + 1;
        r.context.begin = @max(r.context.begin, (pos + 1) -| (window - 1));
        r.restored += 1;
    }
    pub fn propose(_: *Runtime, _: u32, _: u32) !@import("../dflash2/selector.zig").Tree {
        return error.UnexpectedProposal;
    }
};
const Attachment = @import("draft.zig").Binding(Target, Runtime);
const Adapter = @import("backend.zig").Adapter(Target, Attachment);
const Fixture = struct {
    m: Target = .{},
    r: Runtime = .{},
    taps: [1024][2]u32 = undefined,
    history: [window][2]u32 = @splat(.{ std.math.maxInt(u32), 0 }),
    d: Attachment = undefined,
    b: Adapter = undefined,
    fn init(f: *Fixture) void {
        f.d = .{ .m = &f.m, .r = &f.r, .captures = .{ .d = undefined, .ptr = @intFromPtr(&f.taps), .len = @sizeOf(@TypeOf(f.taps)) }, .history = .{ .d = undefined, .ptr = @intFromPtr(&f.history), .len = @sizeOf(@TypeOf(f.history)) } };
        f.b = .{ .m = &f.m, .drafter = &f.d };
    }
    fn deinit(f: *Fixture) void {
        f.b.cached.deinit(a);
    }
    fn fill(f: *Fixture, prompt: []const u32, stop: ?usize) !usize {
        f.r.expected = prompt;
        var s = try lanes.Stream.init(a, .{ .id = "tap-history", .prompt = prompt, .max_new = 1, .eos = &.{} });
        defer s.deinit(a);
        const b = f.b.backend();
        defer b.release(&s);
        while (true) {
            const done = try b.vtable.prefill_step.?(b.ptr, &s);
            if (done or (stop != null and s.fill_at.load(.monotonic) >= stop.?)) break;
        }
        if (stop == null) {
            try std.testing.expectEqual(prompt.len, f.r.context.end);
            for (f.r.context.begin..f.r.context.end) |pos| {
                try std.testing.expectEqualSlices(u32, &.{ @intCast(pos), prompt[pos] }, &f.r.ring[pos % window]);
            }
        }
        return @intCast(s.cached);
    }
};
fn makePrompt() [22000]u32 {
    var tokens: [22000]u32 = undefined;
    for (&tokens, 0..) |*token, pos| token.* = @intCast(pos + 1);
    return tokens;
}

test "retained taps shorten a freshly prefilled 20K history by one token" {
    const tokens = makePrompt();
    var f: Fixture = .{};
    f.init();
    defer f.deinit();
    _ = try f.fill(tokens[0..20000], null);
    try std.testing.expectEqual(@as(usize, 17952), try f.fill(tokens[0..19999], null));
}

test "retained taps cancel a rewritten long history then open its shorter common prefix" {
    var tokens = makePrompt();
    var f: Fixture = .{};
    f.init();
    defer f.deinit();
    _ = try f.fill(tokens[0..20000], null);
    for (tokens[10000..]) |*token| token.* += 30000;
    try std.testing.expectEqual(@as(usize, 10000), try f.fill(tokens[0..20000], 10128));
    try std.testing.expectEqual(@as(usize, 8953), try f.fill(tokens[0..11000], null));
}

test "retained taps cancel initial 20K fill at 18048 then submit 19K prefix" {
    const tokens = makePrompt();
    var f: Fixture = .{};
    f.init();
    defer f.deinit();
    _ = try f.fill(tokens[0..20000], 18048);
    try std.testing.expectEqual(@as(usize, 16953), try f.fill(tokens[0..19000], null));
}

test "retained taps long continuation restores an overlapping tail without target replay" {
    const tokens = makePrompt();
    var f: Fixture = .{};
    f.init();
    defer f.deinit();
    _ = try f.fill(tokens[0..19000], null);
    try std.testing.expectEqual(@as(usize, 19000), try f.fill(tokens[0..20000], null));
    try std.testing.expectEqual(@as(usize, 2047), f.r.restored);
}

test "retained taps cancel then shorten at the inferred window boundary" {
    const tokens = makePrompt();
    var f: Fixture = .{};
    f.init();
    defer f.deinit();
    _ = try f.fill(tokens[0..20000], 18048);
    try std.testing.expectEqual(@as(usize, 16000), try f.fill(tokens[0..18047], null));
}

test "retained taps cancel then extend restores only initialized overlapping positions" {
    const tokens = makePrompt();
    var f: Fixture = .{};
    f.init();
    defer f.deinit();
    _ = try f.fill(tokens[0..20000], 18048);
    try std.testing.expectEqual(@as(usize, 18048), try f.fill(tokens[0..20050], null));
}

test "retained taps unrelated prompt sharing a short prefix discards old identity" {
    var tokens = makePrompt();
    var f: Fixture = .{};
    f.init();
    defer f.deinit();
    _ = try f.fill(tokens[0..20000], null);
    for (tokens[32..]) |*token| token.* += 30000;
    try std.testing.expectEqual(@as(usize, 32), try f.fill(tokens[0..20000], 160));
    try std.testing.expectEqual(@as(usize, 0), try f.fill(tokens[0..2000], null));
}

test "retained taps repeated wrap truncation cancellation and rewrite cycles match cold rows" {
    var tokens = makePrompt();
    var f: Fixture = .{};
    f.init();
    defer f.deinit();
    for (0..8) |cycle| {
        _ = try f.fill(tokens[0..20000], null);
        _ = try f.fill(tokens[0..21000], null);
        for (tokens[18000..]) |*token| token.* += @intCast(cycle + 1);
        _ = try f.fill(tokens[0..21000], 18128);
        _ = try f.fill(tokens[0..19000], null);
        _ = try f.fill(tokens[0..18999], null);
    }
}

test "retained taps refuse uncovered restoration before reading a ring slot" {
    var f: Fixture = .{};
    f.init();
    defer f.deinit();
    try std.testing.expectError(error.MissingTapHistory, f.d.restore(100, 101));
    try std.testing.expectEqual(@as(usize, 0), f.r.restored);
}
