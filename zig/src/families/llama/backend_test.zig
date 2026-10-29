const std = @import("std");
const lanes = @import("lanes");
const a = std.testing.allocator;
const Model = struct {
    gpa: std.mem.Allocator,
    w: struct { config: struct { bos: u32 = 1, vocab: usize = 97 } = .{} } = .{},
    k: struct { draws: struct { penalize_fn: ?usize = 1 } = .{} } = .{},
    s: struct {
        pub fn synchronize(_: @This()) !void {}
    } = .{},
    context: usize = 8192,
    visits: [8192]u8 = @splat(0),
    last_head: bool = false,
    chunks: usize = 0,
    tap_buffer: u64 = 0,
    prompt_rows: usize = 128,
    logits: u64 = 0,
    token: u32 = 42,
    reject_at: ?usize = null,
    eos_at: ?usize = null,
    pub fn prefill(m: *Model, tokens: []const u32, position: usize, last: bool) !void {
        for (0..tokens.len) |row| m.visits[position + row] += 1;
        m.last_head = last;
        m.chunks += 1;
    }
    pub fn step(_: *Model, _: u32, _: usize, _: bool) !void {}
    pub fn batch(_: *Model, _: []const u32, _: usize) !void {}
    pub const Draw = struct { position: u32, sampling: ?lanes.Sampling, penalties: lanes.Penalties, history: []const u32, pending: ?u32 };
    pub fn pick(m: *Model, draw: Draw) !u32 {
        if (m.eos_at == draw.position) return 44;
        return if (m.reject_at == draw.position) 43 else m.token;
    }
};

const Runtime = struct {
    graph: struct { config: struct { window: usize = 2048 } = .{} } = .{},
    context: struct { cache: usize = 1, begin: usize = 0, end: usize = 0 } = .{},
    pub fn reset(r: *Runtime) void {
        r.context = .{};
    }
};
const Attachment = struct {
    m: *Model,
    r: *Runtime,
    held: [7]u32 = @splat(42),
    count: usize = 0,
    pub fn capture(_: *Attachment, _: usize) void {}
    pub fn ingest(d: *Attachment, _: usize, position: usize) !void {
        if (position != d.r.context.end) return error.BadCommittedTaps;
        d.r.context.end = position + 1;
    }
    pub fn reuse(_: *Attachment, _: usize, _: usize) !usize {
        return 0;
    }
    pub fn restore(_: *Attachment, _: usize, _: usize) !void {}
    pub fn propose(d: *Attachment, _: u32, depth: u32) !void {
        d.count = @min(depth, 7);
    }
};

test "fully accepted rounds and later rejection preserve committed drafter position" {
    var m: Model = .{ .gpa = a };
    var runtime: Runtime = .{};
    var attachment: Attachment = .{ .m = &m, .r = &runtime };
    var backend: @import("backend.zig").Adapter(Model, Attachment) = .{ .m = &m, .drafter = &attachment };
    defer backend.cached.deinit(a);
    var cfg = try lanes.Config.init(a, backend.facts(), 8, 7);
    defer cfg.deinit(a);
    var clock: lanes.fake.FixedClock = .{};
    var engine = lanes.Engine.init(a, &cfg, backend.backend(), clock.clock());
    defer engine.deinit();
    var stream = try lanes.Stream.init(a, .{ .id = "repro", .prompt = &.{ 1, 2 }, .max_new = 32, .eos = &.{}, .drafts = true });
    defer stream.deinit(a);
    try engine.addStream(&stream);
    try engine.step();
    std.debug.print("after full acceptance target={d} draft={d} accepted={d} proposed={d}\n", .{ backend.len, runtime.context.end, stream.accepted, stream.drafted });
    try std.testing.expect(stream.accepted > 0);
    try std.testing.expectEqual(stream.accepted, stream.drafted);
    try std.testing.expectEqual(backend.len, runtime.context.end);
    try engine.step();
    try std.testing.expectEqual(backend.len, runtime.context.end);
    m.token = 43;
    try engine.step();
    try std.testing.expect(stream.drafted > stream.accepted);
    try std.testing.expectEqual(backend.len, runtime.context.end);
    while (engine.activeCount() > 0) try engine.step();
    try std.testing.expectEqual(@as(usize, 32), stream.emitted().len);
}

test "target draws and EOS or length at every output offset preserve committed taps" {
    for (0..3) |kind| for (0..8) |offset| {
        var m: Model = .{ .gpa = a };
        var runtime: Runtime = .{};
        var attachment: Attachment = .{ .m = &m, .r = &runtime };
        var backend: @import("backend.zig").Adapter(Model, Attachment) = .{ .m = &m, .drafter = &attachment };
        defer backend.cached.deinit(a);
        var cfg = try lanes.Config.init(a, backend.facts(), 8, 7);
        defer cfg.deinit(a);
        var clock: lanes.fake.FixedClock = .{};
        var engine = lanes.Engine.init(a, &cfg, backend.backend(), clock.clock());
        defer engine.deinit();
        var stream = try lanes.Stream.init(a, .{ .id = "offset", .prompt = &.{ 1, 2 }, .max_new = 64, .eos = &.{44}, .drafts = true });
        defer stream.deinit(a);
        try engine.addStream(&stream);
        try engine.step();
        try std.testing.expectEqual(stream.accepted, stream.drafted);
        try std.testing.expect(stream.accepted > 0);
        const position = backend.len + 1 + offset;
        if (kind == 0) m.reject_at = position;
        if (kind == 1) m.eos_at = position;
        if (kind == 2) stream.max_new = @intCast(stream.emitted().len + 1 + offset);
        while (engine.activeCount() > 0) {
            try engine.step();
            if (backend.owner != null) try std.testing.expectEqual(backend.len, runtime.context.end);
        }
        if (kind == 0) {
            try std.testing.expectEqual(@as(u32, 43), stream.emitted()[position - 2]);
        }
        if (kind == 1) {
            try std.testing.expectEqual(lanes.stream.Reason.stop, stream.reason);
            try std.testing.expectEqual(@as(u32, 44), stream.emitted()[stream.emitted().len - 1]);
        } else try std.testing.expectEqual(lanes.stream.Reason.length, stream.reason);
    };
}

test "prefill chunks yield progress, retain cancelled prefixes and never refill cached rows" {
    var m: Model = .{ .gpa = a, .prompt_rows = 128 };
    var adapter: @import("backend.zig").Adapter(Model, Attachment) = .{ .m = &m };
    defer adapter.cached.deinit(a);
    const b = adapter.backend();
    var tokens: [258]u32 = @splat(2);
    tokens[0] = 1;
    var s = try lanes.Stream.init(a, .{ .id = "chunks", .prompt = &tokens, .max_new = 1, .eos = &.{} });
    defer s.deinit(a);
    try std.testing.expect(!try b.vtable.prefill_step.?(b.ptr, &s));
    try std.testing.expectEqual(@as(u64, 128), s.fill_at.load(.monotonic));
    try std.testing.expect(!m.last_head);
    b.release(&s);
    try std.testing.expectEqual(@as(usize, 1), m.chunks);
    try std.testing.expectEqual(@as(u8, 0), m.visits[128]);
    try std.testing.expect(!try b.vtable.prefill_step.?(b.ptr, &s));
    try std.testing.expectEqual(@as(u64, 128), s.cached);
    try std.testing.expectEqual(@as(u64, 256), s.fill_at.load(.monotonic));
    try std.testing.expect(try b.vtable.prefill_step.?(b.ptr, &s));
    try std.testing.expect(m.last_head);
    for (m.visits[0..258]) |count| try std.testing.expectEqual(@as(u8, 1), count);
    b.release(&s);
    tokens[257] = 3;
    var next = try lanes.Stream.init(a, .{ .id = "continued", .prompt = &tokens, .max_new = 1, .eos = &.{} });
    defer next.deinit(a);
    try b.prefill(&next);
    try std.testing.expectEqual(@as(u64, 257), next.cached);
    for (m.visits[0..257]) |count| try std.testing.expectEqual(@as(u8, 1), count);
    try std.testing.expectEqual(@as(u8, 2), m.visits[257]);
}

test "drafter captures absolute rows across the context window and chunk seam" {
    var m: Model = .{ .gpa = a, .prompt_rows = 128 };
    var r: Runtime = .{};
    r.graph.config.window = 130;
    var d: Attachment = .{ .m = &m, .r = &r };
    var adapter: @import("backend.zig").Adapter(Model, Attachment) = .{ .m = &m, .drafter = &d };
    defer adapter.cached.deinit(a);
    var tokens: [258]u32 = @splat(2);
    tokens[0] = 1;
    var s = try lanes.Stream.init(a, .{ .id = "taps", .prompt = &tokens, .max_new = 1, .eos = &.{}, .drafts = true });
    defer s.deinit(a);
    const b = adapter.backend();
    try std.testing.expect(!try b.vtable.prefill_step.?(b.ptr, &s));
    try std.testing.expectEqual(@as(usize, 129), r.context.end);
    try std.testing.expect(!try b.vtable.prefill_step.?(b.ptr, &s));
    try std.testing.expectEqual(@as(usize, 256), r.context.end);
    try std.testing.expect(try b.vtable.prefill_step.?(b.ptr, &s));
    try std.testing.expectEqual(@as(usize, 258), r.context.end);
}
