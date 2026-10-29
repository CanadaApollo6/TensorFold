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
    context: usize = 128,
    tap_buffer: u64 = 0,
    logits: u64 = 0,
    token: u32 = 42,
    reject_at: ?usize = null,
    eos_at: ?usize = null,
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
    pub fn propose(d: *Attachment, _: u32, depth: u32) !void {
        d.count = @min(depth, 7);
    }
};

test "fully accepted rounds and later rejection preserve committed drafter position" {
    var m: Model = .{ .gpa = a };
    var runtime: Runtime = .{};
    var attachment: Attachment = .{ .m = &m, .r = &runtime };
    var backend: @import("backend.zig").Adapter(Model, Attachment) = .{ .m = &m, .drafter = &attachment };
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
