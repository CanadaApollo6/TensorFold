const std = @import("std");
pub const capacity = 128;
pub const frequency_bytes = 64 * @sizeOf(f32);
pub const device_bytes = frequency_bytes + capacity * @sizeOf(i32);
pub fn frequencies(theta: f64) [64]f32 {
    var out: [64]f32 = undefined;
    for (&out, 0..) |*v, i| v.* = @floatCast(1 / std.math.pow(f64, theta, @as(f64, @floatFromInt(i)) / 64));
    return out;
}
pub const Positions = struct {
    values: [capacity]i32 = undefined,
    len: usize = 0,
    pub const Change = enum { same, first, replace };
    pub fn reset(p: *Positions) void {
        p.len = 0;
    }
    pub fn prepare(p: *Positions, positions: []const u64) !Change {
        if (positions.len == 0 or positions.len > capacity) return error.PastContext;
        for (positions) |v| if (v > std.math.maxInt(i32)) return error.PastContext;
        const same = if (p.len == positions.len) blk: {
            for (positions, p.values[0..p.len]) |v, cached| if (v != cached) break :blk false;
            break :blk true;
        } else false;
        if (same) return .same;
        const change: Change = if (p.len == 0) .first else .replace;
        for (positions, 0..) |v, i| p.values[i] = @intCast(v);
        p.len = positions.len;
        return change;
    }
};
test "RoPE cached frequencies retain the old FP64 pow and FP32 rounding" {
    for ([_]f64{ 10000, 1000000, 10000000, 500000 }) |theta| {
        const inv = frequencies(theta);
        for (inv, 0..) |v, i| {
            const old: f32 = @floatCast(1 / std.math.pow(f64, theta, @as(f64, @floatFromInt(i)) / 64));
            try std.testing.expectEqual(@as(u32, @bitCast(old)), @as(u32, @bitCast(v)));
        }
    }
}
test "RoPE position reuse ends at frame reset and validates before updating" {
    var p: Positions = .{};
    try std.testing.expectEqual(Positions.Change.first, try p.prepare(&.{ 130, 131, 132 }));
    try std.testing.expectEqual(Positions.Change.same, try p.prepare(&.{ 130, 131, 132 }));
    try std.testing.expectEqual(Positions.Change.replace, try p.prepare(&.{ 2047, 2048, 2049 }));
    try std.testing.expectError(error.PastContext, p.prepare(&.{std.math.maxInt(u64)}));
    try std.testing.expectEqual(Positions.Change.same, try p.prepare(&.{ 2047, 2048, 2049 }));
    p.reset();
    try std.testing.expectEqual(Positions.Change.first, try p.prepare(&.{2049}));
    try std.testing.expectEqual(Positions.Change.replace, try p.prepare(&.{std.math.maxInt(i32)}));
    try std.testing.expectError(error.PastContext, p.prepare(&.{}));
}
