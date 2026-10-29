const std = @import("std");
pub const Room = struct { ctx: *anyopaque, read: *const fn (*anyopaque) ?u64 };

pub fn check(peak: usize, room: ?Room) !void {
    const probe = room orelse return;
    if (peak > (probe.read(probe.ctx) orelse 0)) return error.LlamaMemoryBudget;
}

test "configured cap admits target but refuses attachment resident or loader peak" {
    const Probe = struct {
        left: ?u64,
        fn read(p: *anyopaque) ?u64 {
            return @as(*@This(), @ptrCast(@alignCast(p))).left;
        }
    };
    var probe: Probe = .{ .left = 1000 };
    const room: Room = .{ .ctx = &probe, .read = Probe.read };
    try check(900, room);
    probe.left = 100;
    try std.testing.expectError(error.LlamaMemoryBudget, check(101, room));
    try check(100, room);
    try std.testing.expectError(error.LlamaMemoryBudget, check(@max(90, 120), room));
    probe.left = null;
    try std.testing.expectError(error.LlamaMemoryBudget, check(1, room));
}
