const std = @import("std");
const registry = @import("registry.zig");
const limits = @import("limits.zig");

pub const records = blk: {
    var out: [6]registry.CudaRecord = undefined;
    const symbols = [_][]const u8{ "tf_kv4_pack", "tf_kv4_unpack", "tf_kv4_partial128", "tf_kv4_partial256", "tf_kv4_merge", "tf_kv4_merge" };
    const registers = [_]u32{ 21, 16, 56, 80, 36, 36 };
    for (symbols, registers, 0..) |symbol, regs, i| {
        var usage: [5]limits.Usage = @splat(.{ .regs = 0, .static = 0 });
        usage[1] = .{ .regs = regs, .static = 0 };
        out[i] = .{ .launch = .{ .name = std.fmt.comptimePrint("kv4 {s} threads {d}", .{ symbol, if (i == 5) 256 else 128 }), .image = "kv4", .symbol = symbol, .threads = if (i == 5) 256 else 128, .min_sm = 86, .max_sm = 86 }, .usage = usage, .min_nvcc = .{ .major = 13, .minor = 4 } };
    }
    break :blk out;
};
