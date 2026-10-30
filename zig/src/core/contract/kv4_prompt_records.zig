//! Offline nvcc 13.4 records; device equality and timing remain separate gates.
const std = @import("std");
const registry = @import("registry.zig");

pub const records = blk: {
    var out: [9]registry.CudaRecord = undefined;
    out[0] = .{
        .launch = .{ .name = "kv4 tf_kv4_prepare128", .symbol = "tf_kv4_prepare128", .image = "kv4", .threads = 32, .min_sm = 80 },
        .min_nvcc = .{ .major = 13, .minor = 4 },
        .usage = .{ .{ .regs = 80, .static = 2112, .bound = 32 }, .{ .regs = 128, .static = 2112, .bound = 32 }, .{ .regs = 80, .static = 2112, .bound = 32 }, .{ .regs = 80, .static = 2112, .bound = 32 }, .{ .regs = 80, .static = 2112, .bound = 32 } },
    };
    for (1..9) |warps| {
        out[warps] = .{
            .launch = .{ .name = std.fmt.comptimePrint("kv4 tf_kv4_prompt128 warps {d}", .{warps}), .symbol = "tf_kv4_prompt128", .image = "kv4", .threads = 32 * warps, .dynamic = 53248 + 4096 * warps, .opt_in = true, .min_sm = 80 },
            .min_nvcc = .{ .major = 13, .minor = 4 },
            .usage = .{ .{ .regs = 142, .static = 0 }, .{ .regs = 142, .static = 0 }, .{ .regs = 142, .static = 0 }, .{ .regs = 143, .static = 0 }, .{ .regs = 143, .static = 0 } },
        };
    }
    break :blk out;
};

test "prepared KV4 launches fit every registered SM" {
    try std.testing.expectEqual(@as(usize, 0), try registry.checkCuda(std.testing.allocator, &records, false));
}
