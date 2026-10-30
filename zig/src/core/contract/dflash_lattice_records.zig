const std = @import("std");
const registry = @import("registry.zig");
pub const records = [_]registry.CudaRecord{
    .{ .launch = .{ .name = "sample tf_draft_lattice", .symbol = "tf_draft_lattice", .image = "sample", .threads = 256 }, .min_nvcc = .{ .major = 13, .minor = 4 }, .usage = .{ .{ .regs = 24, .static = 2112, .bound = 256 }, .{ .regs = 24, .static = 2112, .bound = 256 }, .{ .regs = 24, .static = 2112, .bound = 256 }, .{ .regs = 48, .static = 2112, .bound = 256 }, .{ .regs = 48, .static = 2112, .bound = 256 } } },
};
test "DFlash2 lattice nvcc 13.4 resources fit all compiled SMs without qualifying older compilers" {
    try std.testing.expectEqual(@as(usize, 0), try registry.checkCuda(std.testing.allocator, &records, true));
    try std.testing.expectEqual(registry.none, registry.usageFor(records[0], .{ .major = 13, .minor = 3 }));
}
