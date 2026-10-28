const registry = @import("registry.zig");
const limits = @import("limits.zig");

const symbols = [_][]const u8{
    "_ZN15dsv41_pointwise13swiglu_kernelEPKfxS1_xPfxif",
    "_ZN15dsv41_pointwise10add_kernelEPfPKfx",
    "_ZN15dsv41_pointwise15add_bf16_kernelEPKfS1_P13__nv_bfloat16x",
};

pub const records = blk: {
    var out: [3]registry.CudaRecord = undefined;
    for (symbols, 0..) |symbol, i| {
        var usage: [5]limits.Usage = @splat(.{ .regs = 0, .static = 0, .bound = 0 });
        usage[1] = .{ .regs = ([_]u32{ 20, 26, 12 })[i], .static = 0, .bound = 256 };
        out[i] = .{ .launch = .{ .name = "dense_pointwise " ++ symbol, .image = "dense_pointwise", .symbol = symbol, .threads = 256, .min_sm = 86, .max_sm = 86 }, .usage = usage, .min_nvcc = .{ .major = 13, .minor = 4 } };
    }
    break :blk out;
};
