//! Per-target admission for optional CUDA variants.
const limits = @import("limits.zig");
pub const TargetDecision = enum { applicable, below_minimum, above_maximum, optional_shared_limit };
pub fn targetDecision(r: anytype, t: limits.Sm, u: limits.Usage) TargetDecision {
    if (t.sm < r.launch.min_sm) return .below_minimum;
    if (t.sm > r.launch.max_sm) return .above_maximum;
    if (r.optional_shared and !@import("gpu_memory").fits(u.static, r.launch.dynamic, t.smem_optin)) return .optional_shared_limit;
    return .applicable;
}

test "CUDA target envelope admits dense pointwise from sm_80 to sm_89" {
    const std = @import("std");
    const record = .{ .launch = limits.CudaLaunch{ .name = "dense_pointwise", .symbol = "add", .image = "dense_pointwise", .threads = 256, .min_sm = 80, .max_sm = 89 }, .optional_shared = false };
    for (limits.sms) |sm| {
        try std.testing.expectEqual(if (sm.sm <= 89) TargetDecision.applicable else TargetDecision.above_maximum, targetDecision(record, sm, .{ .regs = 26, .static = 0, .bound = 256 }));
    }
}
