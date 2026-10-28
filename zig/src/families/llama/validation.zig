const std = @import("std");

pub const Layer = struct { rmse: f64, max_abs: f64, relative_l2: f64, ref_peak: f64 };

pub fn layer(ref: []const f32, got: []const f32) !Layer {
    if (ref.len == 0 or ref.len != got.len) return error.BadReference;
    var worst: f64 = 0;
    var peak: f64 = 0;
    var squared_error: f64 = 0;
    var squared_ref: f64 = 0;
    for (ref, got) |rv, gv| {
        if (!std.math.isFinite(rv) or !std.math.isFinite(gv)) return error.NonFinite;
        const diff = @as(f64, rv) - gv;
        worst = @max(worst, @abs(diff));
        peak = @max(peak, @abs(@as(f64, rv)));
        squared_error += diff * diff;
        squared_ref += @as(f64, rv) * rv;
    }
    return .{ .rmse = @sqrt(squared_error / @as(f64, @floatFromInt(ref.len))), .max_abs = worst, .relative_l2 = @sqrt(squared_error / @max(squared_ref, 1e-30)), .ref_peak = peak };
}

pub fn require(m: Layer) !void {
    if (m.rmse > 1 or m.max_abs > 1e-4 + 0.01 * m.ref_peak or m.relative_l2 > 0.01)
        return error.LlamaLayerDrift;
}

test "Llama layer comparison rejects zeros and nonfinite references" {
    const got = [_]f32{ 1, -2, 3, -4 };
    try require(try layer(&got, &got));
    try std.testing.expectError(error.LlamaLayerDrift, require(try layer(&.{ 0, 0, 0, 0 }, &got)));
    try std.testing.expectError(error.NonFinite, layer(&.{std.math.nan(f32)}, &.{1}));
}
