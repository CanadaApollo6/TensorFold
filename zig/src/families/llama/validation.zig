const std = @import("std");

pub const relative_l2_bound = 0.03;
pub const max_error_fraction = 0.03;
pub const max_error_floor = 1e-4;
pub const Layer = struct { rmse: f64, max_abs: f64, relative_l2: f64, ref_peak: f64 };
pub const Positions = struct {
    relative_l2: f64 = 0,
    relative_position: usize = 0,
    scaled_max: f64 = 0,
    max_position: usize = 0,
    failures: usize = 0,
};

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
    if (m.max_abs > max_error_floor + max_error_fraction * m.ref_peak or m.relative_l2 > relative_l2_bound)
        return error.LlamaLayerDrift;
}

pub fn positions(ref: []const f32, got: []const f32, width: usize) !Positions {
    if (width == 0 or ref.len == 0 or ref.len != got.len or ref.len % width != 0) return error.BadReference;
    var out: Positions = .{};
    for (0..ref.len / width) |pos| {
        const m = try layer(ref[pos * width ..][0..width], got[pos * width ..][0..width]);
        if (m.relative_l2 > out.relative_l2) {
            out.relative_l2 = m.relative_l2;
            out.relative_position = pos;
        }
        const scaled = m.max_abs / @max(m.ref_peak, 1e-30);
        if (scaled > out.scaled_max) {
            out.scaled_max = scaled;
            out.max_position = pos;
        }
        require(m) catch {
            out.failures += 1;
        };
    }
    return out;
}

pub fn requirePositions(m: Positions) !void {
    if (m.failures != 0) return error.LlamaLayerDrift;
}

test "Llama layer comparison rejects zeros and nonfinite references" {
    const got = [_]f32{ 1, -2, 3, -4 };
    try require(try layer(&got, &got));
    try std.testing.expectError(error.LlamaLayerDrift, require(try layer(&.{ 0, 0, 0, 0 }, &got)));
    try std.testing.expectError(error.NonFinite, layer(&.{std.math.nan(f32)}, &.{1}));
}

test "Llama position bounds expose corruption hidden by another position's residual scale" {
    const ref = [_]f32{ 1000000, -1000000, 1, -1 };
    const got = [_]f32{ 1000000, -1000000, 1.1, -1.1 };
    try require(try layer(&ref, &got));
    const met = try positions(&ref, &got, 2);
    try std.testing.expectEqual(@as(usize, 1), met.relative_position);
    try std.testing.expectEqual(@as(usize, 1), met.max_position);
    try std.testing.expectEqual(@as(usize, 1), met.failures);
    try std.testing.expectError(error.LlamaLayerDrift, requirePositions(met));
    try std.testing.expectError(error.BadReference, positions(&ref, &got, 3));
}
