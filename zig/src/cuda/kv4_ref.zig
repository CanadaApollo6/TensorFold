const std = @import("std");

pub fn turn(x: *[32]f32) void {
    turnBits(x, 32);
}

fn rounded(x: f32, bits: u8) f32 {
    if (bits == 32 or !std.math.isFinite(x)) return x;
    if (bits == 16) return @as(f16, @floatCast(x));
    const v: u32 = @bitCast(x);
    return @bitCast((v +% 0x7fff +% ((v >> 16) & 1)) & 0xffff0000);
}

fn turnBits(x: *[32]f32, bits: u8) void {
    var bit: usize = 1;
    while (bit < 32) : (bit *= 2) {
        for (0..32) |i| {
            if (i & bit != 0) continue;
            const a = x[i];
            const b = x[i + bit];
            x[i] = rounded(a + b, bits);
            x[i + bit] = rounded(a - b, bits);
        }
    }
    for (x) |*v| v.* *= 0.17677669529663688110;
}

pub fn pack(x: []const f32, out: []u8) void {
    packBits(x, out, 32);
}

pub fn packBits(x: []const f32, out: []u8, bits: u8) void {
    const dim = x.len;
    for (0..dim / 32) |g| {
        var y: [32]f32 = undefined;
        @memcpy(&y, x[g * 32 ..][0..32]);
        turnBits(&y, bits);
        var scale: f32 = 0;
        var bad = false;
        for (y) |v| {
            bad = bad or !std.math.isFinite(v);
            scale = @max(scale, @abs(v));
        }
        const inverse: f32 = 1.0 / (scale + @as(f32, 1e-10));
        for (0..16) |i| {
            var codes: [2]u8 = undefined;
            for (&codes, 0..) |*c, j| {
                const normalized: f32 = y[2 * i + j] * inverse;
                const scaled: f32 = normalized * 8;
                c.* = if (bad) 0 else @intFromFloat(@min(15, @max(0, @floor(scaled) + 8)));
            }
            out[g * 16 + i] = codes[0] | (codes[1] << 4);
        }
        const half: f16 = @floatCast(scale);
        const stored: f16 = if (bad or !std.math.isFinite(half)) std.math.nan(f16) else half;
        std.mem.writeInt(u16, out[dim / 2 + g * 2 ..][0..2], @bitCast(stored), .little);
    }
}

pub fn unpack(x: []const u8, out: []f32) void {
    const dim = out.len;
    for (0..dim / 32) |g| {
        const scale: f16 = @bitCast(std.mem.readInt(u16, x[dim / 2 + g * 2 ..][0..2], .little));
        var y: [32]f32 = undefined;
        for (&y, 0..) |*v, i| {
            const code = (x[g * 16 + i / 2] >> @as(u3, @intCast(4 * (i & 1)))) & 15;
            v.* = (@as(f32, @floatFromInt(code)) - 7.5) * (@as(f32, scale) * 0.125);
        }
        turn(&y);
        @memcpy(out[g * 32 ..][0..32], &y);
    }
}

test "H32 is orthonormal and zeros stay exactly zero in the Qwen midpoint format" {
    var x: [32]f32 = undefined;
    for (&x, 0..) |*v, i| v.* = @as(f32, @floatFromInt(i)) - 16;
    const original = x;
    turn(&x);
    turn(&x);
    for (x, original) |v, ref| try std.testing.expectApproxEqAbs(ref, v, 4e-6);
    var encoded: [72]u8 = undefined;
    var zeros: [128]f32 = @splat(0);
    pack(&zeros, &encoded);
    unpack(&encoded, &zeros);
    for (zeros) |v| try std.testing.expectEqual(@as(f32, 0), v);
    zeros[1] = std.math.inf(f32);
    pack(&zeros, &encoded);
    unpack(&encoded, &zeros);
    for (zeros[0..32]) |v| try std.testing.expect(std.math.isNan(v));
}
