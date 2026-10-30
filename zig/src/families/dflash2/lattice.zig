const std = @import("std");
pub const threads = 256;
pub const slice = 4096;
pub const Row = extern struct {
    ids: [16]u32 = @splat(0),
    values: [16]f32 = @splat(0),
    log_mass: f64 = 0,
    invalid: u32 = 0,
    pad: u32 = 0,
};
pub fn value(b: u16) f32 {
    return @bitCast(@as(u32, b) << 16);
}
pub fn slices(vocab: u32) u32 {
    return (vocab - 1) / slice + 1;
}
fn insert(out: *Row, v: f32, id: u32, k: usize) void {
    var i: usize = 0;
    while (i < k and (v < out.values[i] or (v == out.values[i] and id >= out.ids[i]))) : (i += 1) {}
    if (i == k) return;
    var j = k - 1;
    while (j > i) : (j -= 1) {
        out.values[j] = out.values[j - 1];
        out.ids[j] = out.ids[j - 1];
    }
    out.values[i] = v;
    out.ids[i] = id;
}
pub fn reference(raw: []const u16, k: usize) !Row {
    @setFloatMode(.strict);
    var out: Row = .{ .values = @splat(-std.math.inf(f32)) };
    var maximum: f32 = -std.math.inf(f32);
    for (raw, 0..) |b, id| {
        const v = value(b);
        if (!std.math.isFinite(v)) return error.DraftLatticeFinite;
        maximum = @max(maximum, v);
        var i: usize = 0;
        while (i < k and v <= out.values[i]) : (i += 1) {}
        if (i == k) continue;
        var j = k - 1;
        while (j > i) : (j -= 1) {
            out.values[j] = out.values[j - 1];
            out.ids[j] = out.ids[j - 1];
        }
        out.values[i] = v;
        out.ids[i] = @intCast(id);
    }
    var sum: f64 = 0;
    for (raw) |b| sum += @exp(@as(f64, value(b) - maximum));
    out.log_mass = @log(sum);
    return out;
}
pub fn emulate(raw: []const u16, k: usize) Row {
    @setFloatMode(.strict);
    var out: Row = .{ .values = @splat(-std.math.inf(f32)) };
    var base: usize = 0;
    while (base < raw.len) : (base += slice) {
        var part: Row = .{ .values = @splat(-std.math.inf(f32)), .ids = @splat(std.math.maxInt(u32)) };
        for (raw[base..@min(base + slice, raw.len)], base..) |b, id| {
            const v = value(b);
            if (!std.math.isFinite(v)) out.invalid = 1;
            insert(&part, v, @intCast(id), k);
        }
        for (part.values[0..k], part.ids[0..k]) |v, id| insert(&out, v, id, k);
    }
    var partial: [threads]f64 = @splat(0);
    for (&partial, 0..) |*sum, t| {
        var i = t;
        while (i < raw.len) : (i += threads) sum.* += @exp(@as(f64, value(raw[i]) - out.values[0]));
    }
    var step: usize = threads / 2;
    while (step > 0) : (step /= 2) for (0..step) |i| {
        partial[i] += partial[i + step];
    };
    out.log_mass = @log(partial[0]);
    return out;
}
pub fn tolerance(vocab: usize) f64 {
    return 8 * @as(f64, @floatFromInt(vocab + 1024)) * std.math.floatEps(f64);
}
pub fn normalizer(row: Row, vocab: usize) !?f32 {
    @setFloatMode(.strict);
    if (row.invalid != 0) return error.DraftLatticeFinite;
    if (!std.math.isFinite(row.log_mass)) return null;
    const e = tolerance(vocab);
    const low = row.values[0] + @as(f32, @floatCast(row.log_mass - e));
    const high = row.values[0] + @as(f32, @floatCast(row.log_mass + e));
    // Equal rounded endpoints preserve the old unary bits even at a selector score tie.
    return if (@as(u32, @bitCast(low)) == @as(u32, @bitCast(high))) low else null;
}
pub fn exactNormalizer(row: Row) f32 {
    @setFloatMode(.strict);
    return row.values[0] + @as(f32, @floatCast(row.log_mass));
}
test {
    _ = @import("lattice_test.zig");
}
