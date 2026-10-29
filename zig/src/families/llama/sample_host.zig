const std = @import("std");
const lanes = @import("lanes");

pub const Draw = struct {
    position: u32,
    sampling: ?lanes.Sampling,
    penalties: lanes.Penalties,
    history: []const u32,
    pending: ?u32,
};

pub fn rounded(v: f32) f32 {
    const bits: u32 = @bitCast(v);
    return @bitCast((bits +% (0x7fff + ((bits >> 16) & 1))) & 0xffff0000);
}

pub const Prepared = struct {
    entries: []lanes.gpu_rule.Cand,
    pub fn init(a: std.mem.Allocator, logits: []const f32, d: Draw) !Prepared {
        const all = try a.alloc(lanes.gpu_rule.Cand, logits.len);
        defer a.free(all);
        for (logits, all, 0..) |v, *c, i| {
            var count: u32 = @intFromBool(d.pending == @as(u32, @intCast(i)));
            for (d.history) |t| count += @intFromBool(t == i);
            const score = rounded(rounded(v) - if (count > 0) d.penalties.of(count) else @as(f32, 0));
            c.* = .{ .v = score * lanes.gpu_rule.rule(d.sampling.?).inv_t, .i = @intCast(i) };
        }
        std.mem.sort(lanes.gpu_rule.Cand, all, {}, lanes.gpu_rule.better);
        const rule = d.sampling.?;
        var n = if (rule.top_k == 0) all.len else @min(all.len, rule.top_k);
        var norm: f64 = 0;
        for (all[0..n]) |c| norm += @exp(@as(f64, c.v - all[0].v));
        var cum: f64 = 0;
        for (all[0..n], 0..) |c, j| {
            cum += @exp(@as(f64, c.v - all[0].v)) / norm;
            if (rule.top_p > 0 and rule.top_p < 1 and cum >= rule.top_p) {
                n = j + 1;
                break;
            }
        }
        while (n > 1 and all[n - 1].v < all[0].v + rule.minLog()) n -= 1;
        while (n > 1 and all[n - 1].v < all[0].v - 24) n -= 1;
        return .{ .entries = try a.dupe(lanes.gpu_rule.Cand, all[0..n]) };
    }
    pub fn deinit(p: Prepared, a: std.mem.Allocator) void {
        a.free(p.entries);
    }
    pub fn pick(p: Prepared, seed: u64, position: u32) u32 {
        var best: f32 = -std.math.inf(f32);
        var token: u32 = p.entries[0].i;
        for (p.entries) |c| {
            const u = @min(lanes.gpu_rule.uniform24(seed, position, c.i), @as(f32, @bitCast(@as(u32, 0x3f7fffff))));
            const score = c.v - @log(-@log(u));
            if (score > best) {
                best = score;
                token = c.i;
            }
        }
        return token;
    }
    pub fn probabilities(p: Prepared, out: []f64) void {
        @memset(out, 0);
        var z: f64 = 0;
        for (p.entries) |c| z += @exp(@as(f64, c.v - p.entries[0].v));
        for (p.entries) |c| out[c.i] = @exp(@as(f64, c.v - p.entries[0].v)) / z;
    }
};
