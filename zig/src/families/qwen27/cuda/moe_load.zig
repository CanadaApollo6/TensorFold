//! A qwen3_5_moe layer's MLP from MLX arrays: the 8-bit router as bf16 rows and the experts as moe.zig's blocks.

const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const moe = @import("moe.zig");

const Tensor = core.checkpoint.Tensor;
const chunk = 32; // experts a repack pass takes (experts.pack's chunk)
const words4 = 288; // uint32 a 4-bit block (groups of 64)
const words3 = 224;

/// MLX words [E, N, K * bits / 32] u32, scales and biases [E, N, K / 64] bf16.
const Mlx = struct {
    t: [3]Tensor,
    bits: u8,

    fn get(ck: *core.Checkpoint, name: []const u8, e: usize, n: usize, k: usize) !Mlx {
        var m: Mlx = undefined;
        var buf: [192]u8 = undefined;
        for ([_][]const u8{ "weight", "scales", "biases" }, 0..) |s, i| m.t[i] = try ck.get(try std.fmt.bufPrint(&buf, "{s}.{s}", .{ name, s }));
        const kg = k / 64;
        const lead: usize = if (m.t[0].rank == 3) m.t[0].shape[0] else 1;
        const width = m.t[0].shape[m.t[0].rank - 1];
        m.bits = @intCast(width * 32 / k);
        const shape3 = [_]usize{ e, n, width };
        const sb3 = [_]usize{ e, n, kg };
        const ok = lead == e and (m.bits == 3 or m.bits == 4 or m.bits == 8) and width * 32 == k * m.bits and
            m.t[0].is(.u32, if (m.t[0].rank == 3) &shape3 else shape3[1..]) and
            m.t[1].is(.bf16, if (m.t[1].rank == 3) &sb3 else sb3[1..]) and m.t[2].is(.bf16, if (m.t[2].rank == 3) &sb3 else sb3[1..]);
        if (!ok) {
            std.log.err("{s}: not an MLX affine group-64 ({d}, {d}, {d}) array", .{ name, e, n, k });
            return error.UnexpectedTensor;
        }
        return m;
    }
};

fn uc(v: usize) c_int {
    return @intCast(v);
}

fn grid(n: usize) usize {
    return @min((n + 255) / 256, 4096);
}

fn go(L: anytype, f: cuda.Function, g: [3]usize, threads: usize, a: *cuda.Args) !void {
    try cuda.launch.launch(f, .{ .grid = .{ .x = @intCast(g[0]), .y = @intCast(g[1]), .z = @intCast(g[2]) }, .block = .{ .x = @intCast(threads) } }, L.ops.s, a);
}

/// Experts [e0, e0 + e) of `m` repacked into `out` (blocks of M = `mm` projections, this one `mi`) from expert `at` on.
fn experts(L: anytype, k: *const moe.Kernels, m: Mlx, e: usize, n: usize, kk: usize, out: u64, mm: usize, mi: usize, at: usize) !void {
    const kg = kk / 64;
    const nb = n / 32;
    const per = [3]usize{ n * kk * m.bits / 8, n * kg * 2, n * kg * 2 };
    const wide = n * kk / 2; // an expert's 4-bit words
    const packed_bytes = nb * kg * words4 * 4;
    const bw: usize = if (m.bits == 4) words4 else words3;
    var e0: usize = 0;
    while (e0 < e) : (e0 += chunk) {
        const c = @min(chunk, e - e0);
        const at_ = [_]usize{ 0, c * per[0], c * (per[0] + per[1]) };
        const w4 = c * (per[0] + per[1] + per[2]);
        const pk = w4 + if (m.bits == 3) c * wide else 0;
        const base = try L.tmp(pk + c * packed_bytes);
        for (0..3) |j| try L.src.upload(base + at_[j], m.t[j].bytes[e0 * per[j] ..][0 .. c * per[j]]);
        try L.src.flush();
        const words = if (m.bits == 3) base + w4 else base;
        if (m.bits == 3) {
            var a: cuda.Args = .{};
            a.add(base);
            a.add(words);
            a.add(@as(c_long, @intCast(c * wide / 4)));
            try go(L, k.widen3, .{ grid(c * wide / 4), 1, 1 }, 256, &a);
        }
        var p: cuda.Args = .{};
        for ([_]u64{ words, base + at_[1], base + at_[2], base + pk }) |v| p.add(v);
        for ([_]usize{ n, kk / 8, kg, nb }) |v| p.add(uc(v));
        try go(L, k.pack, .{ kg, nb, c }, 288, &p);
        const blocks = c * nb * kg;
        const dst = out + ((at + e0) * nb * kg * mm * bw) * 4;
        var q: cuda.Args = .{};
        q.add(base + pk);
        q.add(dst);
        q.add(@as(c_long, @intCast(blocks)));
        if (m.bits == 3) {
            q.add(uc(mm * bw));
            q.add(uc(mi * bw));
            try go(L, k.planes3, .{ grid(blocks * 32), 1, 1 }, 256, &q);
        } else {
            q.add(uc(words4));
            q.add(uc(mm * bw));
            q.add(uc(mi * bw));
            try go(L, k.place, .{ grid(blocks * words4), 1, 1 }, 256, &q);
        }
    }
}

/// An 8-bit MLX matrix [rows, D] as bf16 rows at `out`.
fn dequant(L: anytype, k: *const moe.Kernels, m: Mlx, rows: usize, d: usize, out: u64) !void {
    if (m.bits != 8) return error.UnsupportedQuantization;
    const sizes = [3]usize{ m.t[0].bytes.len, m.t[1].bytes.len, m.t[2].bytes.len };
    const base = try L.tmp(sizes[0] + sizes[1] + sizes[2]);
    const at = [3]usize{ 0, sizes[0], sizes[0] + sizes[1] };
    for (0..3) |j| try L.src.upload(base + at[j], m.t[j].bytes);
    try L.src.flush();
    var a: cuda.Args = .{};
    for ([_]u64{ base, base + at[1], base + at[2], out }) |v| a.add(v);
    a.add(@as(c_long, @intCast(rows * d)));
    try go(L, k.dequant8, .{ grid(rows * d), 1, 1 }, 256, &a);
}

/// Layer `p`'s `mlp.` (with the checkpoint's prefix): E routed experts of `width` over `hidden`, plus the shared one.
pub fn layer(L: anytype, k: *const moe.Kernels, ck: *core.Checkpoint, p: []const u8, hidden: usize, width: usize, e: usize) !moe.Layer {
    var buf: [192]u8 = undefined;
    const name = struct {
        fn f(b: []u8, pre: []const u8, s: []const u8) ![]const u8 {
            return std.fmt.bufPrint(b, "{s}mlp.{s}", .{ pre, s });
        }
    }.f;
    const gate = try Mlx.get(ck, try name(&buf, p, "switch_mlp.gate_proj"), e, width, hidden);
    const up = try Mlx.get(ck, try name(&buf, p, "switch_mlp.up_proj"), e, width, hidden);
    const down = try Mlx.get(ck, try name(&buf, p, "switch_mlp.down_proj"), e, hidden, width);
    if (gate.bits != up.bits or gate.bits != down.bits or gate.bits == 8) return error.UnsupportedQuantization;
    const sg = try Mlx.get(ck, try name(&buf, p, "shared_expert.gate_proj"), 1, width, hidden);
    const su = try Mlx.get(ck, try name(&buf, p, "shared_expert.up_proj"), 1, width, hidden);
    const sd = try Mlx.get(ck, try name(&buf, p, "shared_expert.down_proj"), 1, hidden, width);
    if (sg.bits != 4 or su.bits != 4 or sd.bits != 4) return error.UnsupportedQuantization;
    const router = try Mlx.get(ck, try name(&buf, p, "gate"), 1, e, hidden);
    const shared_gate = try Mlx.get(ck, try name(&buf, p, "shared_expert_gate"), 1, 1, hidden);
    const bits = gate.bits;
    const bw: usize = if (bits == 4) words4 else words3;
    const up_expert = (width / 32) * (hidden / 64) * 2 * bw * 4;
    const down_expert = (hidden / 32) * (width / 64) * bw * 4;
    const apart = bits != 4; // the shared expert stays at 4 bits beside narrower ones
    const n_e = if (apart) e else e + 1;
    var l: moe.Layer = .{ .router = try L.alloc((e + 1) * hidden * 2), .up = try L.alloc(n_e * up_expert), .down = try L.alloc(n_e * down_expert), .bits = bits };
    try dequant(L, k, router, e, hidden, l.router);
    try dequant(L, k, shared_gate, 1, hidden, l.router + e * hidden * 2);
    try experts(L, k, gate, e, width, hidden, l.up, 2, 0, 0);
    try experts(L, k, up, e, width, hidden, l.up, 2, 1, 0);
    try experts(L, k, down, e, hidden, width, l.down, 1, 0, 0);
    if (apart) {
        l.up4 = try L.alloc(up_expert / bw * words4);
        l.down4 = try L.alloc(down_expert / bw * words4);
        l.e4 = @intCast(e);
    }
    try experts(L, k, sg, 1, width, hidden, if (apart) l.up4 else l.up, 2, 0, if (apart) 0 else e);
    try experts(L, k, su, 1, width, hidden, if (apart) l.up4 else l.up, 2, 1, if (apart) 0 else e);
    try experts(L, k, sd, 1, hidden, width, if (apart) l.down4 else l.down, 1, 0, if (apart) 0 else e);
    return l;
}
