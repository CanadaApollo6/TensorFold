//! LFM2-MoE's weights on the GPU from an MLX affine build: tiled 4-bit projections, experts in qwen27's blocks, the
//! fp32 router and expert bias, and the tied table at 4 bits (embed4.safetensors beside the shards; the shards' own
//! table, 6-bit in mlx-community's builds, stays on disk).

const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const q27 = @import("qwen27");
const source = @import("nemotron").source;
const c = @import("shape.zig");
const kern = @import("kernels.zig");
const moe = @import("moe.zig");

pub const QLinear = kern.QLinear;
const Tensor = core.checkpoint.Tensor;
pub const embed_file = "embed4.safetensors";

pub const Embed = struct { w: u64, s: u64, b: u64 };
pub const Conv = struct { in: QLinear, out: QLinear, w: u64 }; // w: bf16 [hidden, 3], tap 2 the newest row
pub const Attention = struct { q: QLinear, k: QLinear, v: QLinear, o: QLinear, q_norm: u64, k_norm: u64 };
pub const Mixer = union(enum) { conv: Conv, attention: Attention };
pub const Dense = struct { gate: QLinear, up: QLinear, down: QLinear };
pub const Ffn = union(enum) { dense: Dense, moe: moe.Layer };
pub const Layer = struct { op_norm: u64, ffn_norm: u64, mixer: Mixer, ffn: Ffn };

pub const Weights = struct {
    gpa: std.mem.Allocator,
    g: c.Geometry,
    embed: Embed = undefined,
    layers: []Layer = &.{},
    norm: u64 = 0,
    head: QLinear = undefined,
    inv_freq: u64 = 0, // fp32 [head_dim / 2]
    eos: [c.max_eos]u32 = undefined,
    eos_count: usize = 0,
    buffers: std.ArrayList(cuda.DeviceBuffer) = .empty,

    pub fn deinit(w: *Weights) void {
        for (w.buffers.items) |*b| b.free();
        w.buffers.deinit(w.gpa);
        w.gpa.free(w.layers);
        w.* = undefined;
    }

    pub fn isEos(w: *const Weights, token: u32) bool {
        for (w.eos[0..w.eos_count]) |e| if (e == token) return true;
        return false;
    }

    pub fn bytes(w: *const Weights) usize {
        var n: usize = 0;
        for (w.buffers.items) |b| n += b.len;
        return n;
    }
};

const chunk = 32; // experts a repack pass takes
const words4 = 288; // uint32 a 4-bit block
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
        const width = m.t[0].shape[m.t[0].rank - 1];
        m.bits = @intCast(width * 32 / k);
        const ok = (m.bits == 3 or m.bits == 4) and width * 32 == k * m.bits and m.t[0].is(.u32, &.{ e, n, width }) and
            m.t[1].is(.bf16, &.{ e, n, kg }) and m.t[2].is(.bf16, &.{ e, n, kg });
        if (!ok) {
            std.log.err("{s}: not MLX affine group-64 3- or 4-bit experts ({d}, {d}, {d})", .{ name, e, n, k });
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

const Loader = struct {
    gpa: std.mem.Allocator,
    ops: kern.base.Ops,
    mk: *const q27.moe.Kernels,
    w: *Weights,
    src: *source.Source,
    scratch: cuda.DeviceBuffer,
    pool: q27.Pool = .{},

    fn alloc(L: *Loader, len: usize) !u64 {
        return L.pool.take(L.gpa, L.ops.k.d, &L.w.buffers, len);
    }

    fn tmp(L: *Loader, len: usize) !u64 {
        if (L.scratch.len < len) {
            try L.src.flush();
            try L.ops.s.synchronize();
            L.scratch.free();
            L.scratch = try cuda.DeviceBuffer.alloc(L.ops.k.d, len);
        }
        return L.scratch.ptr;
    }

    fn raw(L: *Loader, t: Tensor) !u64 {
        const ptr = try L.alloc(t.bytes.len);
        try L.src.upload(ptr, t.bytes);
        return ptr;
    }

    fn vector(L: *Loader, ck: *core.Checkpoint, name: []const u8, dtype: core.checkpoint.DType, len: usize) !u64 {
        return L.raw(try ck.expect(name, dtype, &.{len}));
    }

    /// qmm_fast.tile: an MLX 4-bit (n, k) projection into the lane matmul's tiled layout.
    fn tile(L: *Loader, ck: *core.Checkpoint, name: []const u8, n: usize, k: usize) !QLinear {
        var parts: [3]Tensor = undefined;
        var buf: [192]u8 = undefined;
        for ([_][]const u8{ "weight", "scales", "biases" }, 0..) |s, i| parts[i] = try ck.get(try std.fmt.bufPrint(&buf, "{s}.{s}", .{ name, s }));
        const k8 = k / 8;
        const kg = k / 64;
        if (!parts[0].is(.u32, &.{ n, k8 }) or !parts[1].is(.bf16, &.{ n, kg }) or !parts[2].is(.bf16, &.{ n, kg })) {
            std.log.err("{s}: not a 4-bit group-64 ({d}, {d}) projection", .{ name, n, k });
            return error.UnexpectedTensor;
        }
        const sizes = [3]usize{ n * k8 * 4, n * kg * 2, n * kg * 2 };
        const base = try L.tmp(sizes[0] + sizes[1] + sizes[2]);
        const at = [3]usize{ 0, sizes[0], sizes[0] + sizes[1] };
        for (0..3) |j| try L.src.upload(base + at[j], parts[j].bytes);
        const npad = (n + 127) / 128 * 128;
        const q: QLinear = .{ .w = try L.alloc(npad * k / 2), .s = try L.alloc(kg * npad * 2), .b = try L.alloc(kg * npad * 2), .n = n, .k = k, .npad = npad };
        const total: u64 = @as(u64, npad / 64) * kg * 512;
        try L.src.flush();
        var a: cuda.Args = .{};
        a.add(base);
        for ([_]usize{ n, k8, kg }) |v| a.add(@as(c_int, @intCast(v)));
        a.add(q.w);
        a.add(@as(c_longlong, @intCast(total)));
        try cuda.launch.launch(L.ops.k.pack_dense, .{ .grid = .{ .x = @intCast((total + 255) / 256) }, .block = .{ .x = 256 } }, L.ops.s, &a);
        for ([_]u64{ q.s, q.b }, [_]usize{ at[1], at[2] }) |out, off| {
            var t: cuda.Args = .{};
            t.add(base + off);
            for ([_]usize{ n, kg, npad }) |v| t.add(@as(c_int, @intCast(v)));
            t.add(out);
            const cells = @as(u64, kg) * npad;
            try cuda.launch.launch(L.ops.k.transpose16, .{ .grid = .{ .x = @intCast((cells + 255) / 256) }, .block = .{ .x = 256 } }, L.ops.s, &t);
        }
        return q;
    }

    fn go(L: *Loader, f: cuda.Function, g: [3]usize, threads: usize, a: *cuda.Args) !void {
        try cuda.launch.launch(f, .{ .grid = .{ .x = @intCast(g[0]), .y = @intCast(g[1]), .z = @intCast(g[2]) }, .block = .{ .x = @intCast(threads) } }, L.ops.s, a);
    }

    /// Experts of `m` repacked into `out` (blocks of M = `mm` projections, this one `mi`): moe_load.experts.
    fn experts(L: *Loader, m: Mlx, e: usize, n: usize, kk: usize, out: u64, mm: usize, mi: usize) !void {
        const k = L.mk;
        const kg = kk / 64;
        const nb = n / 32;
        const per = [3]usize{ n * kk * m.bits / 8, n * kg * 2, n * kg * 2 };
        const wide = n * kk / 2;
        const packed_bytes = nb * kg * words4 * 4;
        const bw: usize = if (m.bits == 4) words4 else words3;
        var e0: usize = 0;
        while (e0 < e) : (e0 += chunk) {
            const cn = @min(chunk, e - e0);
            const at_ = [_]usize{ 0, cn * per[0], cn * (per[0] + per[1]) };
            const w4 = cn * (per[0] + per[1] + per[2]);
            const pk = w4 + if (m.bits == 3) cn * wide else 0;
            const base = try L.tmp(pk + cn * packed_bytes);
            for (0..3) |j| try L.src.upload(base + at_[j], m.t[j].bytes[e0 * per[j] ..][0 .. cn * per[j]]);
            try L.src.flush();
            const words = if (m.bits == 3) base + w4 else base;
            if (m.bits == 3) {
                var a: cuda.Args = .{};
                a.add(base);
                a.add(words);
                a.add(@as(c_long, @intCast(cn * wide / 4)));
                try L.go(k.widen3, .{ grid(cn * wide / 4), 1, 1 }, 256, &a);
            }
            var p: cuda.Args = .{};
            for ([_]u64{ words, base + at_[1], base + at_[2], base + pk }) |v| p.add(v);
            for ([_]usize{ n, kk / 8, kg, nb }) |v| p.add(uc(v));
            try L.go(k.pack, .{ kg, nb, cn }, 288, &p);
            const blocks = cn * nb * kg;
            const dst = out + (e0 * nb * kg * mm * bw) * 4;
            var q: cuda.Args = .{};
            q.add(base + pk);
            q.add(dst);
            q.add(@as(c_long, @intCast(blocks)));
            if (m.bits == 3) {
                q.add(uc(mm * bw));
                q.add(uc(mi * bw));
                try L.go(k.planes3, .{ grid(blocks * 32), 1, 1 }, 256, &q);
            } else {
                q.add(uc(words4));
                q.add(uc(mm * bw));
                q.add(uc(mi * bw));
                try L.go(k.place, .{ grid(blocks * words4), 1, 1 }, 256, &q);
            }
        }
    }

    fn moeLayer(L: *Loader, ck: *core.Checkpoint, p: []const u8, g: c.Geometry) !moe.Layer {
        var buf: [192]u8 = undefined;
        const d = g.hidden;
        const gate = try Mlx.get(ck, try std.fmt.bufPrint(&buf, "{s}feed_forward.switch_mlp.gate_proj", .{p}), g.experts, g.moe_inter, d);
        const up = try Mlx.get(ck, try std.fmt.bufPrint(&buf, "{s}feed_forward.switch_mlp.up_proj", .{p}), g.experts, g.moe_inter, d);
        const down = try Mlx.get(ck, try std.fmt.bufPrint(&buf, "{s}feed_forward.switch_mlp.down_proj", .{p}), g.experts, d, g.moe_inter);
        if (gate.bits != up.bits or gate.bits != down.bits) return error.UnsupportedQuantization;
        const bw: usize = if (gate.bits == 4) words4 else words3;
        const up_expert = (g.moe_inter / 32) * (d / 64) * 2 * bw * 4;
        const down_expert = (d / 32) * (g.moe_inter / 64) * bw * 4;
        var l: moe.Layer = .{ .router = 0, .bias = 0, .up = try L.alloc(g.experts * up_expert), .down = try L.alloc(g.experts * down_expert), .bits = gate.bits };
        l.router = try L.raw(try ck.expect(try std.fmt.bufPrint(&buf, "{s}feed_forward.gate.weight", .{p}), .f32, &.{ g.experts, d }));
        l.bias = try L.raw(try ck.expect(try std.fmt.bufPrint(&buf, "{s}feed_forward.expert_bias", .{p}), .f32, &.{g.experts}));
        try L.experts(gate, g.experts, g.moe_inter, d, l.up, 2, 0);
        try L.experts(up, g.experts, g.moe_inter, d, l.up, 2, 1);
        try L.experts(down, g.experts, d, g.moe_inter, l.down, 1, 0);
        return l;
    }
};

/// weights.inv_freq: theta ** (-i / half) in float64, rounded to fp32.
pub fn invFreq(theta: f64, out: []f32) void {
    const half: f64 = @floatFromInt(out.len);
    for (out, 0..) |*f, i| f.* = @floatCast(std.math.pow(f64, theta, -@as(f64, @floatFromInt(i)) / half));
}

pub fn load(gpa: std.mem.Allocator, io: std.Io, ops: kern.Ops, dir: []const u8, cfg: c.Config) !Weights {
    const g = cfg.g;
    var w: Weights = .{ .gpa = gpa, .g = g, .eos = cfg.eos, .eos_count = cfg.eos_count };
    errdefer w.deinit();
    const bo = ops.b();
    var src = try source.Source.init(gpa, .{ .d = bo.k.d, .s = bo.s, .discrete = bo.k.discrete });
    defer src.deinit();
    var L: Loader = .{ .gpa = gpa, .ops = bo, .mk = &ops.k.moe, .w = &w, .src = &src, .scratch = try cuda.DeviceBuffer.alloc(bo.k.d, 1 << 20) };
    defer {
        src.flush() catch {};
        bo.s.synchronize() catch {};
        L.scratch.free();
        L.pool.deinit(gpa);
    }
    var ck = try core.Checkpoint.openModelPrefix(gpa, io, dir, null);
    defer ck.close();
    try src.add(&ck);
    var ek: core.Checkpoint = .{ .gpa = gpa, .io = io };
    defer ek.close();
    ek.add(dir, embed_file) catch |err| {
        std.log.err("{s}/{s}: the tied table at 4 bits is missing (01-inference/lfm25-24b/make_embed4.py writes it)", .{ dir, embed_file });
        return err;
    };
    try src.add(&ek);
    w.layers = try gpa.alloc(Layer, g.layers);
    const d = g.hidden;
    for (w.layers, 0..) |*layer, i| {
        var pb: [64]u8 = undefined;
        var nb: [192]u8 = undefined;
        const p = try std.fmt.bufPrint(&pb, "model.layers.{d}.", .{i});
        const join = struct {
            fn f(buf: []u8, a: []const u8, b: []const u8) ![]const u8 {
                return std.fmt.bufPrint(buf, "{s}{s}", .{ a, b });
            }
        }.f;
        layer.op_norm = try L.vector(&ck, try join(&nb, p, "operator_norm.weight"), .bf16, d);
        layer.ffn_norm = try L.vector(&ck, try join(&nb, p, "ffn_norm.weight"), .bf16, d);
        if (g.isAttention(i)) {
            layer.mixer = .{ .attention = .{
                .q = try L.tile(&ck, try join(&nb, p, "self_attn.q_proj"), g.qInner(), d),
                .k = try L.tile(&ck, try join(&nb, p, "self_attn.k_proj"), g.kvInner(), d),
                .v = try L.tile(&ck, try join(&nb, p, "self_attn.v_proj"), g.kvInner(), d),
                .o = try L.tile(&ck, try join(&nb, p, "self_attn.out_proj"), d, g.qInner()),
                .q_norm = try L.vector(&ck, try join(&nb, p, "self_attn.q_layernorm.weight"), .bf16, c.head_dim),
                .k_norm = try L.vector(&ck, try join(&nb, p, "self_attn.k_layernorm.weight"), .bf16, c.head_dim),
            } };
        } else {
            layer.mixer = .{ .conv = .{
                .in = try L.tile(&ck, try join(&nb, p, "conv.in_proj"), 3 * d, d),
                .out = try L.tile(&ck, try join(&nb, p, "conv.out_proj"), d, d),
                .w = try L.raw(try ck.expect(try join(&nb, p, "conv.conv.weight"), .bf16, &.{ d, c.conv_taps, 1 })),
            } };
        }
        if (i < g.dense_layers) {
            layer.ffn = .{ .dense = .{
                .gate = try L.tile(&ck, try join(&nb, p, "feed_forward.gate_proj"), g.intermediate, d),
                .up = try L.tile(&ck, try join(&nb, p, "feed_forward.up_proj"), g.intermediate, d),
                .down = try L.tile(&ck, try join(&nb, p, "feed_forward.down_proj"), d, g.intermediate),
            } };
        } else layer.ffn = .{ .moe = try L.moeLayer(&ck, p, g) };
    }
    w.norm = try L.vector(&ck, "model.embedding_norm.weight", .bf16, d);
    // the shards' own table (6-bit) stays on disk: embed4's rows and the head read the 4-bit one
    for ([_][]const u8{ "weight", "scales", "biases" }) |s| {
        var nb: [64]u8 = undefined;
        _ = try ck.get(try std.fmt.bufPrint(&nb, "model.embed_tokens.{s}", .{s}));
    }
    w.embed = .{
        .w = try L.raw(try ek.expect("model.embed_tokens.weight", .u32, &.{ g.vocab, d / 8 })),
        .s = try L.raw(try ek.expect("model.embed_tokens.scales", .bf16, &.{ g.vocab, d / 64 })),
        .b = try L.raw(try ek.expect("model.embed_tokens.biases", .bf16, &.{ g.vocab, d / 64 })),
    };
    w.head = try L.tile(&ek, "model.embed_tokens", g.vocab, d);
    if (ck.unused() != 0 or ek.unused() != 0) return error.UnusedCheckpointTensors;
    var inv: [c.head_dim / 2]f32 = undefined;
    invFreq(g.theta, &inv);
    w.inv_freq = try L.alloc(@sizeOf(@TypeOf(inv)));
    try bo.upload(w.inv_freq, std.mem.asBytes(&inv));
    try src.flush();
    try bo.s.synchronize();
    return w;
}
