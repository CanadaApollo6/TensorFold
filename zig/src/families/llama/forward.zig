const std = @import("std");
const cuda = @import("cuda");
const weights = @import("weights.zig");
const Kernels = @import("kernels.zig").Kernels;
const Config = @import("config.zig").Config;
const sampling = @import("sampling.zig");

pub const Model = struct {
    gpa: std.mem.Allocator,
    d: *const cuda.Driver,
    w: *const weights.Weights,
    k: *Kernels,
    s: cuda.Stream,
    context: usize,
    buffers: std.ArrayList(cuda.DeviceBuffer) = .empty,
    host_meta: cuda.HostBuffer,
    meta: u64 = 0,
    embed: u64 = 0,
    res: u64 = 0,
    x: u64 = 0,
    qkv: u64 = 0,
    q: u64 = 0,
    kn: u64 = 0,
    vn: u64 = 0,
    attn: u64 = 0,
    branch: u64 = 0,
    gate: u64 = 0,
    up: u64 = 0,
    act: u64 = 0,
    inv: u64 = 0,
    kc: u64 = 0,
    vc: u64 = 0,
    scratch: u64 = 0,
    offsets: u64 = 0,
    logits: u64 = 0,
    token: u64 = 0,
    groups: usize,
    meta_pending: bool = false,
    position: usize = 0,
    sampler: ?sampling.Sampler = null,

    pub fn init(a: std.mem.Allocator, d: *const cuda.Driver, w: *const weights.Weights, k: *Kernels, s: cuda.Stream, context: usize) !Model {
        if (context == 0 or context > w.config.max_context) return error.PastContext;
        var m: Model = .{ .gpa = a, .d = d, .w = w, .k = k, .s = s, .context = context, .host_meta = try cuda.HostBuffer.alloc(d, 64), .groups = cuda.tree_attention.TreeAttention.groups(context) };
        errdefer m.deinit();
        const c = w.config;
        m.sampler = try sampling.Sampler.init(a, d, c.vocab, context);
        m.meta = try m.alloc(64);
        m.embed = try m.alloc(c.hidden * 2);
        m.res = try m.alloc(c.hidden * 4);
        m.x = try m.alloc(c.hidden * 2);
        m.qkv = try m.alloc((c.qDim() + 2 * c.kvDim()) * 2);
        m.q = try m.alloc(c.qDim() * 2);
        m.kn = try m.alloc(c.kvDim() * 2);
        m.vn = try m.alloc(c.kvDim() * 2);
        m.attn = try m.alloc(c.qDim() * 2);
        m.branch = try m.alloc(c.hidden * 4);
        m.gate = try m.alloc(c.intermediate * 4);
        m.up = try m.alloc(c.intermediate * 4);
        m.act = try m.alloc(c.intermediate * 4);
        m.inv = try m.alloc(c.head_dim / 2 * 4);
        const kv_bytes = try c.kvBytes(w.layers.len, context);
        m.kc = try m.alloc(kv_bytes / 2);
        m.vc = try m.alloc(kv_bytes / 2);
        m.scratch = try m.alloc(cuda.tree_attention.TreeAttention.scratchBytes(1, c.heads, c.kv_heads, m.groups, c.head_dim));
        m.offsets = try m.alloc(w.layers.len * 16);
        const offsets = try a.alloc(i64, w.layers.len * 2);
        defer a.free(offsets);
        for (0..w.layers.len) |i| {
            offsets[2 * i] = @intCast(i * context * c.kvDim());
            offsets[2 * i + 1] = @divExact(@as(i64, @intCast(m.vc)) - @as(i64, @intCast(m.kc)), 2) + offsets[2 * i];
        }
        try d.check(d.api.cuMemcpyHtoD_v2(m.offsets, offsets.ptr, offsets.len * 8), "KV offsets upload");
        m.logits = try m.alloc(c.vocab * 4);
        m.token = try m.alloc(4);
        var inv: [64]f32 = undefined;
        for (&inv, 0..) |*v, i| v.* = 1.0 / std.math.pow(f32, c.rope_theta, @as(f32, @floatFromInt(2 * i)) / 128.0);
        try d.check(d.api.cuMemcpyHtoD_v2(m.inv, &inv, @sizeOf(@TypeOf(inv))), "RoPE inv upload");
        for (w.layers) |l| inline for (.{ "q", "k", "v", "o", "gate", "up", "down" }) |key| try k.exl.reserve(@field(l, key));
        try k.exl.reserve(w.head);
        return m;
    }

    fn alloc(m: *Model, bytes: usize) !u64 {
        var b = try cuda.DeviceBuffer.alloc(m.d, bytes);
        errdefer b.free();
        try m.buffers.append(m.gpa, b);
        return b.ptr;
    }

    pub fn deinit(m: *Model) void {
        m.s.synchronize() catch {};
        if (m.sampler) |*sampler| sampler.deinit();
        for (m.buffers.items) |*b| b.free();
        m.buffers.deinit(m.gpa);
        m.host_meta.free();
    }

    pub fn upload(m: *Model, dst: u64, bytes: []const u8) !void {
        try m.d.check(m.d.api.cuMemcpyHtoDAsync_v2(dst, bytes.ptr, bytes.len, m.s.handle), "Llama upload");
    }

    pub fn download(m: *Model, src: u64, out: []u8) !void {
        try m.s.synchronize();
        m.meta_pending = false;
        try m.d.check(m.d.api.cuMemcpyDtoH_v2(out.ptr, src, out.len), "Llama download");
    }

    pub fn begin(m: *Model, token: u32, position: usize) !void {
        const c = m.w.config;
        if (token >= c.vocab) return error.BadToken;
        if (position >= m.context) return error.PastContext;
        if (position == 0 and token != c.bos) return error.MissingBos;
        // A pending asynchronous metadata upload borrows host_meta until this stream drains.
        if (m.meta_pending) try m.s.synchronize();
        m.position = position;
        const ints = m.host_meta.slice(i32);
        @memset(ints, 0);
        ints[0] = @intCast(position);
        ints[2] = 0;
        ints[3] = 1;
        ints[4] = @intCast(position);
        ints[6] = 0;
        ints[7] = 1;
        m.meta_pending = true;
        try m.upload(m.meta, m.host_meta.bytes);
        const row = m.w.embed.?.bytes[@as(usize, token) * c.hidden * 2 ..][0 .. c.hidden * 2];
        try m.upload(m.embed, row);
        try m.k.widen(m.s, m.embed, m.res, c.hidden);
    }

    pub fn layer(m: *Model, index: usize) !void {
        const c = m.w.config;
        const l = m.w.layers[index];
        const ex = &m.k.exl;
        try m.k.norms.rms(m.s, m.res, .f32, l.input_norm, m.x, .f16, 1, c.hidden, c.eps);
        try ex.decode(m.s, l.q, m.x, .f16, 1, m.qkv, .bf16);
        try ex.decode(m.s, l.k, m.x, .f16, 1, m.qkv + c.qDim() * 2, .bf16);
        try ex.decode(m.s, l.v, m.x, .f16, 1, m.qkv + (c.qDim() + c.kvDim()) * 2, .bf16);
        const offset = index * m.context * c.kvDim() * 2;
        const kc = m.kc + offset;
        const vc = m.vc + offset;
        try m.k.qk.rows(m.s, m.qkv, .{ .h = c.heads, .hk = c.kv_heads, .dim = c.head_dim }, .{ .inv = m.inv, .rot = c.head_dim }, m.meta, null, .{ .q = m.q, .k = m.kn, .v = m.vn, .kc = kc, .vc = vc, .len = m.context }, 1);
        const rows: cuda.tree_attention.Rows = .{ .row_stream = m.meta + 4, .streams = m.meta + 8, .offs = m.offsets + index * 16, .origin = m.kc, .paths = m.meta + 24, .depths = m.meta + 28 };
        const groups = cuda.tree_attention.TreeAttention.groups(m.position + 1);
        const sc = cuda.tree_attention.TreeAttention.scratch(m.scratch, 1, c.heads, c.kv_heads, groups, c.head_dim);
        try m.k.tree.run(m.s, m.q, m.kn, m.vn, rows, sc, m.attn, 1, c.heads, c.kv_heads, c.head_dim, groups, 1.0 / @sqrt(@as(f32, @floatFromInt(c.head_dim))));
        try ex.decode(m.s, l.o, m.attn, .bf16, 1, m.branch, .f32);
        try m.k.norms.addRms(m.s, m.branch, .f32, m.res, null, l.post_norm, m.x, .f16, 1, c.hidden, c.eps);
        try ex.decode(m.s, l.gate, m.x, .f16, 1, m.gate, .f32);
        try ex.decode(m.s, l.up, m.x, .f16, 1, m.up, .f32);
        try m.k.swiglu(m.s, m.gate, m.up, m.act, c.intermediate);
        try ex.decode(m.s, l.down, m.act, .f32, 1, m.branch, .f32);
        try m.k.add(m.s, m.res, m.branch, c.hidden);
    }

    pub fn head(m: *Model) !void {
        const c = m.w.config;
        try m.k.norms.rms(m.s, m.res, .f32, m.w.norm, m.x, .f16, 1, c.hidden, c.eps);
        try m.k.exl.decode(m.s, m.w.head, m.x, .f16, 1, m.logits, .f32);
    }

    pub fn step(m: *Model, token: u32, position: usize, logits: bool) !void {
        try m.begin(token, position);
        for (0..m.w.layers.len) |i| try m.layer(i);
        if (logits) try m.head();
    }

    pub fn greedy(m: *Model) !u32 {
        try m.k.argmax(m.s, m.logits, m.token, m.w.config.vocab);
        var token: u32 = undefined;
        try m.download(m.token, std.mem.asBytes(&token));
        return token;
    }

    pub fn pick(m: *Model, draw: sampling.Draw) !u32 {
        try m.sampler.?.draw(m.k, m.s, m.logits, m.token, draw);
        var token: u32 = undefined;
        try m.download(m.token, std.mem.asBytes(&token));
        return token;
    }
};
