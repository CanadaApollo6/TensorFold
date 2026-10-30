const std = @import("std");
const cuda = @import("cuda");
const weights = @import("weights.zig");
const Kernels = @import("kernels.zig").Kernels;
const Config = @import("config.zig").Config;
const sampling = @import("sampling.zig");
const prof = @import("prefill_profile.zig");

pub const Model = struct {
    gpa: std.mem.Allocator,
    d: *const cuda.Driver,
    w: *const weights.Weights,
    k: *Kernels,
    s: cuda.Stream,
    context: usize,
    capacity_rows: usize = 1,
    prompt_rows: usize = 128,
    host_embed: cuda.HostBuffer,
    positions: u64 = 0,
    kv_mode: cuda.kv4.Mode = .bf16,
    kv4: ?cuda.kv4.Attention = null,
    first: u64 = 0,
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
    tap_ids: [5]u32 = @splat(0),
    tap_buffer: u64 = 0,
    profile: ?prof.Profile = null,

    pub fn init(a: std.mem.Allocator, d: *const cuda.Driver, w: *const weights.Weights, k: *Kernels, s: cuda.Stream, context: usize) !Model {
        return initMode(a, d, w, k, s, context, .bf16);
    }

    pub fn initMode(a: std.mem.Allocator, d: *const cuda.Driver, w: *const weights.Weights, k: *Kernels, s: cuda.Stream, context: usize, mode: cuda.kv4.Mode) !Model {
        return initRows(a, d, w, k, s, context, mode, 1);
    }

    pub fn initRows(a: std.mem.Allocator, d: *const cuda.Driver, w: *const weights.Weights, k: *Kernels, s: cuda.Stream, context: usize, mode: cuda.kv4.Mode, rows: usize) !Model {
        if (rows == 0 or rows > 8) return error.LlamaSerialOnly;
        if (context == 0 or context > w.config.max_context) return error.PastContext;
        const capacity = @max(rows, @import("prefill_plan.zig").rows());
        var m: Model = .{ .gpa = a, .d = d, .w = w, .k = k, .s = s, .context = context, .host_embed = try cuda.HostBuffer.alloc(d, capacity * w.config.hidden * 2), .host_meta = undefined, .groups = cuda.tree_attention.TreeAttention.groups(context) };
        m.host_meta = cuda.HostBuffer.alloc(d, @max(64, capacity * 4)) catch |err| {
            m.host_embed.free();
            return err;
        };
        errdefer m.deinit();
        if (prof.enabled()) m.profile = try prof.Profile.init(d);
        m.kv_mode = mode;
        if (mode == .q4) m.kv4 = try cuda.kv4.Attention.load(d);
        m.capacity_rows = capacity;
        m.prompt_rows = @import("prefill_plan.zig").rows();
        m.positions = try m.alloc(capacity * 4);
        const c = w.config;
        m.sampler = try sampling.Sampler.init(a, d, c.vocab, context);
        m.meta = try m.alloc(64);
        m.embed = try m.alloc(capacity * (c.hidden * 2));
        m.res = try m.alloc(capacity * (c.hidden * 4));
        m.x = try m.alloc(capacity * (c.hidden * 2));
        m.qkv = try m.alloc(capacity * ((c.qDim() + 2 * c.kvDim()) * 2));
        m.q = try m.alloc(capacity * (c.qDim() * 2));
        m.kn = try m.alloc(capacity * (c.kvDim() * 2));
        m.vn = try m.alloc(capacity * (c.kvDim() * 2));
        m.attn = try m.alloc(capacity * (c.qDim() * 2));
        m.branch = try m.alloc(capacity * (c.hidden * 4));
        m.gate = try m.alloc(capacity * (c.intermediate * 4));
        m.up = try m.alloc(capacity * (c.intermediate * 4));
        m.act = try m.alloc(capacity * (c.intermediate * 4));
        m.inv = try m.alloc(c.head_dim / 2 * 4);
        const kv_bytes = try mode.bytes(w.layers.len, context, c.kv_heads, c.head_dim);
        m.kc = try m.alloc(kv_bytes / 2);
        m.vc = try m.alloc(kv_bytes / 2);
        if (mode == .q4) m.first = try m.alloc(w.layers.len * c.kvDim() * 4);
        m.scratch = try m.alloc(if (mode == .q4) cuda.kv4.Attention.workBytes(c.heads, c.head_dim, context) else cuda.tree_attention.TreeAttention.scratchBytes(1, c.heads, c.kv_heads, m.groups, c.head_dim));
        m.offsets = try m.alloc(w.layers.len * 16);
        const offsets = try a.alloc(i64, w.layers.len * 2);
        defer a.free(offsets);
        for (0..w.layers.len) |i| {
            offsets[2 * i] = @intCast(i * context * c.kvDim());
            offsets[2 * i + 1] = @divExact(@as(i64, @intCast(m.vc)) - @as(i64, @intCast(m.kc)), 2) + offsets[2 * i];
        }
        try d.check(d.api.cuMemcpyHtoD_v2(m.offsets, offsets.ptr, offsets.len * 8), "KV offsets upload");
        m.logits = try m.alloc(@max(rows, 8) * (c.vocab * 4));
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
        if (m.profile) |*p| p.deinit();
        if (m.kv4) |*kv| kv.unload();
        if (m.sampler) |*sampler| sampler.deinit();
        for (m.buffers.items) |*b| b.free();
        m.buffers.deinit(m.gpa);
        m.host_meta.free();
        m.host_embed.free();
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
        m.position = position;
        try m.metadata(position);
        const row = m.w.embed.?.bytes[@as(usize, token) * c.hidden * 2 ..][0 .. c.hidden * 2];
        try m.upload(m.embed, row);
        try m.k.widen(m.s, m.embed, m.res, c.hidden);
    }

    fn metadata(m: *Model, position: usize) !void {
        // A pending asynchronous metadata upload borrows host_meta until this stream drains.
        if (m.meta_pending) try m.s.synchronize();
        const ints = m.host_meta.slice(i32);
        @memset(ints, 0);
        ints[0] = @intCast(position);
        ints[2] = 0;
        ints[3] = 1;
        ints[4] = @intCast(position);
        ints[6] = 0;
        ints[7] = 1;
        m.meta_pending = true;
        try m.upload(m.meta, m.host_meta.bytes[0..64]);
    }

    pub fn batch(m: *Model, tokens: []const u32, position: usize) !void {
        if (tokens.len > 8) return error.LlamaSerialOnly;
        if (tokens.len == 1) return m.step(tokens[0], position, true);
        try m.runRows(tokens, position, false, true);
    }

    pub fn prefill(m: *Model, tokens: []const u32, position: usize, last: bool) !void {
        try m.runRows(tokens, position, true, last);
    }

    fn runRows(m: *Model, tokens: []const u32, position: usize, prompt: bool, last: bool) !void {
        const c = m.w.config;
        if (tokens.len == 0 or tokens.len > m.capacity_rows or tokens.len > m.context -| position) return error.PastContext;
        if (position == 0 and tokens[0] != c.bos) return error.MissingBos;
        if (m.meta_pending) try m.s.synchronize();
        const pf = m.prefillProfile(prompt);
        if (pf) |p| try p.begin(m.s, position, tokens.len);
        var t = try prof.enter(pf, m.s, .embed);
        const positions = m.host_meta.slice(i32);
        for (tokens, 0..) |token, row| {
            if (token >= c.vocab) return error.BadToken;
            const bytes = m.w.embed.?.bytes[@as(usize, token) * c.hidden * 2 ..][0 .. c.hidden * 2];
            @memcpy(m.host_embed.bytes[row * c.hidden * 2 ..][0 .. c.hidden * 2], bytes);
            positions[row] = @intCast(position + row);
        }
        try m.upload(m.embed, m.host_embed.bytes[0 .. tokens.len * c.hidden * 2]);
        try m.upload(m.positions, std.mem.sliceAsBytes(positions[0..tokens.len]));
        m.meta_pending = true;
        try m.k.widen(m.s, m.embed, m.res, tokens.len * c.hidden);
        try prof.leave(pf, m.s, t);
        m.position = position;
        for (0..m.w.layers.len) |i| {
            try m.layerRows(i, tokens.len, prompt);
            if (m.tap_buffer != 0) for (m.tap_ids, 0..) |id, plane| if (id == i) {
                for (0..tokens.len) |row| try m.k.narrow(m.s, m.res + row * c.hidden * 4, m.tap_buffer + (row * 5 + plane) * c.hidden * 2, c.hidden);
            };
        }
        t = try prof.enter(pf, m.s, .norm);
        try m.k.norms.rms(m.s, m.res, .f32, m.w.norm, m.x, .f16, tokens.len, c.hidden, c.eps);
        try prof.leave(pf, m.s, t);
        if (prompt) {
            t = try prof.enter(pf, m.s, .head);
            if (last) try m.k.exl.decode(m.s, m.w.head, m.x + (tokens.len - 1) * c.hidden * 2, .f16, 1, m.logits, .f32);
            try prof.leave(pf, m.s, t);
        } else try m.k.exl.decode(m.s, m.w.head, m.x, .f16, tokens.len, m.logits, .f32);
        if (pf) |p| try p.end(m.s, last);
    }

    /// Prompt rows only: decode and verify rows never record profile events.
    fn prefillProfile(m: *Model, prompt: bool) ?*prof.Profile {
        if (!prompt) return null;
        return if (m.profile) |*p| p else null;
    }

    pub fn layer(m: *Model, index: usize) !void {
        return m.layerRows(index, 1, false);
    }

    fn layerRows(m: *Model, index: usize, count: usize, prompt: bool) !void {
        const c = m.w.config;
        const l = m.w.layers[index];
        const ex = &m.k.exl;
        const project: *const fn (*cuda.exl3.Dense, cuda.Stream, cuda.exl3.Weight, u64, cuda.exl3.DType, usize, u64, cuda.exl3.DType) anyerror!void = if (prompt) cuda.exl3.Dense.prompt else cuda.exl3.Dense.decode;
        const pf = m.prefillProfile(prompt);
        var t = try prof.enter(pf, m.s, .norm);
        try m.k.norms.rms(m.s, m.res, .f32, l.input_norm, m.x, .f16, count, c.hidden, c.eps);
        try prof.leave(pf, m.s, t);
        t = try prof.enter(pf, m.s, .q);
        try project(ex, m.s, l.q, m.x, .f16, count, m.qkv, .bf16);
        try prof.leave(pf, m.s, t);
        t = try prof.enter(pf, m.s, .k);
        try project(ex, m.s, l.k, m.x, .f16, count, m.qkv + count * c.qDim() * 2, .bf16);
        try prof.leave(pf, m.s, t);
        t = try prof.enter(pf, m.s, .v);
        try project(ex, m.s, l.v, m.x, .f16, count, m.qkv + count * (c.qDim() + c.kvDim()) * 2, .bf16);
        try prof.leave(pf, m.s, t);
        const offset = index * m.context * c.kv_heads * m.kv_mode.rowBytes(c.head_dim);
        const kc = m.kc + offset;
        const vc = m.vc + offset;
        const source: cuda.qk_rows.Src = .{ .q = m.qkv, .q_ld = @intCast(c.qDim()), .q_hs = @intCast(c.head_dim), .k = m.qkv + count * c.qDim() * 2, .k_ld = @intCast(c.kvDim()), .k_hs = @intCast(c.head_dim), .v = m.qkv + count * (c.qDim() + c.kvDim()) * 2, .v_ld = @intCast(c.kvDim()), .v_hs = @intCast(c.head_dim) };
        t = try prof.enter(pf, m.s, .rope);
        try m.k.qk.rowsFrom(m.s, source, .{ .h = c.heads, .hk = c.kv_heads, .dim = c.head_dim }, .{ .inv = m.inv, .rot = c.head_dim }, if (count == 1 and !prompt) m.meta else m.positions, null, .{ .q = m.q, .k = m.kn, .v = m.vn, .kc = if (m.kv_mode == .bf16) kc else null, .vc = if (m.kv_mode == .bf16) vc else null, .len = m.context }, count);
        try prof.leave(pf, m.s, t);
        // KV4 rows all take the prompt tiles: 4-bit packing turns any cross-kernel ulp into flipped codes downstream
        if (prompt or m.kv4 != null) {
            t = try prof.enter(pf, m.s, .attention);
            try @import("prompt_attention.zig").run(m, index, count, kc, vc);
            try prof.leave(pf, m.s, t);
        } else {
            const first_position = m.position;
            for (0..count) |row| {
                m.position = first_position + row;
                const q = m.q + row * c.qDim() * 2;
                const kn = m.kn + row * c.kvDim() * 2;
                const vn = m.vn + row * c.kvDim() * 2;
                const attention_out = m.attn + row * c.qDim() * 2;
                if (count > 1) try m.metadata(m.position);
                const rows: cuda.tree_attention.Rows = .{ .row_stream = m.meta + 4, .streams = m.meta + 8, .offs = m.offsets + index * 16, .origin = m.kc, .paths = m.meta + 24, .depths = m.meta + 28 };
                const groups = cuda.tree_attention.TreeAttention.groups(m.position + 1);
                const sc = cuda.tree_attention.TreeAttention.scratch(m.scratch, 1, c.heads, c.kv_heads, groups, c.head_dim);
                try m.k.tree.run(m.s, q, kn, vn, rows, sc, attention_out, 1, c.heads, c.kv_heads, c.head_dim, groups, 1.0 / @sqrt(@as(f32, @floatFromInt(c.head_dim))));
            }
            m.position = first_position;
        }
        t = try prof.enter(pf, m.s, .o);
        try project(ex, m.s, l.o, m.attn, .bf16, count, m.branch, .f32);
        try prof.leave(pf, m.s, t);
        t = try prof.enter(pf, m.s, .norm);
        try m.k.norms.addRms(m.s, m.branch, .f32, m.res, null, l.post_norm, m.x, .f16, count, c.hidden, c.eps);
        try prof.leave(pf, m.s, t);
        t = try prof.enter(pf, m.s, .gate);
        try project(ex, m.s, l.gate, m.x, .f16, count, m.gate, .f32);
        try prof.leave(pf, m.s, t);
        t = try prof.enter(pf, m.s, .up);
        try project(ex, m.s, l.up, m.x, .f16, count, m.up, .f32);
        try prof.leave(pf, m.s, t);
        t = try prof.enter(pf, m.s, .swiglu);
        try m.k.swiglu(m.s, m.gate, m.up, m.act, count * c.intermediate);
        try prof.leave(pf, m.s, t);
        t = try prof.enter(pf, m.s, .down);
        try project(ex, m.s, l.down, m.act, .f32, count, m.branch, .f32);
        try prof.leave(pf, m.s, t);
        t = try prof.enter(pf, m.s, .residual);
        try m.k.add(m.s, m.res, m.branch, count * c.hidden);
        try prof.leave(pf, m.s, t);
    }

    pub fn head(m: *Model) !void {
        const c = m.w.config;
        try m.k.norms.rms(m.s, m.res, .f32, m.w.norm, m.x, .f16, 1, c.hidden, c.eps);
        try m.k.exl.decode(m.s, m.w.head, m.x, .f16, 1, m.logits, .f32);
    }

    pub fn step(m: *Model, token: u32, position: usize, logits: bool) !void {
        try m.begin(token, position);
        for (0..m.w.layers.len) |i| {
            try m.layer(i);
            if (m.tap_buffer != 0) for (m.tap_ids, 0..) |id, plane| if (id == i) {
                try m.k.narrow(m.s, m.res, m.tap_buffer + plane * m.w.config.hidden * 2, m.w.config.hidden);
            };
        }
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
