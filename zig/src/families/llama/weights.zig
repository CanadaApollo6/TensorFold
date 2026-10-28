const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const names = @import("names.zig");
const Config = @import("config.zig").Config;

pub const Layer = struct { input_norm: u64, q: cuda.exl3.Weight, k: cuda.exl3.Weight, v: cuda.exl3.Weight, o: cuda.exl3.Weight, post_norm: u64, gate: cuda.exl3.Weight, up: cuda.exl3.Weight, down: cuda.exl3.Weight };
pub const Weights = struct {
    gpa: std.mem.Allocator,
    config: Config,
    embed: ?cuda.HostBuffer = null,
    layers: []Layer = &.{},
    norm: u64 = 0,
    head: cuda.exl3.Weight = undefined,
    buffers: std.ArrayList(cuda.DeviceBuffer) = .empty,

    pub fn deinit(w: *Weights) void {
        for (w.buffers.items) |*b| b.free();
        w.buffers.deinit(w.gpa);
        if (w.embed) |*b| b.free();
        w.gpa.free(w.layers);
    }

    fn upload(w: *Weights, d: *const cuda.Driver, bytes: []const u8) !u64 {
        var b = try cuda.DeviceBuffer.fromHost(d, bytes);
        errdefer b.free();
        try w.buffers.append(w.gpa, b);
        return b.ptr;
    }

    fn normWeight(w: *Weights, d: *const cuda.Driver, t: core.checkpoint.Tensor) !u64 {
        const out = try w.gpa.alloc(u32, t.numel());
        defer w.gpa.free(out);
        for (out, 0..) |*v, i| v.* = @as(u32, std.mem.readInt(u16, t.bytes[2 * i ..][0..2], .little)) << 16;
        return w.upload(d, std.mem.sliceAsBytes(out));
    }

    fn linear(w: *Weights, d: *const cuda.Driver, l: names.Linear) !cuda.exl3.Weight {
        const st = cuda.exl3.strides(l.k, l.n, l.halves, false);
        return .{
            .trellis = try w.upload(d, l.trellis.bytes),
            .suh = try w.upload(d, l.suh.bytes),
            .svh = try w.upload(d, l.svh.bytes),
            .k = @intCast(l.k),
            .n = @intCast(l.n),
            .halves = l.halves,
            .codebook = .mul1,
            .stride_k = st.k,
            .stride_nb = st.nb,
        };
    }
};

pub fn deviceBytes(m: names.Map, count: usize) usize {
    var n = m.head.trellis.bytes.len + m.head.suh.bytes.len + m.head.svh.bytes.len + m.norm.numel() * 4;
    for (m.layers[0..count]) |l| {
        n += (l.input_norm.numel() + l.post_norm.numel()) * 4;
        inline for (.{ "q", "k", "v", "o", "gate", "up", "down" }) |key| {
            const p = @field(l, key);
            n += p.trellis.bytes.len + p.suh.bytes.len + p.svh.bytes.len;
        }
    }
    return n;
}

pub fn load(a: std.mem.Allocator, io: std.Io, ctx: *const cuda.Context, dir: []const u8, count: ?usize, context: usize) !Weights {
    const c = try Config.read(a, io, dir);
    const n = count orelse c.layers;
    if (n == 0 or n > c.layers or context == 0 or context > c.max_context) return error.UnsupportedLlamaShapes;
    var ck = try core.checkpoint.Checkpoint.openModel(a, io, dir);
    defer ck.close();
    var m = try names.map(a, &ck, c);
    defer m.deinit(a);
    const free = (try ctx.memInfo()).free;
    const need = deviceBytes(m, n) + try @import("memory.zig").runtimeBytes(c, n, context);
    if (need + (2 << 30) > free) {
        std.log.err("Llama needs {d} MiB plus a 2048 MiB desktop reserve; {d} MiB is free", .{ need >> 20, free >> 20 });
        return error.LlamaGpuBusy;
    }
    var w: Weights = .{ .gpa = a, .config = c };
    errdefer w.deinit();
    w.embed = try cuda.HostBuffer.alloc(ctx.d, m.embed.bytes.len);
    @memcpy(w.embed.?.bytes, m.embed.bytes);
    w.layers = try a.alloc(Layer, n);
    for (w.layers, m.layers[0..n]) |*l, t| {
        l.input_norm = try w.normWeight(ctx.d, t.input_norm);
        l.post_norm = try w.normWeight(ctx.d, t.post_norm);
        inline for (.{ "q", "k", "v", "o", "gate", "up", "down" }) |key| @field(l, key) = try w.linear(ctx.d, @field(t, key));
    }
    w.norm = try w.normWeight(ctx.d, m.norm);
    w.head = try w.linear(ctx.d, m.head);
    return w;
}
