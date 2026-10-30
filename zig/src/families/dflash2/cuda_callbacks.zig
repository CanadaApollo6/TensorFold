const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const op = @import("operators.zig");
const ck = @import("../qwen27/checkpoint.zig");
const Config = @import("config.zig").Config;
const Runtime = @import("cuda_runtime.zig").Runtime;
fn cast(p: *anyopaque) *Runtime {
    return @ptrCast(@alignCast(p));
}
pub const vtable: op.Ops.VTable = .{ .begin = begin, .release = release, .project = project, .norm = norm, .embed = embed, .conv = conv, .rope = rope, .attention = attention, .swiglu = swiglu, .mask_rows = masks, .head = head, .lattice = lattice, .lattices = lattices, .context_begin = contextBegin, .context_append = append, .context_commit = commit, .context_stage = stage, .context_abort = abort, .stream_append = streamAppend };
fn begin(p: *anyopaque) !u64 {
    const r = cast(p);
    if (r.active) return error.DraftFrameBusy;
    r.active = true;
    r.cursor = 0;
    r.rotary.begin();
    return 1;
}
fn release(p: *anyopaque, _: u64) void {
    const r = cast(p);
    r.s.synchronize() catch {};
    r.active = false;
}
fn narrow(r: *Runtime, x: u64, y: u64, count: usize) !void {
    try r.run(r.narrow_fn, .{ .x = @intCast((count + 255) / 256) }, 256, .{ x, y, @as(u64, count) });
}
fn project(p: *anyopaque, _: u64, x: op.View, t: ck.Tensor, _: op.Mode) !op.View {
    const r = cast(p);
    const timing = if (r.profile) |*pr| try pr.enter(r.s, .project, true) else null;
    defer if (r.profile) |*pr| pr.leave(r.s, timing);
    const y = try r.alloc(x.rows, @intCast(t.shape[0]), .bf16);
    const entry = try r.w.get(t);
    switch (entry.linear orelse return error.MissingDraftWeight) {
        .bf16 => |ptr| {
            const wide = try r.alloc(y.rows, y.width, .f32);
            try r.dense.rows(r.s, x.handle, x.stride, ptr, wide.handle, wide.stride, x.rows, y.width, x.width);
            try narrow(r, wide.handle, y.handle, @as(usize, y.rows) * y.width);
        },
        .affine4 => |q| {
            const sums = try r.alloc(x.rows, x.width / 64, .f32);
            try r.act.groupSums(r.s, x.handle, x.stride, sums.handle, x.rows, x.width);
            var at: u32 = 0;
            while (at < x.rows) : (at += 16) try r.affine.decode(r.s, x.handle + @as(usize, at) * x.stride * 2, sums.handle + @as(usize, at) * sums.stride * 4, q, y.handle + @as(usize, at) * y.stride * 2, @min(16, x.rows - at));
        },
    }
    return y;
}
fn norm(p: *anyopaque, _: u64, x: op.View, n: op.Norm) !op.View {
    const r = cast(p);
    const timing = if (r.profile) |*pr| try pr.enter(r.s, .norm, true) else null;
    defer if (r.profile) |*pr| pr.leave(r.s, timing);
    const y = try r.alloc(x.rows, x.width, .bf16);
    const e = try r.w.get(n.weight);
    try r.norms.rms(r.s, x.handle, .bf16, e.norm, y.handle, .bf16, @as(usize, x.rows) * n.heads, n.head_dim, n.eps);
    return y;
}
fn embed(p: *anyopaque, _: u64, binding: ck.Linear, ids: []const u32) !op.View {
    const r = cast(p);
    const timing = if (r.profile) |*pr| try pr.enter(r.s, .embed, false) else null;
    defer if (r.profile) |*pr| pr.leave(r.s, timing);
    const t = binding.dense;
    const y = try r.alloc(@intCast(ids.len), @intCast(t.shape[1]), .bf16);
    for (ids, 0..) |id, i| {
        if (id >= t.shape[0]) return error.BadToken;
        const width = t.shape[1] * 2;
        try r.upload(y.handle + i * width, t.bytes[@as(usize, id) * width ..][0..width]);
    }
    return y;
}
fn conv(p: *anyopaque, _: u64, x: op.View, c: op.Conv) !op.View {
    const r = cast(p);
    const timing = if (r.profile) |*pr| try pr.enter(r.s, .conv, true) else null;
    defer if (r.profile) |*pr| pr.leave(r.s, timing);
    if (c.block != x.rows) return error.BadRuntimeBlock;
    const y = try r.alloc(x.rows, x.width, .bf16);
    const base = (try r.w.get(c.base)).raw;
    try r.run(r.conv_fn, .{ .x = x.rows }, 256, .{ x.handle, c.dynamic.handle, base, if (c.residual) |v| v.handle else @as(u64, 0), y.handle, @as(i32, @intCast(x.width)), @as(i32, @intCast(c.group)), @as(i32, @intCast(c.branch)) });
    return y;
}
fn rope(p: *anyopaque, _: u64, x: op.View, c: op.Rotary) !op.View {
    const r = cast(p);
    const timing = if (r.profile) |*pr| try pr.enter(r.s, .rope, true) else null;
    defer if (r.profile) |*pr| pr.leave(r.s, timing);
    const y = try r.alloc(x.rows, x.width, .bf16);
    const other = try r.alloc(x.rows, x.width, .bf16);
    try r.rotary.rows(r.s, &r.qk, x, c, y.handle, other.handle);
    return y;
}
fn attention(p: *anyopaque, _: u64, x: op.Attention) !op.View {
    const r = cast(p);
    const timing = if (r.profile) |*pr| try pr.enter(r.s, .attention, true) else null;
    defer if (r.profile) |*pr| pr.leave(r.s, timing);
    const c = x.config;
    if (x.streams.len != 0 or !std.meta.eql(x.context, r.context)) return error.BadDraftContext;
    const y = try r.alloc(x.query.rows, c.qWidth(), .bf16);
    const size = @as(usize, c.window) * c.kvWidth() * 2;
    const k = r.ring.ptr + 2 * x.layer * size;
    try r.run(r.attn_fn, .{ .x = x.query.rows, .y = c.heads }, 32, .{ x.query.handle, x.keys.handle, x.values.handle, k, k + size, @as(i32, @intCast(x.query.rows)), @as(i32, @intCast(c.heads)), @as(i32, @intCast(c.kv_heads)), @as(i32, @intCast(c.window)), @as(i64, @intCast(r.context.begin)), @as(i64, @intCast(r.context.end)), @as(i32, @intCast(c.window)), y.handle });
    return y;
}
fn swiglu(p: *anyopaque, _: u64, gate: op.View, up: op.View) !op.View {
    const r = cast(p);
    const timing = if (r.profile) |*pr| try pr.enter(r.s, .swiglu, true) else null;
    defer if (r.profile) |*pr| pr.leave(r.s, timing);
    const y = try r.alloc(gate.rows, gate.width, .bf16);
    try r.act.swigluRound(r.s, gate.handle, gate.stride, up.handle, up.stride, y.handle, null, gate.rows, gate.width);
    return y;
}
fn masks(_: *anyopaque, _: u64, x: op.View, first: u32, rows: u32) !op.View {
    return Runtime.view(x.handle + @as(usize, first) * x.stride * 2, rows, x.width, .bf16);
}
fn head(p: *anyopaque, _: u64, x: op.View) !op.Head {
    const r = cast(p);
    const timing = if (r.profile) |*pr| try pr.enter(r.s, .head, true) else null;
    defer if (r.profile) |*pr| pr.leave(r.s, timing);
    const wide = try r.alloc(x.rows, r.graph.config.vocab, .f32);
    const y = try r.alloc(x.rows, r.graph.config.vocab, .bf16);
    try r.target.head(r.target.ptr, x.handle, wide.handle, x.rows);
    try narrow(r, wide.handle, y.handle, @as(usize, y.rows) * y.width);
    return .{ .logits = y };
}
fn lattice(p: *anyopaque, _: u64, h: op.Head, projected: op.View, c: Config) !op.Lattice {
    @setFloatMode(.strict);
    const r = cast(p);
    const rows = h.logits.rows;
    const compact = @import("lattice.zig");
    const count = compact.slices(c.vocab);
    const pv = try r.alloc(rows, count * c.topk, .f32);
    const pi = try r.alloc(rows, count * c.topk, .u32);
    const compact_rows = try r.alloc(rows, @sizeOf(compact.Row) / 4, .u32);
    const gpu_timing = if (r.profile) |*pr| try pr.enter(r.s, .lattice_gpu, true) else null;
    try r.run(r.topk_fn, .{ .x = count, .y = rows }, 256, .{ h.logits.handle, c.vocab, c.topk, count, pv.handle, pi.handle });
    try r.run(r.lattice_fn, .{ .x = rows }, 256, .{ h.logits.handle, c.vocab, c.topk, pv.handle, pi.handle, count * c.topk, compact_rows.handle });
    if (r.profile) |*pr| pr.leave(r.s, gpu_timing);
    const raw = try r.a.alloc(compact.Row, rows);
    defer r.a.free(raw);
    const download_timing = if (r.profile) |*pr| try pr.enter(r.s, .download, false) else null;
    try r.download(compact_rows.handle, std.mem.sliceAsBytes(raw));
    const hp = try r.a.alloc(u16, @as(usize, rows) * c.rank);
    defer r.a.free(hp);
    try r.download(projected.handle, std.mem.sliceAsBytes(hp));
    if (r.profile) |*pr| pr.leave(r.s, download_timing);
    var result: op.Lattice = .{ .gpa = r.a, .depth = rows, .topk = c.topk, .rank = c.rank, .candidates = try r.a.alloc(u32, @as(usize, rows) * c.topk), .unary = undefined, .projected = undefined };
    errdefer r.a.free(result.candidates);
    result.unary = try r.a.alloc(f32, result.candidates.len);
    errdefer r.a.free(result.unary);
    result.projected = try r.a.alloc(f32, hp.len);
    errdefer r.a.free(result.projected);
    for (result.projected, hp) |*v, b| v.* = core.affine4.asFloat(b);
    _ = try compact.unary(r.a, raw, h.logits.handle, h.logits.stride, c.vocab, c.topk, Fallback{ .r = r }, result.candidates, result.unary);
    return result;
}
const Fallback = struct {
    r: *Runtime,
    pub fn enter(f: Fallback) !?@import("cuda_profile.zig").Ticket {
        return if (f.r.profile) |*pr| try pr.enter(f.r.s, .lattice_fallback, false) else null;
    }
    pub fn leave(f: Fallback, ticket: ?@import("cuda_profile.zig").Ticket) void {
        if (f.r.profile) |*pr| pr.leave(f.r.s, ticket);
    }
    pub fn download(f: Fallback, address: u64, bytes: []u8) !void {
        try f.r.download(address, bytes);
    }
};
fn lattices(_: *anyopaque, _: u64, _: op.Head, _: op.View, _: Config, _: u32, _: []op.Lattice) !void {
    return error.BadRuntimeBlock;
}
fn contextBegin(p: *anyopaque, current: op.Context) !u64 {
    const r = cast(p);
    if (r.transaction or !std.meta.eql(current, r.context)) return error.BadDraftContext;
    r.transaction = true;
    return 1;
}
fn append(p: *anyopaque, _: u64, _: u64, layer: u32, keys: op.View, values: op.View, positions: []const u64) !void {
    const r = cast(p);
    const c = r.graph.config;
    const size = @as(usize, c.window) * c.kvWidth() * 2;
    for (positions, 0..) |pos, row| {
        const offset = @as(usize, pos % c.window) * c.kvWidth() * 2;
        const k = r.ring.ptr + 2 * layer * size + offset;
        try r.copy(k, keys.handle + row * keys.stride * 2, c.kvWidth() * 2);
        try r.copy(k + size, values.handle + row * values.stride * 2, c.kvWidth() * 2);
    }
}
fn commit(p: *anyopaque, _: u64, _: u64, next: op.Context) !void {
    const r = cast(p);
    try r.s.synchronize();
    r.context = next;
    r.transaction = false;
}
fn stage(p: *anyopaque, _: u64, _: u64, next: op.Context) !void {
    cast(p).context = next;
    cast(p).transaction = false;
}
fn abort(p: *anyopaque, _: u64) void {
    cast(p).transaction = false;
}
fn streamAppend(_: *anyopaque, _: u64, _: u32, _: u32, _: op.View, _: op.View, _: []const u64) !void {
    return error.BadRuntimeBlock;
}
