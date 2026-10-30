const std = @import("std");
const op = @import("operators.zig");
const ck = @import("../qwen27/checkpoint.zig");
const weights = @import("weights.zig");
const Config = @import("config.zig").Config;
const storage = @import("cuda_weights.zig");
pub const split_bytes = 4 * (384 * 16 * 64 * 4 + 192 * 4);
pub const Plan = struct {
    a: std.mem.Allocator,
    format: storage.Format,
    vocab: u32,
    slots: std.ArrayList(usize) = .empty,
    cursor: usize = 0,
    fn self(p: *anyopaque) *Plan {
        return @ptrCast(@alignCast(p));
    }
    fn alloc(p: *Plan, rows: u32, width: u32, dtype: ck.DType) !op.View {
        const bytes = @as(usize, rows) * width * dtype.size();
        if (p.cursor == p.slots.items.len) try p.slots.append(p.a, bytes) else p.slots.items[p.cursor] = @max(p.slots.items[p.cursor], bytes);
        p.cursor += 1;
        return .{ .handle = 1, .rows = rows, .width = width, .stride = width, .dtype = dtype };
    }
    fn begin(p: *anyopaque) !u64 {
        self(p).cursor = 0;
        return 1;
    }
    fn release(_: *anyopaque, _: u64) void {}
    fn project(p: *anyopaque, _: u64, x: op.View, t: ck.Tensor, _: op.Mode) !op.View {
        const b = self(p);
        const y = try b.alloc(x.rows, @intCast(t.shape[0]), .bf16);
        _ = try b.alloc(x.rows, if (b.format == .bf16) y.width else x.width / 64, .f32);
        return y;
    }
    fn norm(p: *anyopaque, _: u64, x: op.View, _: op.Norm) !op.View {
        return self(p).alloc(x.rows, x.width, .bf16);
    }
    fn embed(p: *anyopaque, _: u64, t: ck.Linear, ids: []const u32) !op.View {
        return self(p).alloc(@intCast(ids.len), @intCast(t.dense.shape[1]), .bf16);
    }
    fn conv(p: *anyopaque, _: u64, x: op.View, _: op.Conv) !op.View {
        return self(p).alloc(x.rows, x.width, .bf16);
    }
    fn rope(p: *anyopaque, _: u64, x: op.View, _: op.Rotary) !op.View {
        const b = self(p);
        const y = try b.alloc(x.rows, x.width, .bf16);
        _ = try b.alloc(x.rows, x.width, .bf16);
        return y;
    }
    fn attention(p: *anyopaque, _: u64, x: op.Attention) !op.View {
        return self(p).alloc(x.query.rows, x.config.qWidth(), .bf16);
    }
    fn swiglu(p: *anyopaque, _: u64, x: op.View, _: op.View) !op.View {
        return self(p).alloc(x.rows, x.width, .bf16);
    }
    fn masks(_: *anyopaque, _: u64, x: op.View, _: u32, rows: u32) !op.View {
        var y = x;
        y.rows = rows;
        return y;
    }
    fn head(p: *anyopaque, _: u64, x: op.View) !op.Head {
        const b = self(p);
        _ = try b.alloc(x.rows, b.vocab, .f32);
        return .{ .logits = try b.alloc(x.rows, b.vocab, .bf16) };
    }
    fn lattice(_: *anyopaque, _: u64, _: op.Head, _: op.View, _: Config) !op.Lattice {
        return error.UnexpectedLattice;
    }
    fn lattices(_: *anyopaque, _: u64, _: op.Head, _: op.View, _: Config, _: u32, _: []op.Lattice) !void {
        return error.UnexpectedLattice;
    }
    fn contextBegin(_: *anyopaque, _: op.Context) !u64 {
        return 1;
    }
    fn append(_: *anyopaque, _: u64, _: u64, _: u32, _: op.View, _: op.View, _: []const u64) !void {}
    fn commit(_: *anyopaque, _: u64, _: u64, _: op.Context) !void {}
    fn abort(_: *anyopaque, _: u64) void {}
    fn streamAppend(_: *anyopaque, _: u64, _: u32, _: u32, _: op.View, _: op.View, _: []const u64) !void {}
    fn consume(ptr: *anyopaque, _: u64, output: @import("execution.zig").Output, c: Config) !void {
        const b = self(ptr);
        const l = @import("lattice.zig");
        const rows = output.head.logits.rows;
        _ = try b.alloc(rows, l.slices(c.vocab) * c.topk, .f32);
        _ = try b.alloc(rows, l.slices(c.vocab) * c.topk, .u32);
        _ = try b.alloc(rows, @sizeOf(l.Row) / 4, .u32);
    }
    const table: op.Ops.VTable = .{ .begin = begin, .release = release, .project = project, .norm = norm, .embed = embed, .conv = conv, .rope = rope, .attention = attention, .swiglu = swiglu, .mask_rows = masks, .head = head, .lattice = lattice, .lattices = lattices, .context_begin = contextBegin, .context_append = append, .context_commit = commit, .context_stage = commit, .context_abort = abort, .stream_append = streamAppend };
};
pub fn slots(a: std.mem.Allocator, graph: *const weights.Graph, format: storage.Format) ![]usize {
    const c = graph.config;
    var p: Plan = .{ .a = a, .vocab = c.vocab, .format = format };
    defer p.slots.deinit(a);
    const ops: op.Ops = .{ .ptr = &p, .vtable = &Plan.table };
    const preparation: op.Preparation = .{ .mode = .bf16_reference };
    const before: op.Context = .{ .cache = 1, .begin = 0, .end = 0 };
    const after: op.Context = .{ .cache = 1, .begin = 0, .end = 1 };
    const taps: op.View = .{ .handle = 1, .rows = 1, .width = c.tapWidth(), .stride = c.tapWidth(), .dtype = .bf16 };
    try @import("execution.zig").append(ops, graph, preparation, before, taps, &.{0}, after);
    try @import("execution.zig").forwardTo(ops, graph, preparation, after, 0, .{ .ptr = &p, .call = Plan.consume });
    return a.dupe(usize, p.slots.items);
}
pub fn fixed(c: Config, footprint: storage.Footprint, scratch_bytes: usize) usize {
    return footprint.device + @as(usize, c.layers) * c.window * c.kvWidth() * 4 + 8 * c.tapWidth() * 2 + split_bytes + scratch_bytes + @import("rotary.zig").device_bytes;
}

pub fn scratch(a: std.mem.Allocator, graph: *const weights.Graph, format: storage.Format) !usize {
    const sizes = try slots(a, graph, format);
    defer a.free(sizes);
    var total: usize = 0;
    for (sizes) |n| total += n;
    return total;
}
