const std = @import("std");
const cuda = @import("cuda");
const Config = @import("config.zig").Config;

pub const Room = struct { ctx: *anyopaque, read: *const fn (*anyopaque) ?u64 };

pub fn runtimeBytes(c: Config, layers: usize, context: usize) !usize {
    return runtimeBytesMode(c, layers, context, .bf16);
}

pub fn runtimeBytesMode(c: Config, layers: usize, context: usize, mode: cuda.kv4.Mode) !usize {
    const groups = cuda.tree_attention.TreeAttention.groups(context);
    const rows = 64 + c.hidden * 12 + (c.qDim() * 3 + c.kvDim() * 4) * 2 + c.intermediate * 12 +
        c.head_dim / 2 * 4 + layers * 16 + c.vocab * 4 + 4;
    const attention = if (mode == .q4) cuda.kv4.Attention.scratchBytes(c.heads, c.head_dim, context) else cuda.tree_attention.TreeAttention.scratchBytes(1, c.heads, c.kv_heads, groups, c.head_dim);
    var xh: usize = 0;
    var z: usize = 0;
    var counters: usize = 0;
    for ([_][2]usize{ .{ c.hidden, c.qDim() }, .{ c.hidden, c.kvDim() }, .{ c.qDim(), c.hidden }, .{ c.hidden, c.intermediate }, .{ c.intermediate, c.hidden }, .{ c.hidden, c.vocab } }) |shape| {
        const p = cuda.exl3.plan(shape[0], shape[1], 192, 8);
        xh = @max(xh, 128 * shape[0] * 2);
        z = @max(z, if (p.sk > 1) p.sk * 128 * shape[1] * 4 else 0);
        counters = @max(counters, 8 * (shape[1] / 128) * 4);
    }
    // reserve() allocates the replacement before freeing the previous EXL3 scratch.
    return (try mode.bytes(layers, context, c.kv_heads, c.head_dim)) + (if (mode == .q4) layers * c.kvDim() * 4 else @as(usize, 0)) + rows + attention + 2 * (xh + z + counters) +
        @import("sampling.zig").deviceBytes(c.vocab, context);
}

pub fn admit(c: Config, layers: usize, context: usize, room: ?Room) !void {
    return admitMode(c, layers, context, .bf16, room);
}

pub fn admitMode(c: Config, layers: usize, context: usize, mode: cuda.kv4.Mode, room: ?Room) !void {
    const probe = room orelse return;
    const available = probe.read(probe.ctx) orelse 0;
    if (try runtimeBytesMode(c, layers, context, mode) > available) return error.LlamaMemoryBudget;
}

test "Llama host budget refuses KV beyond a 15 GiB cap and preserves a larger reserve" {
    const c: Config = .{ .layers = 52, .hidden = 7168, .heads = 56, .kv_heads = 8, .head_dim = 128, .intermediate = 22016, .vocab = 166912, .max_context = 524288, .bos = 166100, .rope_theta = 120000000, .eps = 1e-5 };
    const Probe = struct {
        left: ?u64,
        fn read(p: *anyopaque) ?u64 {
            const self: *@This() = @ptrCast(@alignCast(p));
            return self.left;
        }
    };
    var probe: Probe = .{ .left = (15 << 30) - 14355928192 };
    const room: Room = .{ .ctx = &probe, .read = Probe.read };
    try std.testing.expect(probe.left.? < try c.kvBytes(52, 16384));
    try std.testing.expectError(error.LlamaMemoryBudget, admit(c, 52, 16384, room));
    probe.left = (8 << 30) - (6 << 30);
    try std.testing.expectError(error.LlamaMemoryBudget, admit(c, 52, 16384, room));
    const need = try runtimeBytes(c, 52, 16384);
    probe.left = need - 1;
    try std.testing.expectError(error.LlamaMemoryBudget, admit(c, 52, 16384, room));
    probe.left = need;
    try admit(c, 52, 16384, room);
    probe.left = 0;
    try std.testing.expectError(error.LlamaMemoryBudget, admit(c, 52, 16384, room));
    const q4_need = try runtimeBytesMode(c, 52, 115488, .q4);
    try std.testing.expect(q4_need < try runtimeBytes(c, 52, 43219));
    probe.left = q4_need;
    try admitMode(c, 52, 115488, .q4, room);
    probe.left = q4_need - 1;
    try std.testing.expectError(error.LlamaMemoryBudget, admitMode(c, 52, 115488, .q4, room));
    probe.left = null;
    try std.testing.expectError(error.LlamaMemoryBudget, admit(c, 52, 16384, room));
}
