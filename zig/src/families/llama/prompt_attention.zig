const plan = @import("prefill_plan.zig");

pub fn run(m: anytype, index: usize, count: usize, kc: u64, vc: u64) !void {
    const c = m.w.config;
    if (m.kv4) |*kv| {
        const at = plan.cacheOffset(m.position, c.kv_heads, m.kv_mode.rowBytes(c.head_dim));
        try kv.pack(m.s, m.kn, kc + at, count * c.kv_heads, c.head_dim);
        try kv.pack(m.s, m.vn, vc + at, count * c.kv_heads, c.head_dim);
        const fk = m.first + index * c.kvDim() * 4;
        const fv = fk + c.kvDim() * 2;
        if (m.position == 0) {
            try m.d.check(m.d.api.cuMemcpyDtoDAsync_v2(fk, m.kn, c.kvDim() * 2, m.s.handle), "prompt BOS key");
            try m.d.check(m.d.api.cuMemcpyDtoDAsync_v2(fv, m.vn, c.kvDim() * 2, m.s.handle), "prompt BOS value");
        }
        var row: usize = 0;
        while (row < count) : (row += plan.attention_rows) {
            const n = @min(plan.attention_rows, count - row);
            try kv.prompt(m.s, .{ .q = m.q + row * c.qDim() * 2, .k = kc, .v = vc, .first_k = fk, .first_v = fv, .last_k = m.kn + row * c.kvDim() * 2, .last_v = m.vn + row * c.kvDim() * 2 }, m.scratch, m.attn + row * c.qDim() * 2, c.heads, c.kv_heads, c.head_dim, m.position + row, n);
        }
    } else try m.k.promptAttention(m.s, m.q, kc, vc, m.attn, m.position, count, c.heads, c.kv_heads);
}

test "KV4 prompt waves write absolute cache positions and retain current-row exceptions" {
    const std = @import("std");
    const Trace = struct {
        destinations: [2]u64 = @splat(0),
        pack_rows: usize = 0,
        packs: usize = 0,
        positions: [3]usize = @splat(0),
        counts: [3]usize = @splat(0),
        last_keys: [3]u64 = @splat(0),
        waves: usize = 0,
    };
    const Kv = struct {
        trace: *Trace,
        pub fn pack(k: *@This(), _: anytype, _: u64, dst: u64, rows: usize, _: usize) !void {
            k.trace.destinations[k.trace.packs] = dst;
            k.trace.pack_rows = rows;
            k.trace.packs += 1;
        }
        pub fn prompt(k: *@This(), _: anytype, r: anytype, _: u64, _: u64, _: usize, _: usize, _: usize, position: usize, count: usize) !void {
            const t = k.trace;
            t.positions[t.waves] = position;
            t.counts[t.waves] = count;
            t.last_keys[t.waves] = r.last_k;
            t.waves += 1;
        }
    };
    const Driver = struct {
        const Api = struct {
            cuMemcpyDtoDAsync_v2: *const fn (u64, u64, usize, usize) u32 = copy,
            fn copy(_: u64, _: u64, _: usize, _: usize) u32 {
                return 0;
            }
        };
        api: Api = .{},
        pub fn check(_: @This(), _: u32, _: []const u8) !void {}
    };
    const Kernels = struct {
        pub fn promptAttention(_: @This(), _: anytype, _: u64, _: u64, _: u64, _: u64, _: usize, _: usize, _: usize, _: usize) !void {}
    };
    var trace: Trace = .{};
    var m = .{
        .w = .{ .config = @import("config.zig").Config{ .layers = 52, .hidden = 7168, .heads = 56, .kv_heads = 8, .head_dim = 128, .intermediate = 22016, .vocab = 166912, .max_context = 524288, .bos = 166100, .rope_theta = 120000000, .eps = 1e-5 } },
        .kv4 = @as(?Kv, .{ .trace = &trace }),
        .kv_mode = @import("cuda").kv4.Mode.q4,
        .position = @as(usize, 127),
        .kn = @as(u64, 300000),
        .vn = @as(u64, 400000),
        .first = @as(u64, 500000),
        .q = @as(u64, 600000),
        .attn = @as(u64, 700000),
        .scratch = @as(u64, 800000),
        .s = .{ .handle = @as(usize, 0) },
        .d = Driver{},
        .k = Kernels{},
    };
    try run(&m, 2, 17, 100000, 200000);
    try std.testing.expectEqualSlices(u64, &.{ 100000 + 127 * 576, 200000 + 127 * 576 }, &trace.destinations);
    try std.testing.expectEqual(@as(usize, 17 * 8), trace.pack_rows);
    try std.testing.expectEqualSlices(usize, &.{ 127, 135, 143 }, &trace.positions);
    try std.testing.expectEqualSlices(usize, &.{ 8, 8, 1 }, &trace.counts);
    try std.testing.expectEqualSlices(u64, &.{ 300000, 300000 + 8 * 2048, 300000 + 16 * 2048 }, &trace.last_keys);
}
