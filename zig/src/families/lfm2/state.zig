//! A stream's committed state (each conv layer's two rows of B * x, each attention layer's keys and values) and the
//! forward's scratch.

const std = @import("std");
const cuda = @import("cuda");
const c = @import("shape.zig");
const kern = @import("kernels.zig");
const moe = @import("moe.zig");
const Arena = @import("qwen27").state.Arena;

/// The most rows a decode window takes (the decode matmuls' tile); wider calls are prompt chunks.
pub const max_rows = 16;

pub const Seq = struct {
    arena: Arena,
    conv: []u64, // by layer: bf16 [2, hidden] (older, newer), 0 for attention layers
    keys: []u64, // by layer: bf16 [capacity, kv heads, head dim], 0 for conv layers
    values: []u64,
    head_in: u64, // the last prompt row normed (bf16 [hidden]) and its group sums: the first pick's input
    head_xs: u64,
    capacity: usize,
    pos: usize = 0,
    gpa: std.mem.Allocator,

    pub fn bytes(g: c.Geometry, capacity: usize) usize {
        const a = std.mem.alignForward;
        var n: usize = a(usize, g.hidden * 2, 256) + a(usize, g.hidden / 64 * 4, 256);
        for (0..g.layers) |i| n += if (g.isAttention(i)) 2 * a(usize, capacity * g.kvInner() * 2, 256) else a(usize, 2 * g.hidden * 2, 256);
        return n;
    }

    pub fn init(gpa: std.mem.Allocator, ops: kern.Ops, g: c.Geometry, capacity: usize) !Seq {
        var s: Seq = .{ .gpa = gpa, .arena = try Arena.init(ops.k.base.d, bytes(g, capacity)), .conv = &.{}, .keys = &.{}, .values = &.{}, .head_in = 0, .head_xs = 0, .capacity = capacity };
        errdefer s.arena.deinit();
        s.conv = try gpa.alloc(u64, g.layers);
        errdefer gpa.free(s.conv);
        s.keys = try gpa.alloc(u64, g.layers);
        errdefer gpa.free(s.keys);
        s.values = try gpa.alloc(u64, g.layers);
        errdefer gpa.free(s.values);
        s.head_in = try s.arena.take(g.hidden * 2);
        s.head_xs = try s.arena.take(g.hidden / 64 * 4);
        for (0..g.layers) |i| {
            s.conv[i] = 0;
            s.keys[i] = 0;
            s.values[i] = 0;
            if (g.isAttention(i)) {
                s.keys[i] = try s.arena.take(capacity * g.kvInner() * 2);
                s.values[i] = try s.arena.take(capacity * g.kvInner() * 2);
            } else s.conv[i] = try s.arena.take(2 * g.hidden * 2);
        }
        try s.reset(ops, g);
        return s;
    }

    pub fn deinit(s: *Seq) void {
        s.arena.deinit();
        s.gpa.free(s.conv);
        s.gpa.free(s.keys);
        s.gpa.free(s.values);
        s.* = undefined;
    }

    /// Back to an empty prompt: zero conv rows (the conv's left padding).
    pub fn reset(s: *Seq, ops: kern.Ops, g: c.Geometry) !void {
        for (0..g.layers) |i| if (!g.isAttention(i)) try ops.b().fill32(s.conv[i], 0, g.hidden);
        s.pos = 0;
    }
};

pub const Scratch = struct {
    arena: Arena,
    rows: usize, // a call's rows at most
    x: u64, // the residual stream (two homes), the normed rows and their group sums
    h: u64,
    y: u64,
    xs: u64,
    r: u64, // the mixer's output
    pending: u64, // the MLP's output
    bcx: u64, // conv: in_proj's [B | C | x], the conv's output and its sums
    co: u64,
    co_xs: u64,
    q: u64, // attention: projections, normed and rotated rows, the heads' output and its sums
    k: u64,
    v: u64,
    qr: u64,
    kr: u64,
    vr: u64,
    attn: u64,
    attn_xs: u64,
    attn_part: u64,
    attn_sync: u64, // tf_lfm2_attn's counters: a row's key heads
    gate: u64, // the dense MLP
    up: u64,
    act: u64,
    act_xs: u64,
    moe: moe.Scratch,
    last_h: u64, // the head's input, logits and picks (a window's rows)
    last_xs: u64,
    logits: u64,
    picks: u64,
    pick_part: u64,
    ids: u64,
    pos: u64,
    log: u64, // graphed decode: the pick made at each position (i32 [capacity])
    gpa: std.mem.Allocator,

    pub fn init(gpa: std.mem.Allocator, d: *const cuda.Driver, g: c.Geometry, prompt_rows: usize, capacity: usize) !Scratch {
        var s: Scratch = undefined;
        s.gpa = gpa;
        s.arena = .{ .buf = null };
        try s.layout(g, prompt_rows, capacity);
        s.arena = try Arena.init(d, s.arena.used);
        errdefer s.arena.deinit();
        try s.layout(g, prompt_rows, capacity);
        return s;
    }

    fn layout(s: *Scratch, g: c.Geometry, prompt_rows: usize, capacity: usize) !void {
        const R = @max(prompt_rows, max_rows);
        const W = max_rows;
        const H = g.hidden;
        s.rows = R;
        s.arena.used = 0;
        const A = &s.arena;
        s.x = try A.take(R * H * 2);
        s.h = try A.take(R * H * 2);
        s.y = try A.take(R * H * 2);
        s.xs = try A.take(R * H / 64 * 4);
        s.r = try A.take(R * H * 2);
        s.pending = try A.take(R * H * 2);
        s.bcx = try A.take(R * 3 * H * 2);
        s.co = try A.take(R * H * 2);
        s.co_xs = try A.take(R * H / 64 * 4);
        s.q = try A.take(R * g.qInner() * 2);
        s.k = try A.take(R * g.kvInner() * 2);
        s.v = try A.take(R * g.kvInner() * 2);
        s.qr = try A.take(R * g.qInner() * 2);
        s.kr = try A.take(R * g.kvInner() * 2);
        s.vr = try A.take(R * g.kvInner() * 2);
        s.attn = try A.take(R * g.qInner() * 2);
        s.attn_xs = try A.take(R * g.qInner() / 64 * 4);
        var most: usize = 0; // a call's rows times splits at most
        for (1..R + 1) |n| most = @max(most, n * kern.Ops.attnSplits(n));
        s.attn_part = try A.take(most * g.heads * 66 * 4);
        s.attn_sync = try A.take(R * g.kv_heads * 4);
        s.gate = try A.take(R * g.intermediate * 2);
        s.up = try A.take(R * g.intermediate * 2);
        s.act = try A.take(R * g.intermediate * 2);
        s.act_xs = try A.take(R * g.intermediate / 64 * 4);
        s.moe = try moe.take(A, prompt_rows, W, g.experts, g.top_k, H, g.moe_inter);
        s.last_h = try A.take(W * H * 2);
        s.last_xs = try A.take(W * H / 64 * 4);
        s.logits = try A.take(W * g.vocab * 2);
        s.picks = try A.take(W * 4);
        s.pick_part = try A.take(W * kern.base.argmax_slices * 16);
        s.ids = try A.take(R * 4);
        s.pos = try A.take(R * 4);
        s.log = try A.take(capacity * 4);
    }

    pub fn deinit(s: *Scratch) void {
        s.arena.deinit();
        s.* = undefined;
    }
};
