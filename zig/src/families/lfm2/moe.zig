//! LFM2-MoE's routed experts on the qwen27 engine's grouped kernels (qmoe*.cu), with no shared expert: a row's k pairs
//! come from lfm2.cu's sigmoid router (expert bias for the pick only, weights normalized), pair p = row * k + slot.

const std = @import("std");
const cuda = @import("cuda");
const qm = @import("qwen27").moe;

pub const tile = qm.tile; // a decode plan's item
pub const prefill_tile = qm.prefill_tile; // a prompt plan's item
const small = 1024; // pairs the one-block plan takes
const cols = 32; // a warp's output columns
const route_rows = 16; // lfm2.cu's router: a block's run of rows

/// A layer's router (fp32 [E, D]) and expert bias (fp32 [E]); experts as qwen27's blocks [E, N/32, K/64, M, block].
pub const Layer = struct { router: u64, bias: u64, up: u64, down: u64, bits: u8 };

pub const Scratch = struct { picks: u64, wts: u64, scores: u64, counters: u64, counter_words: usize, members: u64, items: u64, counts: u64, rank: u64, hist: u64, act: u64, y: u64, sync: u64, sync_words: usize };

/// Prompt chunks of `rows`, decode windows of `window` rows, from the arena (`A.take(bytes)`).
pub fn take(A: anytype, rows: usize, window: usize, experts: usize, top_k: usize, hidden: usize, width: usize) !Scratch {
    const R = @max(rows, window);
    const pairs = R * top_k;
    return .{
        .picks = try A.take(pairs * 4),
        .wts = try A.take(pairs * 4),
        .scores = try A.take(R * experts * 4),
        .counters = try A.take((R + route_rows - 1) / route_rows * 4),
        .counter_words = (R + route_rows - 1) / route_rows,
        .members = try A.take(pairs * 4),
        .items = try A.take(qm.maxItems(pairs, experts, tile) * 3 * 4),
        .counts = try A.take(16),
        .rank = try A.take(if (pairs > small) pairs * 4 else 16),
        .hist = try A.take(if (pairs > small) (pairs + small - 1) / small * experts * 4 else 16),
        .act = try A.take(pairs * width * 2),
        .y = try A.take(@max(rows * top_k * hidden * 2, window * top_k * hidden * 4)),
        .sync = try A.take(hidden / cols * 4),
        .sync_words = hidden / cols,
    };
}

fn prefillSmem(m: usize, bits: u8) u32 {
    const stage: usize = (64 * 8 + 4 * m * qm.block(bits)) * 16;
    const stages: usize = if (stage * 4 <= 49152) 4 else if (stage * 3 <= 49152) 3 else 2;
    return @intCast(stage * stages);
}

const Form = enum(u2) { gate_up, down, down_combine };
const form_m = [3]usize{ 2, 1, 1 };

/// Split-K decode warps a unit by form (TF_LFM2_SPLITK=gate_up,down to try others, 0 for swap_kernel).
fn splitFor(form: Form) usize {
    const default: usize = if (form == .gate_up) 4 else 2; // best of 2..8 (and 2..4 stages) on the 3090
    const v = std.mem.span(std.c.getenv("TF_LFM2_SPLITK") orelse return default);
    var it = std.mem.splitScalar(u8, v, ',');
    const a = std.fmt.parseInt(usize, it.next() orelse return 0, 10) catch return 0;
    const b = std.fmt.parseInt(usize, it.next() orelse return a, 10) catch return a;
    return if (form == .gate_up) a else b;
}

fn uc(v: usize) c_int {
    return @intCast(v);
}

fn go(f: cuda.Function, s: cuda.Stream, grid: [2]usize, threads: usize, smem: u32, a: *cuda.Args) !void {
    try cuda.launch.launch(f, .{ .grid = .{ .x = @intCast(grid[0]), .y = @intCast(grid[1]), .z = 1 }, .block = .{ .x = @intCast(threads) }, .shared = smem }, s, a);
}

pub const Shape = struct { hidden: usize, width: usize, experts: usize, top_k: usize };

pub const Moe = struct {
    k: *const qm.Kernels,
    splitk: *const [3][3]cuda.Function,
    route_fn: cuda.Function,
    s: cuda.Stream,
    sh: Shape,
    sc: Scratch,

    /// x [rows, D] bf16 -> out [rows, D] bf16: the k picked experts' SwiGLU outputs, weighted and summed in fp32.
    pub fn run(m: Moe, l: Layer, x: u64, out: u64, rows: usize, prompt: bool) !void {
        const sh = m.sh;
        const S = sh.top_k;
        const pairs = rows * S;
        var r: cuda.Args = .{};
        r.add(x);
        r.add(uc(sh.hidden));
        r.add(l.router);
        r.add(l.bias);
        for ([_]usize{ sh.experts, sh.hidden, S, rows }) |v| r.add(uc(v));
        for ([_]u64{ m.sc.scores, m.sc.counters, m.sc.picks, m.sc.wts }) |v| r.add(v);
        const one = rows == 1; // the router plans a lone row's experts too
        for ([_]u64{ m.sc.members, m.sc.items, m.sc.counts }) |v| r.add(if (one) v else 0);
        try go(m.route_fn, m.s, .{ sh.experts, (rows + route_rows - 1) / route_rows }, 256, 0, &r);
        if (!one) try m.plan(pairs, if (prompt) prefill_tile else tile);
        const b: usize = if (l.bits == 4) 1 else 0;
        const kg_up = sh.hidden / 64;
        const kg_down = sh.width / 64;
        const nb_up = sh.width / cols;
        const nb_down = sh.hidden / cols;
        if (prompt) {
            const items = qm.maxItems(pairs, sh.experts, prefill_tile);
            try m.prefill(m.k.prefill[0][b], prefillSmem(2, l.bits), x, sh.hidden, S, l.up, kg_up, nb_up, m.sc.act, sh.width, items);
            try m.prefill(m.k.prefill[1][b], prefillSmem(1, l.bits), m.sc.act, sh.width, 0, l.down, kg_down, nb_down, m.sc.y, sh.hidden, items);
            return m.combine(m.k.combine16, rows, out);
        }
        const items = qm.maxItems(pairs, sh.experts, tile);
        try m.swap(.gate_up, b, x, sh.hidden, S, l.up, kg_up, nb_up, m.sc.act, sh.width, items * nb_up, .{});
        if (rows == 1) return m.swap(.down_combine, b, m.sc.act, sh.width, 0, l.down, kg_down, nb_down, m.sc.y, sh.hidden, items * nb_down, .{ .w = m.sc.wts, .out = out, .sync = m.sc.sync, .slots = uc(S) });
        try m.swap(.down, b, m.sc.act, sh.width, 0, l.down, kg_down, nb_down, m.sc.y, sh.hidden, items * nb_down, .{});
        return m.combine(m.k.combine32, rows, out);
    }

    /// experts.route: the pairs grouped by expert into items of at most `t` pairs.
    fn plan(m: Moe, pairs: usize, t: usize) !void {
        const sc = m.sc;
        const ne = m.sh.experts;
        var a: cuda.Args = .{};
        if (pairs <= small) {
            a.add(sc.picks);
            for ([_]usize{ pairs, ne, t }) |v| a.add(uc(v));
            for ([_]u64{ sc.members, sc.items, sc.counts }) |v| a.add(v);
            return go(m.k.plan, m.s, .{ 1, 1 }, 1024, 0, &a);
        }
        const nblk = (pairs + small - 1) / small;
        a.add(sc.picks);
        a.add(uc(pairs));
        a.add(uc(ne));
        a.add(sc.rank);
        a.add(sc.hist);
        try go(m.k.rank, m.s, .{ nblk, 1 }, 1024, 0, &a);
        var b: cuda.Args = .{};
        for ([_]usize{ nblk, ne, t }) |v| b.add(uc(v));
        for ([_]u64{ sc.hist, sc.items, sc.counts }) |v| b.add(v);
        try go(m.k.offsets, m.s, .{ 1, 1 }, 1024, 0, &b);
        var c: cuda.Args = .{};
        c.add(sc.picks);
        c.add(uc(pairs));
        c.add(uc(ne));
        for ([_]u64{ sc.rank, sc.hist, sc.members }) |v| c.add(v);
        try go(m.k.scatter, m.s, .{ (pairs + 255) / 256, 1 }, 256, 0, &c);
    }

    fn args(m: Moe, a: *cuda.Args, x: u64, xs: usize, slots: usize, w: u64, kg: usize, nb: usize, out: u64, n: usize) void {
        a.add(x);
        a.add(uc(xs));
        a.add(uc(slots));
        a.add(w);
        a.add(uc(kg));
        a.add(uc(nb));
        for ([_]u64{ m.sc.items, m.sc.counts, m.sc.members, out }) |v| a.add(v);
        a.add(uc(n));
        a.add(@as(f32, 0));
        a.add(m.k.none.ptr);
        a.add(@as(c_int, -1));
    }

    fn swap(m: Moe, form: Form, b: usize, x: u64, xs: usize, slots: usize, w: u64, kg: usize, nb: usize, out: u64, n: usize, max_units: usize, comb: qm.Combine) !void {
        const f = @backingInt(form);
        const sk = splitFor(form);
        if (b == 1 and (sk == 2 or sk == 4 or sk == 8)) {
            if (max_units == 0) return;
            var a: cuda.Args = .{};
            a.add(x);
            a.add(uc(xs));
            a.add(uc(slots));
            a.add(w);
            a.add(uc(kg));
            a.add(uc(nb));
            for ([_]u64{ m.sc.items, m.sc.counts, m.sc.members, out }) |v| a.add(v);
            a.add(uc(n));
            a.add(@as(f32, 0));
            a.add(comb);
            return go(m.splitk[f][std.math.log2_int(usize, sk) - 1], m.s, .{ max_units, 1 }, sk * 32, 0, &a);
        }
        const s = qm.pickSplit(max_units, kg, m.k.sms);
        const si = std.math.log2_int(usize, s);
        const per_block: usize = if (s == 1) 4 else 1;
        const grid = @min((max_units + per_block - 1) / per_block, m.k.swap_grid[f][b][si]);
        if (grid == 0) return;
        var a: cuda.Args = .{};
        m.args(&a, x, xs, slots, w, kg, nb, out, n);
        a.add(comb);
        const smem: u32 = if (s == 1) 0 else @intCast(2 * s * (1152 * form_m[f] + 128));
        try go(m.k.swap[f][b][si], m.s, .{ grid, 1 }, (if (s == 1) 4 else s) * 32, smem, &a);
    }

    fn prefill(m: Moe, fun: cuda.Function, smem: u32, x: u64, xs: usize, slots: usize, w: u64, kg: usize, nb: usize, out: u64, n: usize, items: usize) !void {
        var a: cuda.Args = .{};
        m.args(&a, x, xs, slots, w, kg, nb, out, n);
        a.add(uc(items));
        try go(fun, m.s, .{ items * ((nb + 3) / 4), 1 }, 256, smem, &a);
    }

    fn combine(m: Moe, fun: cuda.Function, rows: usize, out: u64) !void {
        var a: cuda.Args = .{};
        a.add(m.sc.y);
        a.add(m.sc.wts);
        a.add(uc(m.sh.top_k));
        a.add(uc(m.sh.hidden));
        a.add(out);
        try go(fun, m.s, .{ (m.sh.hidden + 255) / 256, rows }, 256, 0, &a);
    }
};
