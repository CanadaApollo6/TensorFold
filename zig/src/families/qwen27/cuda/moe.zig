//! qwen3_5_moe's routed experts on the qwen27 CUDA engine: the Python fork's grouped kernels (qmoe*.cu), 3- and 4-bit.
//! A row routes on its own: softmax top-k over fp32 router logits, then the shared expert as slot k by its sigmoid gate.

const std = @import("std");
const cuda = @import("cuda");

pub const tile = 16; // a decode plan's item (experts.TILE)
pub const prefill_tile = 64; // a prompt plan's item (experts.PREFILL_TILE)
const small = 1024; // pairs the one-block plan takes
const select_pairs = 256; // pairs select_kernel routes and plans in one launch
const cols = 32; // a warp's output columns (experts.COLS)
const want_warps = 8; // pick_split's warps an SM

/// A layer's experts as the kernels read them: blocks [E', N/32, K/64, M, block] words, the shared expert last at 4 bits
/// (E' = E + 1); below 4 bits it sits apart in `up4`/`down4` as expert `e4` = E.
pub const Layer = struct { router: u64, up: u64, down: u64, up4: u64 = 0, down4: u64 = 0, bits: u8, e4: c_int = -1 };

/// qmoe.cu's Combine, by value: down's fused weighted sum of a row's slots (decode, one row).
pub const Combine = extern struct { w: u64 = 0, out: u64 = 0, sync: u64 = 0, slots: c_int = 0 };

/// Uint4 words a (32-column block, group of 64) of `bits`-bit weights: the lanes' words, then scales and biases.
pub fn block(bits: u8) usize {
    return if (bits == 4) 72 else 56;
}

fn prefillSmem(m: usize, bits: u8) u32 {
    const stage: usize = (64 * 8 + 4 * m * block(bits)) * 16;
    const stages: usize = if (stage * 4 <= 49152) 4 else if (stage * 3 <= 49152) 3 else 2;
    return @intCast(stage * stages);
}

/// The decode kernel's form: gate/up with SwiGLU (bf16 act), down to fp32 pairs, or down with the slots' sum.
pub const Form = enum(u2) { gate_up, down, down_combine };
const form_m = [3]usize{ 2, 1, 1 };
const form_epi = [3]usize{ 2, 0, 4 };
const splits = [4]usize{ 1, 2, 4, 8 };

const sym = struct {
    fn swap(comptime m: usize, comptime epi: usize, comptime s: usize, comptime b: usize) [:0]const u8 {
        return std.fmt.comptimePrint("_ZN4qmoe11swap_kernelILi64ELi{d}ELi{d}ELi2ELi{d}ELi{d}EEEvPK13__nv_bfloat16iiPK5uint4iiPKiS8_S8_PvifS6_iNS_7CombineE", .{ m, epi, s, b });
    }
    fn prefill(comptime m: usize, comptime epi: usize, comptime b: usize) [:0]const u8 {
        return std.fmt.comptimePrint("_ZN4qmoe14prefill_kernelILi64ELi{d}ELi{d}ELi2ELi2ELi4ELi{d}EEEvPK13__nv_bfloat16iiPK5uint4iiPKiS8_S8_PvifS6_i", .{ m, epi, b });
    }
    const select = "_ZN4qmoe13select_kernelILi8EEEvPKfiiiiiPiPfS3_S3_S3_";
    const plan = "_ZN4qmoe11plan_kernelEPKiiiiPiS2_S2_";
    const rank = "_ZN4qmoe9plan_rankEPKiiiPiS2_";
    const offsets = "_ZN4qmoe12plan_offsetsEiiiPiS0_S0_";
    const scatter = "_ZN4qmoe12plan_scatterEPKiiiS1_S1_Pi";
    const pack = "_ZN4qmoe11pack_kernelILi2EEEvPKjPKtS4_Pjiiii";
};

/// Scratch for `rows` rows of `slots` pairs: logits [rows, E + 1] fp32, picks and weights [rows, slots], the plan,
/// act [pairs, NI] bf16, y [pairs, D] (fp32 decode, bf16 prompt), and down's combine counters (zero between launches).
pub const Scratch = struct { logits: u64, picks: u64, wts: u64, members: u64, items: u64, counts: u64, rank: u64, hist: u64, act: u64, y: u64, sync: u64, sync_words: usize };

pub fn maxItems(pairs: usize, experts: usize, t: usize) usize {
    return @min(pairs, experts) + pairs / t;
}

/// The scratch's arena takes (`A.take(bytes)`): prompt chunks of `rows`, decode windows of `window` rows.
pub fn take(A: anytype, rows: usize, window: usize, experts: usize, top_k: usize, hidden: usize, width: usize) !Scratch {
    const S = top_k + 1;
    const R = @max(rows, window);
    const pairs = R * S;
    const ne = experts + 1;
    return .{
        .logits = try A.take(R * ne * 4),
        .picks = try A.take(pairs * 4),
        .wts = try A.take(pairs * 4),
        .members = try A.take(pairs * 4),
        .items = try A.take(maxItems(pairs, ne, tile) * 3 * 4),
        .counts = try A.take(16),
        .rank = try A.take(if (pairs > small) pairs * 4 else 16),
        .hist = try A.take(if (pairs > small) (pairs + small - 1) / small * ne * 4 else 16),
        .act = try A.take(pairs * width * 2),
        .y = try A.take(@max(rows * S * hidden * 2, window * S * hidden * 4)),
        .sync = try A.take(hidden / cols * 4),
        .sync_words = hidden / cols,
    };
}

pub const Kernels = struct {
    mods: [5]cuda.Module,
    router: cuda.Function,
    router_tile: cuda.Function, // prompt rows at hidden 2048: the same sums, rows staged in shared memory
    topk_rows: cuda.Function,
    combine32: cuda.Function,
    combine16: cuda.Function,
    widen3: cuda.Function,
    dequant8: cuda.Function,
    planes3: cuda.Function,
    place: cuda.Function,
    select: cuda.Function,
    plan: cuda.Function,
    rank: cuda.Function,
    offsets: cuda.Function,
    scatter: cuda.Function,
    pack: cuda.Function,
    swap: [3][2][splits.len]cuda.Function, // form x bits (3, 4) x warps a unit
    swap_grid: [3][2][splits.len]usize, // resident blocks: per SM times SMs
    prefill: [2][2]cuda.Function, // gate/up, down (bf16 pairs) x bits (3, 4)
    sms: usize,
    none: cuda.DeviceBuffer, // w4 when there is no 4-bit shared expert apart

    pub fn load(d: *const cuda.Driver, sms: usize) !Kernels {
        var k: Kernels = undefined;
        const kk = cuda.kernels;
        const images = [_][]const u8{ kk.qmoe, kk.qmoe_plan, kk.qmoe_prefill, kk.qmoe_pack, kk.qmoe_glue };
        var loaded: usize = 0;
        errdefer for (k.mods[0..loaded]) |*m| m.unload();
        for (images, 0..) |img, i| {
            k.mods[i] = try cuda.Module.load(d, img);
            loaded += 1;
        }
        k.sms = sms;
        inline for (0..3) |f| inline for (0..2) |b| inline for (splits, 0..) |s, si| {
            const fun = try k.mods[0].function(sym.swap(form_m[f], form_epi[f], s, 3 + b));
            const smem: u32 = if (s == 1) 0 else @intCast(2 * s * (1152 * form_m[f] + 128));
            if (smem > 0) try fun.allowDynamicShared(smem);
            k.swap[f][b][si] = fun;
            k.swap_grid[f][b][si] = @max(1, try fun.occupancy(@intCast((if (s == 1) 4 else s) * 32), smem)) * sms;
        };
        inline for (0..2) |b| inline for (.{ .{ 2, 2 }, .{ 1, 3 } }, 0..) |me, f| {
            const fun = try k.mods[2].function(sym.prefill(me[0], me[1], 3 + b));
            try fun.allowDynamicShared(prefillSmem(me[0], 3 + b));
            k.prefill[f][b] = fun;
        };
        k.select = try k.mods[1].function(sym.select);
        k.plan = try k.mods[1].function(sym.plan);
        k.rank = try k.mods[1].function(sym.rank);
        k.offsets = try k.mods[1].function(sym.offsets);
        k.scatter = try k.mods[1].function(sym.scatter);
        k.pack = try k.mods[3].function(sym.pack);
        const g = k.mods[4];
        k.router = try g.function("tf_qmoe_router");
        k.router_tile = try g.function("tf_qmoe_router_tile");
        k.topk_rows = try g.function("tf_qmoe_topk_rows");
        k.combine32 = try g.function("tf_qmoe_combine_f32");
        k.combine16 = try g.function("tf_qmoe_combine_bf16");
        k.widen3 = try g.function("tf_qmoe_widen3");
        k.dequant8 = try g.function("tf_qmoe_dequant8");
        k.planes3 = try g.function("tf_qmoe_planes3");
        k.place = try g.function("tf_qmoe_place");
        k.none = try cuda.DeviceBuffer.alloc(d, 256);
        errdefer k.none.free();
        try k.none.fill32(0, null);
        return k;
    }

    pub fn unload(k: *Kernels) void {
        k.none.free();
        for (&k.mods) |*m| m.unload();
    }
};

fn uc(v: usize) c_int {
    return @intCast(v);
}

fn go(f: cuda.Function, s: cuda.Stream, grid: [2]usize, threads: usize, smem: u32, a: *cuda.Args) !void {
    try cuda.launch.launch(f, .{ .grid = .{ .x = @intCast(grid[0]), .y = @intCast(grid[1]), .z = 1 }, .block = .{ .x = @intCast(threads) }, .shared = smem }, s, a);
}

/// pick_split: warps a unit, up to about 8 an SM in all when units are few, never more than a unit has groups.
pub fn pickSplit(max_units: usize, kg: usize, sms: usize) usize {
    var s: usize = 1;
    while (2 * s <= 8 and 2 * s <= kg and max_units * 2 * s <= want_warps * sms) s *= 2;
    return s;
}

/// A layer's shapes: hidden D, expert width NI, routed experts E, top k.
pub const Shape = struct { hidden: usize, width: usize, experts: usize, top_k: usize };

pub const Moe = struct {
    k: *const Kernels,
    s: cuda.Stream,
    sh: Shape,
    sc: Scratch,

    /// moe.run: x [rows, D] bf16 (unit-stride rows) -> out [rows, D] bf16, the top-k experts' and the shared one's sum.
    pub fn run(m: Moe, l: Layer, x: u64, out: u64, rows: usize, prompt: bool) !void {
        const sh = m.sh;
        const S = sh.top_k + 1;
        const pairs = rows * S;
        const ne = sh.experts + 1;
        try m.router(l.router, x, rows);
        if (!prompt and pairs <= select_pairs) {
            var a: cuda.Args = .{};
            a.add(m.sc.logits);
            for ([_]usize{ rows, ne, sh.experts, sh.top_k, tile }) |v| a.add(uc(v));
            for ([_]u64{ m.sc.picks, m.sc.wts, m.sc.members, m.sc.items, m.sc.counts }) |v| a.add(v);
            try go(m.k.select, m.s, .{ 1, 1 }, select_pairs, 0, &a);
        } else {
            var a: cuda.Args = .{};
            a.add(m.sc.logits);
            for ([_]usize{ rows, ne, sh.experts, sh.top_k }) |v| a.add(uc(v));
            a.add(m.sc.picks);
            a.add(m.sc.wts);
            try go(m.k.topk_rows, m.s, .{ (rows + 7) / 8, 1 }, 256, 0, &a);
            try m.route(pairs, ne, if (prompt) prefill_tile else tile);
        }
        const b: usize = if (l.bits == 4) 1 else 0;
        const kg_up = sh.hidden / 64;
        const kg_down = sh.width / 64;
        const nb_up = sh.width / cols;
        const nb_down = sh.hidden / cols;
        if (prompt) {
            const items = maxItems(pairs, ne, prefill_tile);
            try m.prefill(m.k.prefill[0][b], prefillSmem(2, l.bits), x, sh.hidden, S, l.up, kg_up, nb_up, m.sc.act, sh.width, items, l.up4, l.e4);
            try m.prefill(m.k.prefill[1][b], prefillSmem(1, l.bits), m.sc.act, sh.width, 0, l.down, kg_down, nb_down, m.sc.y, sh.hidden, items, l.down4, l.e4);
            return m.combine(m.k.combine16, rows, out);
        }
        const items = maxItems(pairs, ne, tile);
        try m.swap(.gate_up, b, x, sh.hidden, S, l.up, kg_up, nb_up, m.sc.act, sh.width, items * nb_up, l.up4, l.e4, .{});
        if (rows == 1) return m.swap(.down_combine, b, m.sc.act, sh.width, 0, l.down, kg_down, nb_down, m.sc.y, sh.hidden, items * nb_down, l.down4, l.e4, .{ .w = m.sc.wts, .out = out, .sync = m.sc.sync, .slots = uc(S) });
        try m.swap(.down, b, m.sc.act, sh.width, 0, l.down, kg_down, nb_down, m.sc.y, sh.hidden, items * nb_down, l.down4, l.e4, .{});
        return m.combine(m.k.combine32, rows, out);
    }

    /// Router logits [rows, E + 1] fp32: the router's rows, then the shared expert's gate.
    fn router(m: Moe, w: u64, x: u64, rows: usize) !void {
        var a: cuda.Args = .{};
        a.add(x);
        a.add(uc(m.sh.hidden));
        a.add(w);
        for ([_]usize{ rows, m.sh.experts + 1, m.sh.hidden }) |v| a.add(uc(v));
        a.add(m.sc.logits);
        if (rows >= 64 and m.sh.hidden == 2048) return go(m.k.router_tile, m.s, .{ (m.sh.experts + 1 + 31) / 32, (rows + 7) / 8 }, 256, 0, &a);
        try go(m.k.router, m.s, .{ (m.sh.experts + 1 + 7) / 8, (rows + 15) / 16 }, 256, 0, &a);
    }

    /// experts.route: the pairs' picks grouped by expert into items of at most `t` pairs.
    fn route(m: Moe, pairs: usize, ne: usize, t: usize) !void {
        const sc = m.sc;
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

    fn args(m: Moe, a: *cuda.Args, x: u64, xs: usize, slots: usize, w: u64, kg: usize, nb: usize, out: u64, n: usize, w4: u64, e4: c_int) void {
        a.add(x);
        a.add(uc(xs));
        a.add(uc(slots));
        a.add(w);
        a.add(uc(kg));
        a.add(uc(nb));
        for ([_]u64{ m.sc.items, m.sc.counts, m.sc.members, out }) |v| a.add(v);
        a.add(uc(n));
        a.add(@as(f32, 0));
        a.add(if (w4 != 0) w4 else m.k.none.ptr);
        a.add(e4);
    }

    fn swap(m: Moe, form: Form, b: usize, x: u64, xs: usize, slots: usize, w: u64, kg: usize, nb: usize, out: u64, n: usize, max_units: usize, w4: u64, e4: c_int, comb: Combine) !void {
        const f = @intFromEnum(form);
        const s = pickSplit(max_units, kg, m.k.sms);
        const si = std.math.log2_int(usize, s);
        const per_block: usize = if (s == 1) 4 else 1;
        const grid = @min((max_units + per_block - 1) / per_block, m.k.swap_grid[f][b][si]);
        if (grid == 0) return;
        var a: cuda.Args = .{};
        m.args(&a, x, xs, slots, w, kg, nb, out, n, w4, e4);
        a.add(comb);
        const smem: u32 = if (s == 1) 0 else @intCast(2 * s * (1152 * form_m[f] + 128));
        try go(m.k.swap[f][b][si], m.s, .{ grid, 1 }, (if (s == 1) 4 else s) * 32, smem, &a);
    }

    fn prefill(m: Moe, fun: cuda.Function, smem: u32, x: u64, xs: usize, slots: usize, w: u64, kg: usize, nb: usize, out: u64, n: usize, items: usize, w4: u64, e4: c_int) !void {
        var a: cuda.Args = .{};
        m.args(&a, x, xs, slots, w, kg, nb, out, n, w4, e4);
        a.add(uc(items));
        try go(fun, m.s, .{ items * ((nb + 3) / 4), 1 }, 256, smem, &a);
    }

    /// moe.combine: each row's slots summed in slot order, fp32 w * y, rounded once to bf16.
    fn combine(m: Moe, fun: cuda.Function, rows: usize, out: u64) !void {
        var a: cuda.Args = .{};
        a.add(m.sc.y);
        a.add(m.sc.wts);
        a.add(uc(m.sh.top_k + 1));
        a.add(uc(m.sh.hidden));
        a.add(out);
        try go(fun, m.s, .{ (m.sh.hidden + 255) / 256, rows }, 256, 0, &a);
    }
};

test "pick_split as experts.cu's, 3090 (82 SMs)" {
    // one row, top 8 + shared: 9 items; gate/up 16 column blocks, K 32 groups; down 64 blocks, 8 groups
    try std.testing.expectEqual(@as(usize, 4), pickSplit(9 * 16, 32, 82));
    try std.testing.expectEqual(@as(usize, 1), pickSplit(9 * 64, 8, 82));
    try std.testing.expectEqual(@as(u32, 46080), prefillSmem(2, 3));
    try std.testing.expectEqual(@as(u32, 34816), prefillSmem(2, 4));
    try std.testing.expectEqual(@as(u32, 47104), prefillSmem(1, 3));
    try std.testing.expectEqual(@as(u32, 38400), prefillSmem(1, 4));
}
