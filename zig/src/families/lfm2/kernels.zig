//! LFM2's CUDA kernels: the qwen27 engine's matmuls, experts, glue and greedy pick, and lfm2.cu.

const std = @import("std");
const cuda = @import("cuda");
const q27 = @import("qwen27");

pub const base = q27.kernels;
pub const QLinear = base.QLinear;

pub const Kernels = struct {
    base: base.Kernels, // dense matmuls, argmax, copies; glue: norms, act rows, qk rows, embed rows
    moe: q27.moe.Kernels,
    splitk: [3][3]cuda.Function, // qmoe.cu's split-K decode: gate/up, down, down combined x 2, 4, 8 warps a unit
    mod: cuda.Module,
    conv_rows: cuda.Function,
    conv_commit: cuda.Function,
    route: cuda.Function,
    next: cuda.Function,
    attn: cuda.Function,

    pub fn load(gpa: std.mem.Allocator, io: std.Io, ctx: *const cuda.Context) !Kernels {
        var k: Kernels = undefined;
        k.base = try base.Kernels.load(gpa, io, ctx, "", 64);
        errdefer k.base.deinit();
        k.moe = try q27.moe.Kernels.load(ctx.d, k.base.sms);
        errdefer k.moe.unload();
        inline for (.{ "gu", "down", "dc" }, 0..) |form, f| inline for (.{ 2, 4, 8 }, 0..) |sp, si| {
            k.splitk[f][si] = try k.moe.mods[0].function(std.fmt.comptimePrint("tf_qmoe_splitk_{s}_b4_s{d}", .{ form, sp }));
        };
        k.mod = try cuda.Module.load(ctx.d, cuda.kernels.lfm2);
        errdefer k.mod.unload();
        k.conv_rows = try k.mod.function("tf_lfm2_conv_rows");
        k.conv_commit = try k.mod.function("tf_lfm2_conv_commit");
        k.route = try k.mod.function("tf_lfm2_route");
        k.next = try k.mod.function("tf_lfm2_next");
        k.attn = try k.mod.function("tf_lfm2_attn");
        return k;
    }

    pub fn deinit(k: *Kernels) void {
        k.mod.unload();
        k.moe.unload();
        k.base.deinit();
    }
};

fn int(x: usize) c_int {
    return @intCast(x);
}

pub const Ops = struct {
    k: *const Kernels,
    s: cuda.Stream,

    pub fn b(o: Ops) base.Ops {
        return .{ .k = &o.k.base, .s = o.s };
    }

    /// Up to 16 rows the decode matmuls (x with its group sums xs); a prompt chunk's rows, prefill_mm's tiles.
    pub fn mm(o: Ops, x: u64, xs: u64, q: QLinear, out: u64, rows: usize) !void {
        if (rows <= 16) return o.b().dense(x, xs, q, out, rows);
        return o.b().prefillDense(x, q, out, rows);
    }

    /// h = bf16(x + r) (r null: h = x), y = bf16(rms(h) * w), xs: y's 64-value sums.
    pub fn addRms(o: Ops, x: u64, r: ?u64, w: u64, h: u64, y: u64, xs: u64, rows: usize, width: usize, eps: f32) !void {
        try o.k.base.glue.norms.addRmsH(o.s, x, r, w, .bf16, h, y, xs, rows, width, eps);
    }

    pub fn groupSums(o: Ops, x: u64, ld: usize, xs: u64, rows: usize, width: usize) !void {
        try o.k.base.glue.act.groupSums(o.s, x, ld, xs, rows, width);
    }

    /// lfm2.cu's short conv over `rows` rows of in_proj's [B | C | x]; state moves past the first `commit` rows.
    pub fn convRows(o: Ops, bcx: u64, w: u64, state: u64, hist: ?u64, out: u64, rows: usize, h: usize, commit: usize) !void {
        var a: cuda.Args = .{};
        a.add(bcx);
        a.add(@as(c_longlong, @intCast(3 * h)));
        for ([_]u64{ w, state, hist orelse 0, out }) |v| a.add(v);
        a.add(int(rows));
        a.add(int(h));
        a.add(int(commit));
        try cuda.launch.launch(o.k.conv_rows, .{ .grid = .{ .x = @intCast((h + 255) / 256) }, .block = .{ .x = 256 } }, o.s, &a);
    }

    /// After a verify window: state moves past the first *accepted rows of hist.
    pub fn convCommit(o: Ops, state: u64, hist: u64, accepted: u64, h: usize) !void {
        var a: cuda.Args = .{};
        for ([_]u64{ state, hist, accepted }) |v| a.add(v);
        a.add(int(h));
        try cuda.launch.launch(o.k.conv_commit, .{ .grid = .{ .x = @intCast((h + 255) / 256) }, .block = .{ .x = 256 } }, o.s, &a);
    }

    /// A graphed decode step's tail: ids[0] = picks[0], log[pos] = it, pos += 1 (all on the device).
    pub fn next(o: Ops, picks: u64, ids: u64, pos: u64, log: u64) !void {
        var a: cuda.Args = .{};
        for ([_]u64{ picks, ids, pos, log }) |v| a.add(v);
        try cuda.launch.launch(o.k.next, .{ .grid = .{ .x = 1 }, .block = .{ .x = 1 } }, o.s, &a);
    }

    /// Splits a row's keys take (lfm2.cu's tf_lfm2_attn): about 256 blocks of 4 warps over `rows` rows and 8 key heads.
    pub fn attnSplits(rows: usize) usize {
        return std.math.clamp(32 / rows, 1, 32);
    }

    /// Attention of `rows` rows at pos (int32 [rows], keys 0 ..= pos) over the cache [capacity, hk, 64]: q and out
    /// [rows, 4 hk, 64]; `part`: rows * 4 hk * attnSplits(rows) * 66 fp32; counters rows * hk int32, zero.
    pub fn attn(o: Ops, q: u64, kc: u64, vc: u64, pos: u64, hk: usize, part: u64, counters: u64, out: u64, rows: usize) !void {
        const S = attnSplits(rows);
        const parts = rows * 4 * hk * S;
        var a: cuda.Args = .{};
        for ([_]u64{ q, kc, vc, pos }) |v| a.add(v);
        a.add(int(hk));
        a.add(@as(f32, 0.125));
        for ([_]u64{ part, part + parts * 64 * 4, part + parts * 65 * 4, counters, out }) |v| a.add(v);
        try cuda.launch.launch(o.k.attn, .{ .grid = .{ .x = @intCast(rows), .y = @intCast(hk), .z = @intCast(S) }, .block = .{ .x = 128 } }, o.s, &a);
    }
};
