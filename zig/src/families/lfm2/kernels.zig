//! LFM2's CUDA kernels: the qwen27 engine's matmuls, experts, glue and greedy pick, window attention, and lfm2.cu.

const std = @import("std");
const cuda = @import("cuda");
const q27 = @import("qwen27");

pub const base = q27.kernels;
pub const QLinear = base.QLinear;
pub const Window = cuda.window_attention.Window;

pub const Kernels = struct {
    base: base.Kernels, // dense matmuls, argmax, copies; glue: norms, act rows, qk rows, embed rows
    moe: q27.moe.Kernels,
    win: cuda.window_attention.WindowAttention,
    mod: cuda.Module,
    conv_rows: cuda.Function,
    conv_commit: cuda.Function,
    route: cuda.Function,

    pub fn load(gpa: std.mem.Allocator, io: std.Io, ctx: *const cuda.Context) !Kernels {
        var k: Kernels = undefined;
        k.base = try base.Kernels.load(gpa, io, ctx, "", 64);
        errdefer k.base.deinit();
        k.moe = try q27.moe.Kernels.load(ctx.d, k.base.sms);
        errdefer k.moe.unload();
        k.win = try cuda.window_attention.WindowAttention.load(ctx.d);
        errdefer k.win.unload();
        k.mod = try cuda.Module.load(ctx.d, cuda.kernels.lfm2);
        errdefer k.mod.unload();
        k.conv_rows = try k.mod.function("tf_lfm2_conv_rows");
        k.conv_commit = try k.mod.function("tf_lfm2_conv_commit");
        k.route = try k.mod.function("tf_lfm2_route");
        return k;
    }

    pub fn deinit(k: *Kernels) void {
        k.mod.unload();
        k.win.unload();
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
};
