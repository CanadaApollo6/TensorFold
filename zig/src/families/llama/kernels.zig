const std = @import("std");
const cuda = @import("cuda");

pub const Kernels = struct {
    exl: cuda.exl3.Dense,
    norms: cuda.norms.Norms,
    qk: cuda.qk_rows.QkRows,
    tree: cuda.tree_attention.TreeAttention,
    pointwise: cuda.Module,
    cast: cuda.Module,
    sample: cuda.Module,
    swiglu_fn: cuda.Function,
    add_fn: cuda.Function,
    widen_fn: cuda.Function,
    argmax_fn: cuda.Function,

    pub fn load(ctx: *const cuda.Context) !Kernels {
        if (!cuda.kernels.available) return error.BuiltWithoutKernels;
        const d = ctx.d;
        var k: Kernels = undefined;
        k.exl = try cuda.exl3.Dense.load(d);
        errdefer k.exl.unload();
        k.norms = try cuda.norms.Norms.load(d);
        errdefer k.norms.unload();
        k.qk = try cuda.qk_rows.QkRows.load(d);
        errdefer k.qk.unload();
        k.tree = try cuda.tree_attention.TreeAttention.load(d);
        errdefer k.tree.unload();
        k.tree.sms = @intCast(try ctx.attribute(.multiprocessor_count));
        k.pointwise = try cuda.Module.load(d, cuda.kernels.dense_pointwise);
        errdefer k.pointwise.unload();
        k.cast = try cuda.Module.load(d, cuda.kernels.torch_pointwise);
        errdefer k.cast.unload();
        k.sample = try cuda.Module.load(d, cuda.kernels.torch_argmax);
        errdefer k.sample.unload();
        k.swiglu_fn = try k.pointwise.function("_ZN15dsv41_pointwise13swiglu_kernelEPKfxS1_xPfxif");
        k.add_fn = try k.pointwise.function("_ZN15dsv41_pointwise10add_kernelEPfPKfx");
        k.widen_fn = try k.cast.function("tf_bf16_to_f32_kernel");
        k.argmax_fn = try k.sample.function("tf_argmax_rows_i32_kernel");
        return k;
    }

    pub fn deinit(k: *Kernels) void {
        k.sample.unload();
        k.cast.unload();
        k.pointwise.unload();
        k.tree.unload();
        k.qk.unload();
        k.norms.unload();
        k.exl.unload();
    }

    fn go(f: cuda.Function, s: cuda.Stream, count: usize, a: *cuda.Args) !void {
        try cuda.launch.launch(f, .{ .grid = .{ .x = @intCast((count + 255) / 256) }, .block = .{ .x = 256 } }, s, a);
    }

    pub fn widen(k: *const Kernels, s: cuda.Stream, x: u64, y: u64, n: usize) !void {
        var a: cuda.Args = .{};
        a.add(x);
        a.add(y);
        a.add(@as(u64, n));
        try go(k.widen_fn, s, n, &a);
    }

    pub fn add(k: *const Kernels, s: cuda.Stream, res: u64, x: u64, n: usize) !void {
        var a: cuda.Args = .{};
        a.add(res);
        a.add(x);
        a.add(@as(i64, @intCast(n)));
        try go(k.add_fn, s, n, &a);
    }

    pub fn swiglu(k: *const Kernels, s: cuda.Stream, gate: u64, up: u64, out: u64, n: usize) !void {
        var a: cuda.Args = .{};
        a.add(gate);
        a.add(@as(i64, @intCast(n)));
        a.add(up);
        a.add(@as(i64, @intCast(n)));
        a.add(out);
        a.add(@as(i64, @intCast(n)));
        a.add(@as(i32, @intCast(n)));
        a.add(std.math.inf(f32));
        try go(k.swiglu_fn, s, n, &a);
    }

    pub fn argmax(k: *const Kernels, s: cuda.Stream, logits: u64, out: u64, vocab: usize) !void {
        var a: cuda.Args = .{};
        a.add(logits);
        a.add(out);
        for ([_]usize{ 1, vocab, vocab }) |n| a.add(@as(i64, @intCast(n)));
        a.add(@as(i32, 1));
        try cuda.launch.launch(k.argmax_fn, .{ .grid = .{ .x = 1 }, .block = .{ .x = 1024 } }, s, &a);
    }
};
