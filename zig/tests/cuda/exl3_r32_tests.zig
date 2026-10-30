//! Synthetic Nanbeige projections, R32 versus R16 with identical tiles and production split-K plans.
const std = @import("std");
const cuda = @import("cuda");
const check = @import("check.zig");
const exl3 = cuda.exl3;
const Rig = @import("quant_ref_tests.zig").RigT;
const zeroed = @import("exl3_tests.zig").zeroed;
const dense = @import("exl3_dense_tests.zig");

const Shape = struct { k: usize, n: usize, sk: u32, wk: u32, iterations: usize };
const shapes = [_]Shape{
    .{ .k = 7168, .n = 7168, .sk = 4, .wk = 4, .iterations = 28 },
    .{ .k = 7168, .n = 1024, .sk = 8, .wk = 4, .iterations = 14 },
    .{ .k = 7168, .n = 22016, .sk = 1, .wk = 8, .iterations = 56 },
    .{ .k = 22016, .n = 7168, .sk = 4, .wk = 4, .iterations = 86 },
};

// 1..33 covers every residue mod 16/32; later tails cover the second, third and last 32-row passes.
const rows = blk: {
    var out: [42]usize = undefined;
    for (0..33) |i| out[i] = i + 1;
    out[33..].* = .{ 49, 63, 64, 65, 95, 96, 97, 127, 128 };
    break :blk out;
};

pub fn run(gpu: check.Gpu) !void {
    var lin = try exl3.Linear.load(gpu.d);
    defer lin.unload();
    var s = try cuda.Stream.init(gpu.d, false);
    defer s.deinit();
    var cases: usize = 0;
    for (shapes) |shape| {
        const p = exl3.plan(shape.k, shape.n, 192, 8);
        try check.expect(p.sk == shape.sk and p.wk == shape.wk and
            shape.k / 16 / p.sk / p.wk == shape.iterations, "exl3-r32-unit production plan changed", .{});
        for (exl3.variants) |v| {
            if (v.warps != p.wk) continue;
            try one(gpu, s, &lin, shape, v);
            cases += rows.len;
        }
    }
    std.debug.print("SUMMARY exl3-r32-unit shapes={d} variants_per_shape=30 rows={d} cases={d} bit_equal=PASS\n", .{ shapes.len, rows.len, cases });
}

fn one(gpu: check.Gpu, s: cuda.Stream, lin: *const exl3.Linear, shape: Shape, v: exl3.Variant) !void {
    var arena = std.heap.ArenaAllocator.init(gpu.gpa);
    defer arena.deinit();
    var prng = std.Random.DefaultPrng.init(0x32321616);
    var r: Rig = .{ .gpu = gpu, .a = arena.allocator(), .s = s, .rng = prng.random() };
    defer {
        s.synchronize() catch {};
        for (r.bufs.items) |*b| b.free();
        r.bufs.deinit(gpu.gpa);
    }
    const k = shape.k;
    const n = shape.n;
    const trellis, const suh, const svh = try dense.tensors(&r, k, n, .{ .halves = @intCast(v.halves) });
    const w = try dense.weight(&r, trellis, suh, svh, k, n, v.halves, v.codebook);
    const x = try r.bfs(128 * k, 1.0);
    const dx = try r.dev(std.mem.sliceAsBytes(x));
    const counters = try zeroed(&r, exl3.Linear.counterBytes(w));
    const xh = try zeroed(&r, 128 * k * 2);
    const z: u64 = if (shape.sk > 1) try zeroed(&r, exl3.Linear.zBytes(w, 128)) else 0;
    const outputs = [_]u64{ try r.zeros(128 * n * 4), try r.zeros(128 * n * 4) };
    var scratch = std.heap.ArenaAllocator.init(gpu.gpa);
    defer scratch.deinit();
    for (rows) |m| {
        _ = scratch.reset(.retain_capacity);
        r.a = scratch.allocator();
        var raw: [2][]const u8 = undefined;
        for (&raw, [_]bool{ false, true }, outputs) |*out, wide, dy| {
            const cb: cuda.DeviceBuffer = .{ .d = gpu.d, .ptr = counters, .len = exl3.Linear.counterBytes(w) };
            try cb.fill8(0, s.handle);
            const yb: cuda.DeviceBuffer = .{ .d = gpu.d, .ptr = dy, .len = 128 * n * 4 };
            try yb.fill8(if (wide) 0x7f else 0xff, s.handle);
            try lin.matmulPasses(s, w, dx, .bf16, m, dy, .f32, xh, z, counters, wide);
            out.* = try r.back(dy, m * n * 4);
        }
        const id = try std.fmt.allocPrint(r.a, "exl3-r32-unit k={d} n={d} halves={d} cb={s} sk={d} wk={d} m={d}", .{ k, n, v.halves, @tagName(v.codebook), shape.sk, shape.wk, m });
        try check.sameBytes(id, raw[1], raw[0]);
        for (0..m * n) |i| {
            const value = std.mem.bytesToValue(f32, raw[0][4 * i ..][0..4]);
            try check.expect(std.math.isFinite(value), "{s} nonfinite output at {d}", .{ id, i });
        }
    }
    check.pass("exl3-r32-unit k={d} n={d} halves={d} cb={s} sk={d} wk={d}", .{ k, n, v.halves, @tagName(v.codebook), shape.sk, shape.wk });
}
