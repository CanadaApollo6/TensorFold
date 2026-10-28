const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const check = @import("check.zig");
const rig_ = @import("contract_rig.zig");
const contract = core.contract;
const c = contract.spec;
const g = contract.gates;
const records = @import("core").contract.registry.dense_pointwise;

const Which = enum { swiglu, add, add_bf16 };
const shapes = [_]c.Shape{ .{ .rows = 1, .n = 7168, .k = 1 }, .{ .rows = 1, .n = 22016, .k = 1 } };

fn depth(_: c.Shape) usize {
    return 0;
}

pub const Point = struct {
    rig: rig_.Rig,
    f: cuda.Function,
    which: Which,
    wrong_operand: bool = false,
    x: []f32 = &.{},
    y: []f32 = &.{},
    xp: u64 = 0,
    yp: u64 = 0,
    out: u64 = 0,
    shape: c.Shape = .{ .rows = 0, .n = 0, .k = 0 },

    pub fn deinit(self: *Point) void {
        self.rig.gpu.gpa.free(self.x);
        self.rig.gpu.gpa.free(self.y);
        self.rig.deinit();
    }

    const vt = g.hooks(Point);

    pub fn kernel(self: *Point) g.Kernel {
        const names = [_][]const u8{ "cuda dense_pointwise swiglu fp32", "cuda dense_pointwise add fp32", "cuda dense_pointwise add bf16" };
        const spec: c.Spec = .{ .name = names[@backingInt(self.which)], .accum = .none, .depth = depth, .ops = if (self.which == .swiglu) 8 else 1, .out = if (self.which == .add_bf16) .bf16 else .f32, .in_place = self.which == .add, .tiles = .{ .{}, .{ .size = 256 }, .{} }, .served = &shapes };
        return .{ .spec = spec, .ctx = self, .vt = &vt, .control = if (self.wrong_operand) .shapes else null };
    }

    pub fn prepare(self: *Point, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        self.rig.reset();
        self.shape = shape;
        const a = self.rig.gpu.gpa;
        a.free(self.x);
        a.free(self.y);
        self.x = &.{};
        self.y = &.{};
        self.x = try a.alloc(f32, shape.rows * shape.n);
        self.y = try a.alloc(f32, shape.rows * shape.n);
        var prng = std.Random.DefaultPrng.init(seed);
        for (self.x, self.y) |*x, *y| {
            x.* = (prng.random().float(f32) * 2 - 1) * 4;
            y.* = (prng.random().float(f32) * 2 - 1) * 4;
        }
        if (fill == .nonfinite) {
            const plants = [_]f32{ std.math.nan(f32), std.math.inf(f32), -std.math.inf(f32), std.math.inf(f32) };
            for (plants, 0..) |v, i| self.x[i] = v;
            self.y[3] = -std.math.inf(f32);
        }
        self.xp = try self.rig.input(std.mem.sliceAsBytes(self.x));
        self.yp = try self.rig.input(std.mem.sliceAsBytes(self.y));
        self.out = try self.rig.output(shape.rows * shape.n * c.size(if (self.which == .add_bf16) .bf16 else .f32));
        return shape.rows > 0 and shape.n > 0 and shape.k == 1;
    }

    pub fn run(self: *Point) !void {
        var args: cuda.Args = .{};
        const up = if (self.wrong_operand) self.xp else self.yp;
        const n: i64 = @intCast(self.x.len);
        if (self.which == .swiglu) {
            for ([_]u64{ self.xp, self.shape.n, up, self.shape.n, self.out, self.shape.n }) |v| args.add(v);
            args.add(@as(i32, @intCast(self.shape.n)));
            args.add(std.math.inf(f32));
        } else if (self.which == .add) {
            try self.rig.gpu.d.check(self.rig.gpu.d.api.cuMemcpyHtoD_v2(self.out, self.x.ptr, self.x.len * 4), "pointwise residual reset");
            args.add(self.out);
            args.add(up);
            args.add(n);
        } else {
            args.add(self.xp);
            args.add(up);
            args.add(self.out);
            args.add(n);
        }
        const width = if (self.which == .swiglu) self.shape.n else self.x.len;
        try cuda.launch.launch(self.f, .{ .grid = .{ .x = @intCast((width + 255) / 256), .y = if (self.which == .swiglu) @intCast(self.shape.rows) else 1 }, .block = .{ .x = 256 } }, self.rig.s, &args);
        try self.rig.s.synchronize();
    }

    pub fn count(self: *Point) usize {
        return self.x.len;
    }

    pub fn reference(self: *Point, want: []f64, mag: []f64) !void {
        for (self.x, self.y, want, mag) |x, y, *w, *m| {
            w.* = if (self.which == .swiglu) @as(f64, x) / (1 + @exp(-@as(f64, x))) * y else @as(f64, x) + y;
            m.* = @abs(w.*);
        }
    }
};

pub const Set = struct {
    mod: ?cuda.Module = null,
    points: [6]Point = undefined,
    used: usize = 0,

    pub fn add(self: *Set, gpu: check.Gpu, s: cuda.Stream, kernels: *std.ArrayList(g.Kernel)) !void {
        if (try gpu.ctx.capability() != 86) return;
        self.mod = try cuda.Module.load(gpu.d, cuda.kernels.dense_pointwise);
        for (&self.points, 0..) |*point, i| {
            const which: Which = @fromBackingInt(@intCast(i % 3));
            const symbol = records[i % 3].launch.symbol;
            const z = try gpu.gpa.dupeSentinel(u8, symbol, 0);
            defer gpu.gpa.free(z);
            point.* = .{ .rig = .{ .gpu = gpu, .s = s }, .f = try self.mod.?.function(z), .which = which, .wrong_operand = i >= 3 };
            self.used += 1;
            try kernels.append(gpu.gpa, point.kernel());
        }
    }

    pub fn deinit(self: *Set) void {
        for (self.points[0..self.used]) |*point| point.deinit();
        if (self.mod) |*mod| mod.unload();
    }
};

pub fn run(gpu: check.Gpu) !void {
    if (try gpu.ctx.capability() != 86) return error.LlamaUnqualifiedGpu;
    var s = try cuda.Stream.init(gpu.d, false);
    defer s.deinit();
    var set: Set = .{};
    defer set.deinit();
    var kernels: std.ArrayList(g.Kernel) = .empty;
    defer kernels.deinit(gpu.gpa);
    try set.add(gpu, s, &kernels);
    var norms = try cuda.norms.Norms.load(gpu.d);
    defer norms.unload();
    var controls_mod = try cuda.Module.load(gpu.d, &@import("contract_limits.zig").control_bytes);
    defer controls_mod.unload();
    const row_k = @import("contract_rows.zig");
    var controls: [5]row_k.Row = undefined;
    for ([_]row_k.Which{ .fmax_max, .atomic_sum, .scribble_sum, .tail_sum, .half_dot }, &controls) |which, *row| {
        row.* = try row_k.Row.init(gpu, s, &norms, controls_mod, which);
        try kernels.append(gpu.gpa, row.kernel());
    }
    defer for (&controls) |*row| row.deinit();
    var tally: g.Tally = .{};
    try g.runAll(gpu.gpa, kernels.items, 0x6b63, &tally);
    var problems = try contract.registry.checkCuda(gpu.gpa, &records, false);
    const images = try contract.cubin.images(gpu.gpa, cuda.kernels.dense_pointwise);
    defer contract.cubin.freeImages(gpu.gpa, images);
    for (images) |image| {
        if (image.sm != 86) return error.UnqualifiedPointwiseImage;
        const compiled = try contract.cubin.kernels(gpu.gpa, image.elf, image.sm);
        defer gpu.gpa.free(compiled);
        for (records) |r| {
            const symbol = try gpu.gpa.dupeSentinel(u8, r.launch.symbol, 0);
            defer gpu.gpa.free(symbol);
            const f = try set.mod.?.function(symbol);
            const regs: u32 = @intCast(try f.attribute(.num_regs));
            const shared: u32 = @intCast(try f.attribute(.shared_size_bytes));
            const bound: u32 = @intCast(try f.attribute(.max_threads_per_block));
            const usage = for (compiled) |k| {
                if (std.mem.eql(u8, k.name, r.launch.symbol)) break k.usage;
            } else return error.MissingPointwiseSymbol;
            std.debug.print("RECORD dense_pointwise {s} sm=86 regs={d} static={d} bound={d} compiled={any}\n", .{ symbol, regs, shared, bound, usage });
            if (regs != r.usage[1].regs or shared != r.usage[1].static or usage.regs != regs or usage.bound != r.usage[1].bound or bound < 256) problems += 1;
        }
    }
    var messages: std.ArrayList(u8) = .empty;
    defer messages.deinit(gpu.gpa);
    var red_limits: usize = 0;
    for (contract.registry.cuda_controls) |control| {
        const symbol = try gpu.gpa.dupeSentinel(u8, control.launch.symbol, 0);
        defer gpu.gpa.free(symbol);
        const f = try controls_mod.function(symbol);
        const usage: contract.limits.Usage = .{ .regs = @intCast(try f.attribute(.num_regs)), .static = @intCast(try f.attribute(.shared_size_bytes)), .bound = @intCast(try f.attribute(.max_threads_per_block)) };
        if (try contract.limits.cuda(control.launch, usage, contract.limits.sms[1], &messages, gpu.gpa) > 0) red_limits += 1;
    }
    if (red_limits != 2) problems += 1;
    std.debug.print("RED limits controls={d}/2\n", .{red_limits});
    std.debug.print("SUMMARY dense_pointwise kernels={d} pass={d} fail={d} numerical_controls={d} red={d} missed={d} limits={d} sm=86\n", .{ tally.kernels, tally.passed, tally.failed, tally.controls, tally.caught, tally.missed, problems });
    if (tally.problems() + problems != 0) return error.TestFailed;
}
