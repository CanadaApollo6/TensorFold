const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const check = @import("check.zig");
const rig_ = @import("contract_rig.zig");
const c = core.contract.spec;
const g = core.contract.gates;
const shapes = [_]c.Shape{ .{ .rows = 1, .n = 128, .k = 32 }, .{ .rows = 9, .n = 256, .k = 32 } };

fn depth(_: c.Shape) usize {
    return 12;
}

pub const Codec = struct {
    rig: rig_.Rig,
    k: *cuda.kv4.Attention,
    x: []u16 = &.{},
    shape: c.Shape = .{ .rows = 0, .n = 0, .k = 0 },
    input: u64 = 0,
    codes: u64 = 0,
    out: u64 = 0,
    bypass: bool = false,

    pub fn deinit(self: *Codec) void {
        self.rig.gpu.gpa.free(self.x);
        self.rig.deinit();
    }

    const vt = g.hooks(Codec);

    pub fn kernel(self: *Codec) g.Kernel {
        return .{ .spec = .{
            .name = "cuda kv4 H32 codec",
            .accum = .f32,
            .depth = depth,
            .ops = 8,
            .out = .bf16,
            .tiles = .{ .{ .size = 4 }, .{}, .{} },
            .served = &shapes,
            .sums = .{ .f16 = 0, .bf16 = 0, .exact = false, .why = "midpoint bins are discontinuous; direct FP32 codec equality supplements this smooth error bound" },
        }, .ctx = self, .vt = &vt, .control = if (self.bypass) .shapes else null };
    }

    pub fn prepare(self: *Codec, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        self.rig.reset();
        self.rig.gpu.gpa.free(self.x);
        self.shape = shape;
        self.x = try self.rig.gpu.gpa.alloc(u16, shape.rows * shape.n);
        var rng: std.Random.DefaultPrng = .init(seed);
        for (self.x) |*x| x.* = c.toBf16((rng.random().float(f32) - 0.5) * 4);
        if (fill == .nonfinite) {
            const values = [_]f32{ std.math.nan(f32), std.math.inf(f32), -std.math.inf(f32) };
            for (values, 0..) |v, i| self.x[i * 32] = c.toBf16(v);
        }
        self.input = try self.rig.input(std.mem.sliceAsBytes(self.x));
        self.codes = try self.rig.scratch(shape.rows * cuda.kv4.Mode.q4.rowBytes(shape.n));
        self.out = try self.rig.output(self.x.len * 2);
        return shape.n == 128 or shape.n == 256;
    }

    pub fn run(self: *Codec) !void {
        if (self.bypass) {
            try self.rig.gpu.d.check(self.rig.gpu.d.api.cuMemcpyDtoDAsync_v2(self.out, self.input, self.x.len * 2, self.rig.s.handle), "codec control");
        } else {
            try self.k.pack(self.rig.s, self.input, self.codes, self.shape.rows, self.shape.n);
            try self.k.unpack(self.rig.s, self.codes, self.out, self.shape.rows, self.shape.n);
        }
        try self.rig.s.synchronize();
    }

    pub fn count(self: *Codec) usize {
        return self.x.len;
    }

    fn expected(self: *Codec, bits: u8, want: []f64, mag: []f64) void {
        for (0..self.shape.rows) |r| {
            var input: [256]f32 = undefined;
            var output: [256]f32 = undefined;
            var codes: [144]u8 = undefined;
            const dim = self.shape.n;
            for (0..dim) |i| input[i] = c.fromBf16(self.x[r * dim + i]);
            cuda.kv4.reference.packBits(input[0..dim], codes[0..cuda.kv4.Mode.q4.rowBytes(dim)], bits);
            cuda.kv4.reference.unpack(&codes, output[0..dim]);
            for (0..dim) |i| {
                want[r * dim + i] = output[i];
                mag[r * dim + i] = @abs(output[i]);
            }
        }
    }

    pub fn reference(self: *Codec, want: []f64, mag: []f64) !void {
        self.expected(32, want, mag);
    }

    pub fn simulate(self: *Codec, acc: c.Accum, _: g.Order, out: []f64) !void {
        const mag = try self.rig.gpu.gpa.alloc(f64, out.len);
        defer self.rig.gpu.gpa.free(mag);
        self.expected(switch (acc) {
            .f32 => 32,
            .f16 => 16,
            .bf16 => 8,
            .none => unreachable,
        }, out, mag);
    }
};

pub const Set = struct {
    k: cuda.kv4.Attention = undefined,
    codecs: [2]Codec = undefined,
    attention: [2]@import("contract_kv4_attention.zig").Attention = undefined,
    attention_used: usize = 0,
    used: usize = 0,
    loaded: bool = false,

    pub fn add(self: *Set, gpu: check.Gpu, s: cuda.Stream, kernels: *std.ArrayList(g.Kernel)) !void {
        if (try gpu.ctx.capability() != 86) return;
        self.k = try cuda.kv4.Attention.load(gpu.d);
        self.loaded = true;
        for (&self.codecs, 0..) |*codec, i| {
            codec.* = .{ .rig = .{ .gpu = gpu, .s = s }, .k = &self.k, .bypass = i == 1 };
            self.used += 1;
            try kernels.append(gpu.gpa, codec.kernel());
        }

        for (&self.attention, 0..) |*attention, i| {
            attention.* = .{ .rig = .{ .gpu = gpu, .s = s }, .k = &self.k, .bypass = i == 1 };
            self.attention_used += 1;
            try kernels.append(gpu.gpa, attention.kernel());
        }
    }

    pub fn deinit(self: *Set) void {
        for (self.codecs[0..self.used]) |*codec| codec.deinit();
        for (self.attention[0..self.attention_used]) |*attention| attention.deinit();
        if (self.loaded) self.k.unload();
    }
};

pub fn run(gpu: check.Gpu) !void {
    var s = try cuda.Stream.init(gpu.d, false);
    defer s.deinit();
    var set: Set = .{};
    defer set.deinit();
    var kernels: std.ArrayList(g.Kernel) = .empty;
    defer kernels.deinit(gpu.gpa);
    try set.add(gpu, s, &kernels);
    var tally: g.Tally = .{};
    try g.runAll(gpu.gpa, kernels.items, 0x799, &tally);
    const records = core.contract.registry.kv4;
    var problems = try core.contract.registry.checkCuda(gpu.gpa, &records, false);
    const images = try core.contract.cubin.images(gpu.gpa, cuda.kernels.kv4);
    defer core.contract.cubin.freeImages(gpu.gpa, images);
    for (images) |img| {
        if (img.sm != 86) continue;
        const compiled = try core.contract.cubin.kernels(gpu.gpa, img.elf, img.sm);
        defer gpu.gpa.free(compiled);
        for (records) |r| {
            const symbol = try gpu.gpa.dupeSentinel(u8, r.launch.symbol, 0);
            defer gpu.gpa.free(symbol);
            const f = try set.k.mod.function(symbol);
            const usage = for (compiled) |entry| {
                if (std.mem.eql(u8, entry.name, r.launch.symbol)) break entry.usage;
            } else return error.KvMissingSymbol;
            if (usage.regs != r.usage[1].regs or usage.static != r.usage[1].static or usage.bound != r.usage[1].bound or try f.attribute(.num_regs) != usage.regs or try f.attribute(.max_threads_per_block) < r.launch.threads) problems += 1;
        }
    }
    std.debug.print("LIMITS kv4 launches=6 sm=86 problems={d}\n", .{problems});
    if (problems != 0) return error.KvResourceDrift;
    std.debug.print("SUMMARY kv4 codec kernels={d} pass={d} fail={d} controls={d} red={d} missed={d}\n", .{ tally.kernels, tally.passed, tally.failed, tally.controls, tally.caught, tally.missed });
    if (tally.problems() != 0) return error.TestFailed;
}
