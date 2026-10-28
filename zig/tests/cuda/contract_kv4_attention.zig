const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const check = @import("check.zig");
const rig_ = @import("contract_rig.zig");
const c = core.contract.spec;
const g = core.contract.gates;
const shapes = [_]c.Shape{ .{ .rows = 7, .n = 128, .k = 513 }, .{ .rows = 6, .n = 256, .k = 511 } };

fn depth(s: c.Shape) usize {
    return 2 * s.k + s.n;
}

fn round(x: f64, acc: c.Accum) f64 {
    return switch (acc) {
        .f32 => @as(f32, @floatCast(x)),
        .f16 => @as(f16, @floatCast(x)),
        .bf16 => c.fromBf16(c.toBf16(@floatCast(x))),
        .none => x,
    };
}

pub const Attention = struct {
    rig: rig_.Rig,
    k: *cuda.kv4.Attention,
    x: []f32 = &.{},
    shape: c.Shape = .{ .rows = 0, .n = 0, .k = 0 },
    q: u64 = 0,
    key: u64 = 0,
    val: u64 = 0,
    scratch: u64 = 0,
    out: u64 = 0,
    bypass: bool = false,

    pub fn deinit(self: *Attention) void {
        self.rig.gpu.gpa.free(self.x);
        self.rig.deinit();
    }

    const vt = g.hooks(Attention);

    pub fn kernel(self: *Attention) g.Kernel {
        return .{ .spec = .{
            .name = "cuda kv4 attention and merge",
            .accum = .f32,
            .depth = depth,
            .ops = 16,
            .out = .bf16,
            .tiles = .{ .{}, .{}, .{ .size = 512 } },
            .served = &shapes,
            .sums = .{ .f16 = 0, .bf16 = 0, .exact = false, .why = "softmax saturation does not guarantee reduced-precision separation on this bounded fixture" },
        }, .ctx = self, .vt = &vt, .control = if (self.bypass) .shapes else null };
    }

    pub fn prepare(self: *Attention, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        self.rig.reset();
        const a = self.rig.gpu.gpa;
        a.free(self.x);
        self.shape = shape;
        const n = shape.n * (shape.rows + 2 * shape.k);
        self.x = try a.alloc(f32, n);
        const raw = try a.alloc(u16, n);
        defer a.free(raw);
        var rng: std.Random.DefaultPrng = .init(seed);
        for (raw, self.x) |*v, *x| {
            v.* = c.toBf16((rng.random().float(f32) - 0.5) * 2);
            x.* = c.fromBf16(v.*);
        }
        const qlen = shape.rows * shape.n;
        if (fill == .nonfinite) {
            raw[0] = c.toBf16(std.math.nan(f32));
            self.x[0] = std.math.nan(f32);
        }
        const input = try self.rig.input(std.mem.sliceAsBytes(raw));
        self.q = input;
        const bytes = shape.k * cuda.kv4.Mode.q4.rowBytes(shape.n);
        self.key = try self.rig.scratch(bytes);
        self.val = try self.rig.scratch(bytes);
        try self.k.pack(self.rig.s, input + qlen * 2, self.key, shape.k, shape.n);
        try self.k.pack(self.rig.s, input + (qlen + shape.k * shape.n) * 2, self.val, shape.k, shape.n);
        for (0..2 * shape.k) |row| {
            const values = self.x[qlen + row * shape.n ..][0..shape.n];
            var codes: [144]u8 = undefined;
            cuda.kv4.reference.pack(values, &codes);
            cuda.kv4.reference.unpack(&codes, values);
        }
        self.scratch = try self.rig.scratch(cuda.kv4.Attention.scratchBytes(shape.rows, shape.n, shape.k));
        self.out = try self.rig.output(qlen * 2);
        return shape.n == 128 or shape.n == 256;
    }

    pub fn run(self: *Attention) !void {
        if (self.bypass) {
            try self.rig.zero(self.out, self.shape.rows * self.shape.n * 2);
        } else {
            try self.k.run(self.rig.s, .{ .q = self.q, .k = self.key, .v = self.val }, self.scratch, self.out, self.shape.rows, 1, self.shape.n, self.shape.k);
        }
        try self.rig.s.synchronize();
    }

    pub fn count(self: *Attention) usize {
        return self.shape.rows * self.shape.n;
    }

    fn expected(self: *Attention, acc: c.Accum, order: g.Order, want: []f64, mag: []f64) !void {
        const dim = self.shape.n;
        const keys = self.shape.k;
        const qlen = self.shape.rows * dim;
        const scores = try self.rig.gpu.gpa.alloc(f64, keys);
        defer self.rig.gpu.gpa.free(scores);
        for (0..self.shape.rows) |head| {
            var maximum = -std.math.inf(f64);
            for (scores, 0..) |*score, t| {
                var products: [256]f64 = undefined;
                for (0..dim) |i| products[i] = @as(f64, self.x[head * dim + i]) * self.x[qlen + t * dim + i];
                var sum: f64 = 0;
                if (order == .pairwise) {
                    var width = dim;
                    while (width > 1) : (width /= 2) {
                        for (0..width / 2) |i| products[i] = round(products[2 * i] + products[2 * i + 1], acc);
                    }
                    sum = products[0];
                } else {
                    for (0..dim) |i| {
                        const d = if (order == .strided) (i % 4) * (dim / 4) + i / 4 else i;
                        sum = round(sum + products[d], acc);
                    }
                }
                score.* = sum / @sqrt(@as(f64, @floatFromInt(dim)));
                maximum = @max(maximum, score.*);
            }
            const out = want[head * dim ..][0..dim];
            const magnitude = mag[head * dim ..][0..dim];
            @memset(out, 0);
            @memset(magnitude, 0);
            var sum: f64 = 0;
            for (scores, 0..) |score, t| {
                const p = @exp(score - maximum);
                sum += p;
                for (out, magnitude, 0..) |*w, *m, d| {
                    const v = self.x[qlen + (keys + t) * dim + d];
                    w.* += p * v;
                    m.* += p * @abs(v);
                }
            }
            for (out, magnitude) |*w, *m| {
                w.* /= sum;
                m.* /= sum;
            }
        }
    }

    pub fn reference(self: *Attention, want: []f64, mag: []f64) !void {
        try self.expected(.none, .sequential, want, mag);
    }

    pub fn simulate(self: *Attention, acc: c.Accum, order: g.Order, out: []f64) !void {
        const mag = try self.rig.gpu.gpa.alloc(f64, out.len);
        defer self.rig.gpu.gpa.free(mag);
        try self.expected(acc, order, out, mag);
    }
};
