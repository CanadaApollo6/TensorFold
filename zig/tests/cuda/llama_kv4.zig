const std = @import("std");
const cuda = @import("cuda");
const check = @import("check.zig");
const llama = @import("llama");
const common = @import("llama_tests.zig");

fn bf(x: f32) u16 {
    const bits: u32 = @bitCast(x);
    return @truncate((bits + 0x7fff + ((bits >> 16) & 1)) >> 16);
}

fn float(x: u16) f32 {
    return @bitCast(@as(u32, x) << 16);
}

fn read(gpu: check.Gpu, s: cuda.Stream, ptr: u64, out: []u8) !void {
    try s.synchronize();
    try gpu.d.check(gpu.d.api.cuMemcpyDtoH_v2(out.ptr, ptr, out.len), "KV4 read");
}

pub fn synthetic(gpu: check.Gpu) !void {
    var k = try cuda.kv4.Attention.load(gpu.d);
    defer k.unload();
    var tree = try cuda.tree_attention.TreeAttention.load(gpu.d);
    defer tree.unload();
    var s = try cuda.Stream.init(gpu.d, false);
    defer s.deinit();
    for ([_]usize{ 128, 256 }) |dim| {
        const rows = 9;
        const values = rows * dim;
        const host = try gpu.gpa.alloc(u16, values);
        defer gpu.gpa.free(host);
        for (host, 0..) |*v, i| v.* = bf(@sin(@as(f32, @floatFromInt(i)) * 0.731) * (if (i < dim) @as(f32, 0) else 2.5));
        var x = try cuda.DeviceBuffer.fromHost(gpu.d, std.mem.sliceAsBytes(host));
        defer x.free();
        const rb = cuda.kv4.Mode.q4.rowBytes(dim);
        var codes = try cuda.DeviceBuffer.alloc(gpu.d, rows * rb + 128);
        defer codes.free();
        try gpu.d.check(gpu.d.api.cuMemsetD8_v2(codes.ptr, 0xa5, codes.len), "KV4 guard fill");
        var decoded = try cuda.DeviceBuffer.alloc(gpu.d, values * 2);
        defer decoded.free();
        try k.pack(s, x.ptr, codes.ptr + 64, rows, dim);
        try k.unpack(s, codes.ptr + 64, decoded.ptr, rows, dim);
        const encoded = try gpu.gpa.alloc(u8, codes.len);
        defer gpu.gpa.free(encoded);
        try read(gpu, s, codes.ptr, encoded);
        for (encoded[0..64]) |v| if (v != 0xa5) return error.KvGuard;
        for (encoded[64 + rows * rb ..]) |v| if (v != 0xa5) return error.KvGuard;
        const actual = try gpu.gpa.alloc(u16, values);
        defer gpu.gpa.free(actual);
        try read(gpu, s, decoded.ptr, std.mem.sliceAsBytes(actual));
        var max_error: f32 = 0;
        for (0..rows) |r| {
            var input: [256]f32 = undefined;
            var output: [256]f32 = undefined;
            var expected: [144]u8 = undefined;
            for (0..dim) |i| input[i] = float(host[r * dim + i]);
            cuda.kv4.reference.pack(input[0..dim], expected[0..rb]);
            if (!std.mem.eql(u8, expected[0..rb], encoded[64 + r * rb ..][0..rb])) return error.KvQuantReference;
            cuda.kv4.reference.unpack(expected[0..rb], output[0..dim]);
            for (0..dim) |i| {
                max_error = @max(max_error, @abs(output[i] - float(actual[r * dim + i])));
                if (bf(output[i]) != actual[r * dim + i]) return error.KvDequantReference;
            }
        }
        std.debug.print("QUANT dim={d} rows={d} encoded=byte_equal dequant=bf16_equal fp32_max_rounding={e} guards=pass\n", .{ dim, rows, max_error });
    }
    for ([_]usize{ 1, 2, 127, 128, 511, 512, 513, 2049, 20000, 115001 }) |keys| try attention(gpu, &k, &tree, s, keys);
    inline for (.{ "pack_fn", "unpack_fn", "merge_fn" }) |field| {
        const f = @field(k, field);
        std.debug.print("RECORD kv4 {s} regs={d} shared={d} max_threads={d}\n", .{ field, try f.attribute(.num_regs), try f.attribute(.shared_size_bytes), try f.attribute(.max_threads_per_block) });
    }
    for (k.partial, 0..) |f, i| std.debug.print("RECORD kv4 partial{d} regs={d} shared={d} max_threads={d}\n", .{ i, try f.attribute(.num_regs), try f.attribute(.shared_size_bytes), try f.attribute(.max_threads_per_block) });
    std.debug.print("SUMMARY llama kv4 synthetic pass peak_bytes={d}\n", .{cuda.usage(false).peak});
}

fn attention(gpu: check.Gpu, k: *cuda.kv4.Attention, tree: *cuda.tree_attention.TreeAttention, s: cuda.Stream, keys: usize) !void {
    const h = 56;
    const hk = 8;
    const dim = 128;
    const n = keys * hk * dim;
    const host = try gpu.gpa.alloc(u16, n * 2 + h * dim);
    defer gpu.gpa.free(host);
    var rng: std.Random.DefaultPrng = .init(799 + keys);
    for (host) |*v| v.* = bf((rng.random().float(f32) - 0.5) * 2);
    var input = try cuda.DeviceBuffer.fromHost(gpu.d, std.mem.sliceAsBytes(host));
    defer input.free();
    const kc = input.ptr;
    const vc = kc + n * 2;
    const query = vc + n * 2;
    const bytes = keys * hk * cuda.kv4.Mode.q4.rowBytes(dim);
    var encoded = try cuda.DeviceBuffer.alloc(gpu.d, bytes * 2);
    defer encoded.free();
    try k.pack(s, kc, encoded.ptr, keys * hk, dim);
    try k.pack(s, vc, encoded.ptr + bytes, keys * hk, dim);
    const groups = cuda.tree_attention.TreeAttention.groups(keys);
    const scratch_size = @max(cuda.kv4.Attention.scratchBytes(h, dim, keys), cuda.tree_attention.TreeAttention.scratchBytes(1, h, hk, groups, dim));
    var scratch = try cuda.DeviceBuffer.alloc(gpu.d, scratch_size);
    defer scratch.free();
    var out = try cuda.DeviceBuffer.alloc(gpu.d, h * dim * 4);
    defer out.free();
    const offset: [2]i64 = .{ 0, @intCast(n) };
    var offsets = try cuda.DeviceBuffer.fromHost(gpu.d, std.mem.asBytes(&offset));
    defer offsets.free();
    const meta: [8]i32 = .{ @intCast(keys - 1), 0, 0, 1, @intCast(keys - 1), 0, 0, 1 };
    var metadata = try cuda.DeviceBuffer.fromHost(gpu.d, std.mem.asBytes(&meta));
    defer metadata.free();
    const rows: cuda.tree_attention.Rows = .{ .row_stream = metadata.ptr + 4, .streams = metadata.ptr + 8, .offs = offsets.ptr, .origin = kc, .paths = metadata.ptr + 24, .depths = metadata.ptr + 28 };
    const last_k = kc + (keys - 1) * hk * dim * 2;
    const last_v = vc + (keys - 1) * hk * dim * 2;
    try tree.run(s, query, last_k, last_v, rows, cuda.tree_attention.TreeAttention.scratch(scratch.ptr, 1, h, hk, groups, dim), out.ptr, 1, h, hk, dim, groups, 1.0 / @sqrt(@as(f32, dim)));
    try k.run(s, .{ .q = query, .k = encoded.ptr, .v = encoded.ptr + bytes, .first_k = kc, .first_v = vc, .last_k = last_k, .last_v = last_v }, scratch.ptr, out.ptr + h * dim * 2, h, hk, dim, keys);
    var actual: [h * dim * 2]u16 = undefined;
    try read(gpu, s, out.ptr, std.mem.asBytes(&actual));
    var ref: [h * dim]f32 = undefined;
    var got: [h * dim]f32 = undefined;
    for (&ref, &got, 0..) |*a, *b, i| {
        a.* = float(actual[i]);
        b.* = float(actual[h * dim + i]);
    }
    if (keys <= 2049) try cpuAttention(gpu, host, &got, keys);
    if (keys == 513) {
        const expected = actual;
        for (0..5) |_| {
            try k.run(s, .{ .q = query, .k = encoded.ptr, .v = encoded.ptr + bytes, .first_k = kc, .first_v = vc, .last_k = last_k, .last_v = last_v }, scratch.ptr, out.ptr + h * dim * 2, h, hk, dim, keys);
            try read(gpu, s, out.ptr, std.mem.asBytes(&actual));
            if (!std.mem.eql(u16, &expected, &actual)) return error.KvNondeterministic;
        }
    }
    const met = try llama.validation.layer(&ref, &got);
    if (!std.math.isFinite(met.rmse) or met.rmse > 0.05 or met.max_abs > 0.2) return error.KvAttentionDrift;
    std.debug.print("ATTENTION heads=56/8 dim=128 keys={d} rmse={e} max_abs={e} relative_l2={e}\n", .{ keys, met.rmse, met.max_abs, met.relative_l2 });
}

fn cpuAttention(gpu: check.Gpu, input: []const u16, output: []const f32, keys: usize) !void {
    const dim = 128;
    const width = 8 * dim;
    const n = keys * width;
    const scores = try gpu.gpa.alloc(f64, keys);
    defer gpu.gpa.free(scores);
    const values = try gpu.gpa.alloc([dim]f32, keys);
    defer gpu.gpa.free(values);
    var maximum = -std.math.inf(f64);
    for (scores, values, 0..) |*score, *v, t| {
        var key: [dim]f32 = undefined;
        for (&key, v, 0..) |*a, *b, d| {
            a.* = float(input[t * width + d]);
            b.* = float(input[n + t * width + d]);
        }
        if (t != 0 and t + 1 != keys) {
            var codes: [72]u8 = undefined;
            cuda.kv4.reference.pack(&key, &codes);
            cuda.kv4.reference.unpack(&codes, &key);
            cuda.kv4.reference.pack(v, &codes);
            cuda.kv4.reference.unpack(&codes, v);
        }
        score.* = 0;
        for (key, 0..) |x, d| score.* += @as(f64, x) * float(input[2 * n + d]);
        score.* /= @sqrt(@as(f64, dim));
        maximum = @max(maximum, score.*);
    }
    var sum: f64 = 0;
    var want: [dim]f64 = @splat(0);
    for (scores, values) |score, v| {
        const p = @exp(score - maximum);
        sum += p;
        for (&want, v) |*a, x| a.* += p * x;
    }
    var max_error: f64 = 0;
    for (want, output[0..dim]) |v, got| max_error = @max(max_error, @abs(v / sum - got));
    if (max_error > 0.004) return error.KvCpuAttention;
    std.debug.print("REFERENCE keys={d} head=0 fp64_max_abs={e}\n", .{ keys, max_error });
}

pub fn real(gpu: check.Gpu, model: []const u8, refs: []const u8, count: usize) !void {
    const seqs = try common.sequences(gpu, refs);
    defer seqs.deinit();
    var r: common.Rig = undefined;
    try r.initCache(gpu, model, count, 2048, 52);
    defer r.deinit();
    var q4 = try llama.forward.Model.initMode(gpu.gpa, gpu.d, &r.w, &r.k, r.s, 2048, .q4);
    defer q4.deinit();
    const c = r.w.config;
    var worst: f64 = 0;
    const ref = try gpu.gpa.alloc(f32, c.qDim());
    defer gpu.gpa.free(ref);
    const got = try gpu.gpa.alloc(f32, c.qDim());
    defer gpu.gpa.free(got);
    var positions: usize = 0;
    for (seqs.value) |seq| {
        for (seq[0..@min(seq.len, 2048)], 0..) |token, pos| {
            try r.m.begin(token, pos);
            try q4.begin(token, pos);
            for (0..count) |li| {
                const slot = 52 - count + li;
                try r.m.layer(slot);
                try q4.layer(slot);
                var raw: [7168]u16 = undefined;
                try r.m.download(r.m.attn, std.mem.asBytes(&raw));
                for (ref, raw) |*v, bits| v.* = float(bits);
                try q4.download(q4.attn, std.mem.asBytes(&raw));
                for (got, raw) |*v, bits| v.* = float(bits);
                const met = try llama.validation.layer(ref, got);
                if (!std.math.isFinite(met.relative_l2)) return error.KvAttentionNonfinite;
                worst = @max(worst, met.relative_l2);
                if (pos < 2 or pos == 511 or pos == 512 or pos + 1 == @min(seq.len, 2048)) std.debug.print("REAL layer={d} cache_slot={d} position={d} attention_rmse={e} max_abs={e} relative_l2={e}\n", .{ li, slot, pos, met.rmse, met.max_abs, met.relative_l2 });
            }
            positions += 1;
        }
    }
    std.debug.print("SUMMARY llama kv4 real layers={d} cache_layers=52 tokens={d} worst_attention_relative_l2={e} peak_bytes={d}\n", .{ count, positions, worst, cuda.usage(false).peak });
}
