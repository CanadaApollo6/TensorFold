//! Direct old/new equality at query, causal, segment and GQA boundaries, with guarded buffers.
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const check = @import("check.zig");
const Rig = @import("contract_rig.zig").Rig;
const Attention = cuda.kv4.Attention;
const bf = core.contract.spec.toBf16;

pub fn run(gpu: check.Gpu) !void {
    var k = try Attention.load(gpu.d);
    defer k.unload();
    var s = try cuda.Stream.init(gpu.d, false);
    defer s.deinit();
    var cases: usize = 0;
    for (1..34) |rows| {
        try one(gpu, s, &k, 56, 8, 0, rows);
        try one(gpu, s, &k, 14, 2, 2039, rows);
        cases += 2;
    }
    for ([_]usize{ 1, 2, 7, 8 }) |gqa| {
        for ([_]usize{ 1, 2, 63, 64, 65, 127, 128 }) |rows| {
            for ([_]usize{ 1, 2, 63, 64, 65, 2047, 2048, 2049, 4095 }) |position| {
                try one(gpu, s, &k, 2 * gqa, 2, position, rows);
                cases += 1;
            }
        }
    }
    try one(gpu, s, &k, 56, 8, 60000 - 128, 128);
    cases += 1;
    std.debug.print("SUMMARY kv4-prompt-unit cases={d} repeats=5 output_bits=PASS partial_bits=PASS guards=PASS\n", .{cases});
}

fn one(gpu: check.Gpu, s: cuda.Stream, k: *const Attention, heads: usize, kvheads: usize, position: usize, rows: usize) !void {
    var r: Rig = .{ .gpu = gpu, .s = s };
    defer r.deinit();
    var arena = std.heap.ArenaAllocator.init(gpu.gpa);
    defer arena.deinit();
    const a = arena.allocator();
    const dim = 128;
    const keys = position + rows + 65;
    const kv_len = keys * kvheads * dim;
    const q_len = rows * heads * dim;
    const host = try a.alloc(u16, 2 * kv_len + q_len);
    var rng: std.Random.DefaultPrng = .init(0x4ab + position + rows * heads);
    for (host) |*x| x.* = bf((rng.random().float(f32) - 0.5) * 2);
    const q = host[2 * kv_len ..];
    @memset(q[0 .. heads * dim], bf(0));
    if (rows > 1) for (q[heads * dim .. 2 * heads * dim], 0..) |*x, i| {
        x.* = bf(if (i % 32 == 0) 1.0e8 else 0.03125);
    };
    const input = try r.input(std.mem.sliceAsBytes(host));
    const vbf = input + kv_len * 2;
    const query = vbf + kv_len * 2;
    const code_len = keys * kvheads * cuda.kv4.Mode.q4.rowBytes(dim);
    const codes = try r.scratch(2 * code_len);
    try k.pack(s, input, codes, keys * kvheads, dim);
    try k.pack(s, vbf, codes + code_len, keys * kvheads, dim);
    const poisoned = kvheads * cuda.kv4.Mode.q4.rowBytes(dim);
    for ([_]u64{ codes, codes + code_len }) |ptr| {
        const bos: cuda.DeviceBuffer = .{ .d = gpu.d, .ptr = ptr, .len = poisoned };
        try bos.fill8(0xff, s.handle);
        const future: cuda.DeviceBuffer = .{ .d = gpu.d, .ptr = ptr + (position + rows) * poisoned, .len = 65 * poisoned };
        try future.fill8(0xff, s.handle);
    }
    const segs = if (position + rows - 1 >= 2) (position + rows - 2) / Attention.segment + 1 else 1;
    const partial_len = rows * heads * segs * (dim + 2) * 4;
    const work_len = Attention.workBytes(heads, dim, keys);
    const work = [_]u64{ try r.scratch(work_len), try r.scratch(work_len) };
    const out_len = q_len * 2;
    const out = [_]u64{ try r.output(out_len), try r.output(out_len) };
    const inputs = try a.dupe(u8, try r.inputBytes());
    const encoded = try a.dupe(u8, try r.bytes(codes, 2 * code_len));
    const id = try std.fmt.allocPrint(a, "kv4-prompt-unit h={d} hk={d} pos={d} rows={d}", .{ heads, kvheads, position, rows });
    const args: Attention.Rows = .{ .q = query, .k = codes, .v = codes + code_len, .first_k = input, .first_v = vbf, .last_k = input + position * kvheads * dim * 2, .last_v = vbf + position * kvheads * dim * 2 };
    var expected: ?[]const u8 = null;
    for (0..5) |repeat| {
        for ([_]cuda.kv4.PromptTiles{ .old, .new }, work, out) |mode, scratch, output| {
            const wb: cuda.DeviceBuffer = .{ .d = gpu.d, .ptr = scratch, .len = work_len };
            const ob: cuda.DeviceBuffer = .{ .d = gpu.d, .ptr = output, .len = out_len };
            try wb.fill8(0x6d, s.handle);
            try ob.fill8(if (mode == .old) 0xff else 0x7f, s.handle);
            try k.promptWith(s, args, scratch, output, heads, kvheads, dim, position, rows, mode);
        }
        const old = try a.dupe(u8, try r.bytes(out[0], out_len));
        try check.sameBytes(id, try r.bytes(out[1], out_len), old);
        if (expected) |e| try check.sameBytes(id, old, e) else expected = old;
        for (0..q_len) |i| {
            const value: f32 = @bitCast(@as(u32, std.mem.bytesToValue(u16, old[2 * i ..][0..2])) << 16);
            try check.expect(std.math.isFinite(value), "{s} nonfinite at {d}, repeat {d}", .{ id, i, repeat });
        }
        const partials = try a.dupe(u8, try r.bytes(work[0], partial_len));
        try check.sameBytes(id, try r.bytes(work[1], partial_len), partials);
        try check.sameBytes(id, try r.inputBytes(), inputs);
        try check.sameBytes(id, try r.bytes(codes, 2 * code_len), encoded);
        try check.expect(try r.intact(), "{s} damaged guard", .{id});
    }
}
