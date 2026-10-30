const std = @import("std");
const cuda = @import("cuda");
const l = @import("llama").draft.lattice;
const check = @import("check.zig");
fn launch(s: cuda.Stream, f: cuda.Function, grid: cuda.Dim3, values: anytype) !void {
    var args: cuda.Args = .{};
    inline for (values) |v| args.add(v);
    try cuda.launch.launch(f, .{ .grid = grid, .block = .{ .x = 256 } }, s, &args);
}
const Device = struct {
    input: cuda.DeviceBuffer,
    downloads: usize = 0,
    pub fn enter(_: *Device) !void {}
    pub fn leave(_: *Device, _: void) void {}
    pub fn download(d: *Device, address: u64, bytes: []u8) !void {
        if (address < d.input.ptr) return error.DraftLatticeFallbackRange;
        try d.input.download(address - d.input.ptr, bytes);
        d.downloads += 1;
    }
};
pub fn run(gpu: check.Gpu) !void {
    const a = gpu.gpa;
    var m = try cuda.Module.load(gpu.d, cuda.kernels.sample);
    defer m.unload();
    const top = try m.function("tf_topk_slices");
    const finish = try m.function("tf_draft_lattice");
    var s = try cuda.Stream.init(gpu.d, false);
    defer s.deinit();
    var checked: usize = 0;
    var fallbacks: usize = 0;
    for ([_]u32{ 17, 4095, 4096, 4097, 166912, 248320 }) |vocab| {
        const rows = 8;
        const n = l.slices(vocab);
        const raw = try a.alloc(u16, rows * vocab);
        defer a.free(raw);
        for (0..rows) |row| for (raw[row * vocab ..][0..vocab], 0..) |*b, id| {
            const v: f32 = switch (row) {
                0 => @as(f32, @floatFromInt((id * 7919) % 997)) / 32 - 15,
                1 => 1,
                2 => if (id < 15) 4 else if (id == 15) 1.0078125 else 1,
                3 => if (id % 2 == 0) 65536 else -65536,
                4 => if (id % 2 == 0) -0.0 else 0.0,
                5 => if (id == vocab - 1) 100 else -100,
                6 => if (id % 2 == 0) @as(f32, @bitCast(@as(u32, 0x7f7f0000))) else -3.3895314e38,
                // Too close to an FP32 rounding boundary to certify: this row takes the full-logit fallback.
                else => if (id == 0) 0 else -32,
            };
            b.* = @intCast(@as(u32, @bitCast(v)) >> 16);
        };
        var input = try cuda.DeviceBuffer.alloc(gpu.d, raw.len * 2);
        defer input.free();
        var pv = try cuda.DeviceBuffer.alloc(gpu.d, rows * n * 16 * 4);
        defer pv.free();
        var pi = try cuda.DeviceBuffer.alloc(gpu.d, pv.len);
        defer pi.free();
        var output = try cuda.DeviceBuffer.alloc(gpu.d, rows * @sizeOf(l.Row));
        defer output.free();
        try input.upload(0, std.mem.sliceAsBytes(raw));
        var want: [rows]l.Row = undefined;
        for (&want, 0..) |*w, r| w.* = try l.reference(raw[r * vocab ..][0..vocab], 16);
        var previous: [rows]l.Row = undefined;
        for (0..5) |repeat| {
            try launch(s, top, .{ .x = n, .y = rows }, .{ input.ptr, vocab, @as(u32, 16), n, pv.ptr, pi.ptr });
            try launch(s, finish, .{ .x = rows }, .{ input.ptr, vocab, @as(u32, 16), pv.ptr, pi.ptr, n * 16, output.ptr });
            try s.synchronize();
            var got: [rows]l.Row = undefined;
            try output.download(0, std.mem.sliceAsBytes(&got));
            if (repeat > 0 and !std.mem.eql(u8, std.mem.asBytes(&previous), std.mem.asBytes(&got)))
                return error.DraftLatticeRepeat;
            previous = got;
            for (got, want, 0..) |row, w, r| {
                if (!std.mem.eql(u8, std.mem.asBytes(&w.ids), std.mem.asBytes(&row.ids)) or
                    !std.mem.eql(u8, std.mem.asBytes(&w.values), std.mem.asBytes(&row.values)))
                    return error.DraftLatticeCandidates;
                if (!std.math.isFinite(row.log_mass) or @abs(w.log_mass - row.log_mass) > l.tolerance(vocab)) return error.DraftLatticeLse;
                if (((try l.normalizer(row, vocab)) != null) != (r != rows - 1)) return error.DraftLatticeCertified;
            }
            var device: Device = .{ .input = input };
            var ids: [rows * 16]u32 = undefined;
            var unary: [rows * 16]f32 = undefined;
            const fell = try l.unary(a, &got, input.ptr, vocab, vocab, 16, &device, &ids, &unary);
            if (fell != 1 or device.downloads != 1) return error.DraftLatticeFallback;
            fallbacks += fell;
            for (want, 0..) |w, r| {
                if (!std.mem.eql(u32, &w.ids, ids[r * 16 ..][0..16])) return error.DraftLatticeCandidates;
                for (w.values, unary[r * 16 ..][0..16]) |v, got_unary| {
                    if (@as(u32, @bitCast(v - l.exactNormalizer(w))) != @as(u32, @bitCast(got_unary))) return error.DraftLatticeUnary;
                }
                checked += 1;
            }
        }
        const unchanged = try a.alloc(u16, raw.len);
        defer a.free(unchanged);
        try input.download(0, std.mem.sliceAsBytes(unchanged));
        if (!std.mem.eql(u16, raw, unchanged)) return error.DraftLatticeInputWrite;
        for ([_]u16{ 0x7fc0, 0x7f80, 0xff80 }) |bad| {
            raw[vocab - 1] = bad;
            try input.upload(0, std.mem.sliceAsBytes(raw));
            try launch(s, top, .{ .x = n, .y = rows }, .{ input.ptr, vocab, @as(u32, 16), n, pv.ptr, pi.ptr });
            try launch(s, finish, .{ .x = rows }, .{ input.ptr, vocab, @as(u32, 16), pv.ptr, pi.ptr, n * 16, output.ptr });
            try s.synchronize();
            var got: l.Row = undefined;
            try output.download(0, std.mem.asBytes(&got));
            if (got.invalid != 1) return error.DraftLatticeNonfinite;
        }
    }
    std.debug.print("SUMMARY dflash2 lattice rows={d} exact_candidates=PASS exact_unary=PASS fallback={d} nonfinite=PASS repeat=PASS\n", .{ checked, fallbacks });
}
