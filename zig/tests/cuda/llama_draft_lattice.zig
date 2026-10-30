const std = @import("std");
const cuda = @import("cuda");
const l = @import("llama").draft.lattice;
const check = @import("check.zig");
fn launch(s: cuda.Stream, f: cuda.Function, grid: cuda.Dim3, values: anytype) !void {
    var args: cuda.Args = .{};
    inline for (values) |v| args.add(v);
    try cuda.launch.launch(f, .{ .grid = grid, .block = .{ .x = 256 } }, s, &args);
}
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
        const rows = 7;
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
                else => if (id % 2 == 0) @as(f32, @bitCast(@as(u32, 0x7f7f0000))) else -3.3895314e38,
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
            for (got, 0..) |row, r| {
                const want = try l.reference(raw[r * vocab ..][0..vocab], 16);
                if (!std.mem.eql(u8, std.mem.asBytes(&want.ids), std.mem.asBytes(&row.ids)) or
                    !std.mem.eql(u8, std.mem.asBytes(&want.values), std.mem.asBytes(&row.values)))
                    return error.DraftLatticeCandidates;
                if (@abs(want.log_mass - row.log_mass) > l.tolerance(vocab)) return error.DraftLatticeLse;
                const certified = try l.normalizer(row, vocab);
                fallbacks += @intFromBool(certified == null);
                const z = certified orelse l.exactNormalizer(want);
                if (@as(u32, @bitCast(z)) != @as(u32, @bitCast(l.exactNormalizer(want))))
                    return error.DraftLatticeNormalizer;
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
