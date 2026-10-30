const std = @import("std");
const cuda = @import("cuda");
const draft = @import("llama").draft;
const check = @import("check.zig");
pub fn run(gpu: check.Gpu) !void {
    var s = try cuda.Stream.init(gpu.d, false);
    defer s.deinit();
    var qk = try cuda.qk_rows.QkRows.load(gpu.d);
    defer qk.unload();
    const bytes = 8 * 32 * 128 * 2;
    var input = try cuda.DeviceBuffer.alloc(gpu.d, bytes);
    defer input.free();
    var old = try cuda.DeviceBuffer.alloc(gpu.d, bytes);
    defer old.free();
    var new = try cuda.DeviceBuffer.alloc(gpu.d, bytes);
    defer new.free();
    var other = try cuda.DeviceBuffer.alloc(gpu.d, bytes);
    defer other.free();
    var metadata = try cuda.DeviceBuffer.alloc(gpu.d, 256 + 8 * 4);
    defer metadata.free();
    const raw = try gpu.gpa.alloc(u16, bytes / 2);
    defer gpu.gpa.free(raw);
    for (raw, 0..) |*b, i| {
        const v = @as(f32, @floatFromInt(i % 79)) / 32 - 1;
        b.* = @intCast(@as(u32, @bitCast(v)) >> 16);
    }
    try input.upload(0, std.mem.sliceAsBytes(raw));
    const expected = try gpu.gpa.alloc(u8, bytes);
    defer gpu.gpa.free(expected);
    const actual = try gpu.gpa.alloc(u8, bytes);
    defer gpu.gpa.free(actual);
    var cases: usize = 0;
    for ([_]f64{ 10000, 10000000 }) |theta| {
        var r = try draft.rotary.Rotary.init(gpu.d, s, theta);
        defer r.deinit();
        var inv: [64]f32 = undefined;
        for (&inv, 0..) |*v, i| v.* = @floatCast(1 / std.math.pow(f64, theta, @as(f64, @floatFromInt(i)) / 64));
        try metadata.upload(0, std.mem.sliceAsBytes(&inv));
        for ([_]u32{ 1, 8 }) |rows| for ([_]u32{ 8, 32 }) |heads| {
            r.begin();
            for ([_]u64{ 0, 130, 2047, std.math.maxInt(i32) - 8 }) |base| {
                var positions: [8]u64 = undefined;
                var old_pos: [8]i32 = undefined;
                for (positions[0..rows], old_pos[0..rows], 0..) |*p, *o, i| {
                    p.* = base + i;
                    o.* = @intCast(p.*);
                }
                try metadata.upload(256, std.mem.sliceAsBytes(old_pos[0..rows]));
                const width = heads * 128;
                const x: draft.operators.View = .{ .handle = input.ptr, .rows = rows, .width = width, .stride = width, .dtype = .bf16 };
                const src: cuda.qk_rows.Src = .{ .q = x.handle, .k = x.handle, .q_ld = width, .k_ld = width, .q_hs = 128, .k_hs = 128 };
                try qk.rowsFrom(s, src, .{ .h = heads, .hk = heads, .dim = 128 }, .{ .inv = metadata.ptr, .rot = 128 }, metadata.ptr + 256, null, .{ .q = old.ptr, .k = other.ptr, .v = 0 }, rows);
                for (0..2) |_| {
                    try r.rows(s, &qk, x, .{ .positions = positions[0..rows], .heads = heads, .dim = 128, .theta = theta }, new.ptr, other.ptr);
                    try s.synchronize();
                    const size = rows * width * 2;
                    try old.download(0, expected[0..size]);
                    try new.download(0, actual[0..size]);
                    if (!std.mem.eql(u8, expected[0..size], actual[0..size])) return error.DraftRopeMismatch;
                    cases += 1;
                }
            }
        };
    }
    std.debug.print("SUMMARY dflash2 rope cases={d} bf16_bits=PASS first=reuse=replace=PASS\n", .{cases});
}
