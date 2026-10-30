const std = @import("std");
const l = @import("lattice.zig");
fn bits(v: f32) u16 {
    return @intCast(@as(u32, @bitCast(v)) >> 16);
}
test "compact lattice preserves BF16 candidates and rounded normalizers at real vocab widths" {
    const a = std.testing.allocator;
    var certified: usize = 0;
    for ([_]usize{ 17, 255, 4095, 4096, 4097, 166912, 248320 }) |width| {
        const raw = try a.alloc(u16, width);
        defer a.free(raw);
        for (0..9) |kind| {
            for (raw, 0..) |*b, i| b.* = bits(switch (kind) {
                0 => 0,
                1 => @as(f32, @floatFromInt(i % 73)) / 16,
                2 => if (i < 17) 1 else -100,
                3 => if (i % 31 == 0) 65536 else -65536,
                4 => if (i < 15) 4 else if (i == 15) 1.0078125 else 1,
                5 => if (i % 2 == 0) @as(f32, @bitCast(@as(u32, 0x7f7f0000))) else -3.3895314e38,
                6 => if (i % 2 == 0) -0.0 else 0.0,
                7 => if (i < 15) 4 else if (i >= 4095 and i < 4098) 1 else 0,
                else => @as(f32, @bitCast(@as(u32, @intCast((i * 7919) % 128)) << 16)) * if (i % 5 == 0) @as(f32, -1) else 1,
            });
            const old = try l.reference(raw, 16);
            const gpu = l.emulate(raw, 16);
            try std.testing.expectEqualSlices(u32, old.ids[0..16], gpu.ids[0..16]);
            try std.testing.expectEqualSlices(u32, @as(*const [16]u32, @ptrCast(&old.values)), @as(*const [16]u32, @ptrCast(&gpu.values)));
            try std.testing.expectApproxEqAbs(old.log_mass, gpu.log_mass, l.tolerance(width));
            const z = (try l.normalizer(gpu, width)) orelse l.exactNormalizer(old);
            try std.testing.expectEqual(@as(u32, @bitCast(l.exactNormalizer(old))), @as(u32, @bitCast(z)));
            certified += @intFromBool((try l.normalizer(gpu, width)) != null);
        }
        for ([_]u16{ 0x7fc0, 0x7f80, 0xff80 }) |bad| {
            @memset(raw, 0);
            raw[width - 1] = bad;
            try std.testing.expectError(error.DraftLatticeFinite, l.reference(raw, 16));
            try std.testing.expectError(error.DraftLatticeFinite, l.normalizer(l.emulate(raw, 16), width));
        }
    }
    try std.testing.expectEqual(@as(usize, 63), certified);
}
test "ambiguous FP32 rounding requests exact CPU fallback" {
    const midpoint = (@as(f64, 1) + @as(f64, std.math.nextAfter(f32, 1, 2))) / 2;
    try std.testing.expectEqual(@as(?f32, null), try l.normalizer(.{ .log_mass = midpoint }, 166912));
    try std.testing.expectEqual(@as(usize, 144), @sizeOf(l.Row));
}

const Host = struct {
    logits: []const u16,
    downloads: usize = 0,
    pub fn enter(_: *Host) !void {}
    pub fn leave(_: *Host, _: void) void {}
    pub fn download(h: *Host, address: u64, bytes: []u8) !void {
        const all = std.mem.sliceAsBytes(h.logits);
        if (address > all.len or bytes.len > all.len - address) return error.DraftLatticeFallbackRange;
        @memcpy(bytes, all[address..][0..bytes.len]);
        h.downloads += 1;
    }
};
test "an uncertified row recomputes the old scalar normalizer from its own logits" {
    const a = std.testing.allocator;
    const vocab = 166912;
    const raw = try a.alloc(u16, 2 * vocab);
    defer a.free(raw);
    for (raw, 0..) |*b, i| b.* = bits(if (i <= vocab) 0 else -32);
    const rows = [_]l.Row{ l.emulate(raw[0..vocab], 16), l.emulate(raw[vocab..], 16) };
    try std.testing.expect((try l.normalizer(rows[0], vocab)) != null);
    try std.testing.expectEqual(@as(?f32, null), try l.normalizer(rows[1], vocab));
    try std.testing.expectEqual(@as(u32, 0x31114213), @as(u32, @bitCast(l.exactNormalizer(rows[1]))));
    var host: Host = .{ .logits = raw };
    var ids: [32]u32 = undefined;
    var unary: [32]f32 = undefined;
    try std.testing.expectEqual(@as(usize, 1), try l.unary(a, &rows, 0, vocab, vocab, 16, &host, &ids, &unary));
    try std.testing.expectEqual(@as(usize, 1), host.downloads);
    for (rows, 0..) |row, r| {
        const old = try l.reference(raw[r * vocab ..][0..vocab], 16);
        if (r == 1) try std.testing.expectEqual(@as(u32, 0x31112bc7), @as(u32, @bitCast(l.exactNormalizer(old))));
        try std.testing.expectEqualSlices(u32, old.ids[0..16], ids[r * 16 ..][0..16]);
        try std.testing.expectEqualSlices(u8, std.mem.asBytes(&old.values), std.mem.asBytes(&row.values));
        for (old.values, unary[r * 16 ..][0..16]) |v, got| {
            try std.testing.expectEqual(@as(u32, @bitCast(v - l.exactNormalizer(old))), @as(u32, @bitCast(got)));
        }
    }
}

test "slice merge keeps all supported candidate widths and rejects no finite BF16 patterns" {
    const raw = try std.testing.allocator.alloc(u16, 4097);
    defer std.testing.allocator.free(raw);
    for (0..16) |seed| {
        for (raw, 0..) |*b, id| {
            b.* = @truncate((id * 7919 + seed * 104729) % 65536);
            if ((b.* & 0x7f80) == 0x7f80) b.* &= 0xff7f;
        }
        for ([_]usize{ 1, 2, 8, 16 }) |k| {
            const old = try l.reference(raw, k);
            const gpu = l.emulate(raw, k);
            try std.testing.expectEqualSlices(u32, old.ids[0..k], gpu.ids[0..k]);
            try std.testing.expectEqualSlices(u8, std.mem.sliceAsBytes(old.values[0..k]), std.mem.sliceAsBytes(gpu.values[0..k]));
            try std.testing.expectApproxEqAbs(old.log_mass, gpu.log_mass, l.tolerance(raw.len));
            const z = (try l.normalizer(gpu, raw.len)) orelse l.exactNormalizer(old);
            for (old.values[0..k], gpu.values[0..k]) |v, got| {
                try std.testing.expectEqual(@as(u32, @bitCast(v - l.exactNormalizer(old))), @as(u32, @bitCast(got - z)));
            }
        }
    }
}
