const std = @import("std");

pub fn run(m: anytype, seq: []const u32, rows: usize, logits: []f32) !void {
    var pos: usize = 0;
    const end = seq.len - @min(seq.len, @max(rows, 1));
    while (pos + 8 <= end) : (pos += 8) try m.batch(seq[pos..][0..8], pos);
    while (pos < seq.len) : (pos += 1) {
        const scored = pos + rows >= seq.len;
        try m.step(seq[pos], pos, scored or pos + 1 == seq.len);
        if (scored) try m.download(m.logits, std.mem.sliceAsBytes(logits[(pos + rows - seq.len) * m.w.config.vocab ..][0..m.w.config.vocab]));
    }
}

pub fn score(m: anytype, seq: []const u32, length: usize, rows: usize, logits: []f32) !void {
    for (length - rows..length) |pos| {
        try m.step(seq[pos], pos, true);
        try m.download(m.logits, std.mem.sliceAsBytes(logits[(pos + rows - length) * m.w.config.vocab ..][0..m.w.config.vocab]));
    }
}

test "descending quality and decode preserve the shorter causal prefixes" {
    const Cache = struct {
        w: struct { config: struct { vocab: usize = 1 } = .{} } = .{},
        logits: u64 = 0,
        cache: [1280]u32 = @splat(0),
        value: f32 = 0,
        pub fn step(m: *@This(), token: u32, pos: usize, head: bool) !void {
            m.cache[pos] = token;
            if (head) {
                var hash: u32 = 0;
                for (m.cache[0 .. pos + 1]) |t| hash = (hash *% 7 + t) % 65521;
                m.value = @floatFromInt(hash);
            }
        }
        pub fn batch(m: *@This(), tokens: []const u32, pos: usize) !void {
            for (tokens, 0..) |token, row| try m.step(token, pos + row, false);
        }
        pub fn download(m: *@This(), _: u64, out: []u8) !void {
            @memcpy(out, std.mem.asBytes(&m.value));
        }
    };
    var seq: [1024]u32 = undefined;
    for (&seq, 0..) |*token, pos| token.* = @intCast(pos % 31 + 1);
    var shared: Cache = .{};
    try run(&shared, &seq, 0, &.{});
    for ([_]usize{ 1024, 768, 512 }) |length| {
        var fresh: Cache = .{};
        var expected: [512]f32 = undefined;
        var actual: [512]f32 = undefined;
        try run(&fresh, seq[0..length], 512, &expected);
        try score(&shared, &seq, length, 512, &actual);
        try std.testing.expectEqualSlices(f32, &expected, &actual);
        for (length..length + 128) |pos| try shared.step(17, pos, true);
    }
}

test "timing prefill produces final logits with no scored downloads" {
    var m: Fake = .{};
    try run(&m, &(@as([1024]u32, @splat(1))), 0, &.{});
    try std.testing.expectEqual(@as(u32, 7), try m.greedy());
    try std.testing.expectEqual(@as(usize, 0), m.downloads);
    try std.testing.expectEqual(@as(usize, 127), m.batches);
}

test "quality prefill downloads exactly the final 512 initialized rows" {
    var m: Fake = .{};
    var logits: [512]f32 = undefined;
    try run(&m, &(@as([1024]u32, @splat(1))), 512, &logits);
    try std.testing.expectEqual(@as(usize, 512), m.downloads);
    try std.testing.expectEqual(@as(usize, 64), m.batches);
    try std.testing.expectEqual(@as(u32, 7), try m.greedy());
}
const Fake = struct {
    w: struct { config: struct { vocab: usize = 1 } = .{} } = .{},
    logits: u64 = 0,
    valid: bool = false,
    downloads: usize = 0,
    batches: usize = 0,
    fn batch(m: *Fake, _: []const u32, _: usize) !void {
        m.batches += 1;
    }
    fn step(m: *Fake, _: u32, _: usize, head: bool) !void {
        m.valid = head;
    }
    fn download(m: *Fake, _: u64, out: []u8) !void {
        if (!m.valid) return error.UninitializedLogits;
        m.downloads += 1;
        @memset(out, 0);
    }
    fn greedy(m: *Fake) !u32 {
        if (!m.valid) return error.UninitializedLogits;
        return 7;
    }
};
