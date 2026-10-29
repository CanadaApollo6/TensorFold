const std = @import("std");
pub const default_rows = 128;
pub const max_rows = 1024;
pub const attention_rows = 8;

pub fn parse(value: []const u8) !usize {
    const n = std.fmt.parseInt(usize, value, 10) catch return error.BadPrefillRows;
    if (n == 0 or n > max_rows) return error.BadPrefillRows;
    return n;
}

pub fn rows() usize {
    return if (std.c.getenv("TF_LLAMA_PREFILL_ROWS")) |v| parse(std.mem.span(v)) catch default_rows else default_rows;
}

pub fn end(start: usize, length: usize, chunk: usize) usize {
    return start + @min(length - start, chunk);
}

pub fn cacheOffset(position: usize, heads: usize, row_bytes: usize) usize {
    return position * heads * row_bytes;
}

test "prefill boundaries and kv4 absolute cache writes" {
    try std.testing.expectEqual(@as(usize, 128), end(0, 257, 128));
    try std.testing.expectEqual(@as(usize, 256), end(128, 257, 128));
    try std.testing.expectEqual(@as(usize, 257), end(256, 257, 128));
    try std.testing.expectEqual(@as(usize, 145), end(17, 257, 128));
    for ([_]usize{ 0, 1, 127, 128, 129, 2047, 20000 }) |position| {
        try std.testing.expectEqual(position * 576, cacheOffset(position, 8, 72));
    }
    try std.testing.expectError(error.BadPrefillRows, parse("0"));
    try std.testing.expectError(error.BadPrefillRows, parse("1025"));
    try std.testing.expectEqual(@as(usize, 1), try parse("1"));
}
