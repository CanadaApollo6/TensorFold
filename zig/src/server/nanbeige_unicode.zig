const std = @import("std");
const data = @import("nanbeige_unicode_data.zig");
const Allocator = std.mem.Allocator;

pub const Names = struct {
    bytes: ?[]const u8 = null,

    pub fn lookup(names: *Names, a: Allocator, name: []const u8) Allocator.Error!?u21 {
        if (std.mem.startsWith(u8, name, "HANGUL SYLLABLE ")) return hangul(name[16..]);
        var upper: [128]u8 = undefined;
        if (name.len == 0 or name.len > upper.len) return null;
        for (name, upper[0..name.len]) |c, *u| {
            if (c >= 128) return null;
            u.* = std.ascii.toUpper(c);
        }
        const key = upper[0..name.len];
        for (data.families) |family| {
            const candidate = if (family.insensitive) key else name;
            if (!std.mem.startsWith(u8, candidate, family.prefix)) continue;
            const suffix = candidate[family.prefix.len..];
            if (suffix.len != 5 and (family.insensitive or suffix.len != 4)) return null;
            var cp: u21 = 0;
            for (suffix) |c| {
                const digit: u21 = if (c >= '0' and c <= '9') c - '0' else if (c >= 'A' and c <= 'F') c - 'A' + 10 else return null;
                cp = cp * 16 + digit;
            }
            for (family.ranges) |range| if (cp >= range[0] and cp <= range[1]) return cp;
            return null;
        }
        // Request-owned inflation avoids shared mutable state and costs nothing until an explicit name is used.
        if (names.bytes == null) {
            var in: std.Io.Reader = .fixed(@embedFile("fixtures/llama/unicode-15.names.gz"));
            var gz: std.compress.flate.Decompress = .init(&in, .gzip, &.{});
            names.bytes = gz.reader.allocRemaining(a, .unlimited) catch |err| switch (err) {
                error.OutOfMemory => return error.OutOfMemory,
                else => unreachable,
            };
        }
        const bytes = names.bytes.?;
        const count = std.mem.readInt(u32, bytes[0..4], .little);
        var lo: usize = 0;
        var hi: usize = count;
        while (lo < hi) {
            const mid = lo + (hi - lo) / 2;
            const at = 4 + count * 4 + std.mem.readInt(u32, bytes[4 + mid * 4 ..][0..4], .little);
            const stored = std.mem.sliceTo(bytes[at + 3 ..], 0);
            switch (std.mem.order(u8, key, stored)) {
                .lt => hi = mid,
                .gt => lo = mid + 1,
                .eq => return @intCast(std.mem.readInt(u24, bytes[at..][0..3], .little)),
            }
        }
        return null;
    }
};

fn hangul(name: []const u8) ?u21 {
    const leads = [_][]const u8{ "G", "GG", "N", "D", "DD", "R", "M", "B", "BB", "S", "SS", "", "J", "JJ", "C", "K", "T", "P", "H" };
    const vowels = [_][]const u8{ "A", "AE", "YA", "YAE", "EO", "E", "YEO", "YE", "O", "WA", "WAE", "OE", "YO", "U", "WEO", "WE", "WI", "YU", "EU", "YI", "I" };
    const tails = [_][]const u8{ "", "G", "GG", "GS", "N", "NJ", "NH", "D", "L", "LG", "LM", "LB", "LS", "LT", "LP", "LH", "M", "B", "BS", "S", "SS", "NG", "J", "C", "K", "T", "P", "H" };
    var rest = name;
    var code: u21 = 0;
    for ([_][]const []const u8{ &leads, &vowels, &tails }) |parts| {
        var best: ?usize = null;
        for (parts, 0..) |part, i| if (std.mem.startsWith(u8, rest, part)) {
            if (best == null or part.len > parts[best.?].len) best = i;
        };
        const i = best orelse return null;
        rest = rest[parts[i].len..];
        code = code * @as(u21, @intCast(parts.len)) + @as(u21, @intCast(i));
    }
    return if (rest.len == 0) 0xac00 + code else null;
}
