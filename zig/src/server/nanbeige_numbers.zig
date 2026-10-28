const std = @import("std");
const json = @import("json");
const Allocator = std.mem.Allocator;

// CPython numeric conversion accepts Unicode White_Space, but excludes str.strip's U+001C through U+001F.
fn whitespace(c: u21) bool {
    return switch (c) {
        9...13, 32, 0x85, 0xa0, 0x1680, 0x2000...0x200a, 0x2028, 0x2029, 0x202f, 0x205f, 0x3000 => true,
        else => false,
    };
}

// Unicode 15 Nd blocks, matching the reference pod's Python 3.12 Unicode database.
const zeros = [_]u21{ 0x30, 0x660, 0x6f0, 0x7c0, 0x966, 0x9e6, 0xa66, 0xae6, 0xb66, 0xbe6, 0xc66, 0xce6, 0xd66, 0xde6, 0xe50, 0xed0, 0xf20, 0x1040, 0x1090, 0x17e0, 0x1810, 0x1946, 0x19d0, 0x1a80, 0x1a90, 0x1b50, 0x1bb0, 0x1c40, 0x1c50, 0xa620, 0xa8d0, 0xa900, 0xa9d0, 0xa9f0, 0xaa50, 0xabf0, 0xff10, 0x104a0, 0x10d30, 0x11066, 0x110f0, 0x11136, 0x111d0, 0x112f0, 0x11450, 0x114d0, 0x11650, 0x116c0, 0x11730, 0x118e0, 0x11950, 0x11c50, 0x11d50, 0x11da0, 0x11f50, 0x16a60, 0x16ac0, 0x16b50, 0x1d7ce, 0x1d7d8, 0x1d7e2, 0x1d7ec, 0x1d7f6, 0x1e140, 0x1e2f0, 0x1e4f0, 0x1e950, 0x1fbf0 };

pub fn normalized(a: Allocator, text: []const u8) ![]const u8 {
    var out: std.ArrayList(u8) = .empty;
    var it = (std.unicode.Utf8View.init(text) catch return "").iterator();
    while (it.nextCodepoint()) |c| {
        if (whitespace(c)) {
            try out.append(a, ' ');
        } else if (c < 128) {
            try out.append(a, @intCast(c));
        } else {
            const digit: u8 = for (zeros) |zero| {
                if (c >= zero and c < zero + 10) break @intCast(c - zero);
            } else return "";
            try out.append(a, '0' + digit);
        }
    }
    return std.mem.trim(u8, out.items, " ");
}

pub fn digits(a: Allocator, s: []const u8, base: u8) !?[]const u8 {
    if (s.len == 0) return null;
    var out: std.ArrayList(u8) = .empty;
    for (s, 0..) |ch, i| {
        if (ch == '_') {
            if (i == 0 or i + 1 == s.len or s[i - 1] == '_') return null;
        } else {
            const n = std.fmt.charToDigit(ch, base) catch return null;
            _ = n;
            try out.append(a, ch);
        }
    }
    return out.items;
}

pub fn integer(a: Allocator, text: []const u8) !?[]const u8 {
    var s = try normalized(a, text);
    const negative = s.len > 0 and s[0] == '-';
    if (s.len > 0 and (s[0] == '-' or s[0] == '+')) s = s[1..];
    const ds = (try digits(a, s, 10)) orelse return null;
    if (ds.len > 4300) return null;
    const trimmed = std.mem.trimStart(u8, ds, "0");
    if (trimmed.len == 0) return "0";
    return if (negative) try std.mem.concat(a, u8, &.{ "-", trimmed }) else trimmed;
}

pub fn floating(a: Allocator, text: []const u8) !?f64 {
    const s = try normalized(a, text);
    var i: usize = 0;
    if (s.len > 0 and (s[0] == '-' or s[0] == '+')) i = 1;
    const word = s[i..];
    if (std.ascii.eqlIgnoreCase(word, "nan")) return std.math.nan(f64);
    if (std.ascii.eqlIgnoreCase(word, "inf") or std.ascii.eqlIgnoreCase(word, "infinity"))
        return if (s[0] == '-') -std.math.inf(f64) else std.math.inf(f64);
    const begin = i;
    while (i < s.len and (std.ascii.isDigit(s[i]) or s[i] == '_')) i += 1;
    const whole = s[begin..i];
    if (whole.len > 0 and try digits(a, whole, 10) == null) return null;
    var fraction: []const u8 = "";
    if (i < s.len and s[i] == '.') {
        i += 1;
        const start = i;
        while (i < s.len and (std.ascii.isDigit(s[i]) or s[i] == '_')) i += 1;
        fraction = s[start..i];
        if (fraction.len > 0 and try digits(a, fraction, 10) == null) return null;
    }
    if (whole.len == 0 and fraction.len == 0) return null;
    if (i < s.len and (s[i] == 'e' or s[i] == 'E')) {
        i += 1;
        if (i < s.len and (s[i] == '-' or s[i] == '+')) i += 1;
        if (try digits(a, s[i..], 10) == null) return null;
        i = s.len;
    }
    if (i != s.len) return null;
    const clean = try std.mem.replaceOwned(u8, a, s, "_", "");
    return std.fmt.parseFloat(f64, clean) catch null;
}

pub fn integralFloat(a: Allocator, f: f64) !json.Value {
    const n: i2048 = @intFromFloat(f);
    return json.intValue(a, n);
}
