const std = @import("std");
const json = @import("json");
const numbers = @import("nanbeige_numbers.zig");
const unicode = @import("nanbeige_unicode.zig");
const Value = json.Value;
const Allocator = std.mem.Allocator;

pub fn parse(a: Allocator, text: []const u8) Allocator.Error!?Value {
    if (std.mem.indexOfScalar(u8, text, 0) != null) return null;
    const lf = try std.mem.replaceOwned(u8, a, text, "\r\n", "\n");
    const normalized = try std.mem.replaceOwned(u8, a, lf, "\r", "\n");
    var p: Literal = .{ .a = a, .s = std.mem.trimStart(u8, normalized, " \t") };
    return p.document() catch |e| switch (e) {
        error.OutOfMemory => return error.OutOfMemory,
        error.Invalid => return null,
    };
}

const Literal = struct {
    a: Allocator,
    s: []const u8,
    i: usize = 0,
    nesting: usize = 0,
    names: unicode.Names = .{},

    const Error = error{ Invalid, OutOfMemory };

    fn document(p: *Literal) Error!Value {
        while (true) {
            const line = p.i;
            p.space();
            if (p.i < p.s.len and p.s[p.i] == '\n') {
                p.i += 1;
                continue;
            }
            if (std.mem.trimStart(u8, p.s[line..p.i], "\x0c").len > 0) return error.Invalid;
            break;
        }
        var v = try p.expr(0);
        if (p.peek() == ',') {
            var items: std.ArrayList(Value) = .empty;
            try items.append(p.a, v);
            while (p.peek() == ',') {
                p.i += 1;
                if (p.peek() == null or p.peek() == '\n') break;
                try items.append(p.a, try p.expr(0));
            }
            v = .{ .array = items.items };
        }
        p.space();
        while (p.i < p.s.len and p.s[p.i] == '\n') {
            p.i += 1;
            const line = p.i;
            p.space();
            if (p.i == p.s.len and p.i > line and p.s[p.i - 1] != 12 and
                std.mem.indexOfScalar(u8, p.s[line..p.i], '#') == null) return error.Invalid;
        }
        if (p.i != p.s.len) return error.Invalid;
        return v;
    }

    fn space(p: *Literal) void {
        while (p.i < p.s.len) {
            const c = p.s[p.i];
            if (c == ' ' or c == '\t' or c == 12 or (p.nesting > 0 and (c == '\n' or c == '\r'))) {
                p.i += 1;
            } else if (c == '#') {
                while (p.i < p.s.len and p.s[p.i] != '\n' and p.s[p.i] != '\r') p.i += 1;
            } else if (c == '\\' and p.i + 1 < p.s.len and p.s[p.i + 1] == '\n') {
                p.i += 2;
            } else break;
        }
    }

    fn peek(p: *Literal) ?u8 {
        p.space();
        return if (p.i < p.s.len) p.s[p.i] else null;
    }

    fn expr(p: *Literal, depth: usize) Error!Value {
        if (depth > 200) return error.Invalid;
        const c = p.peek() orelse return error.Invalid;
        switch (c) {
            '[' => return p.sequence(']', depth),
            '(' => return p.sequence(')', depth),
            '{' => return p.dict(depth),
            '\'', '"' => return p.strings(),
            'r', 'R', 'u', 'U' => if (p.i + 1 < p.s.len and (p.s[p.i + 1] == '\'' or p.s[p.i + 1] == '"')) return p.strings(),
            else => {},
        }
        if (std.mem.startsWith(u8, p.s[p.i..], "None") and !p.wordAt(p.i + 4)) return p.word(4, .null);
        if (std.mem.startsWith(u8, p.s[p.i..], "True") and !p.wordAt(p.i + 4)) return p.word(4, .{ .bool = true });
        if (std.mem.startsWith(u8, p.s[p.i..], "False") and !p.wordAt(p.i + 5)) return p.word(5, .{ .bool = false });
        return p.number();
    }

    fn wordAt(p: *Literal, at: usize) bool {
        return at < p.s.len and (std.ascii.isAlphanumeric(p.s[at]) or p.s[at] == '_');
    }

    fn word(p: *Literal, n: usize, v: Value) Value {
        p.i += n;
        return v;
    }

    fn sequence(p: *Literal, close: u8, depth: usize) Error!Value {
        p.i += 1;
        p.nesting += 1;
        defer p.nesting -= 1;
        var items: std.ArrayList(Value) = .empty;
        var commas: usize = 0;
        while (true) {
            if (p.peek() == close) {
                p.i += 1;
                if (close == ')' and items.items.len == 1 and commas == 0) return items.items[0]; // (x) is x
                return .{ .array = items.items };
            }
            try items.append(p.a, try p.expr(depth + 1));
            const next = p.peek() orelse return error.Invalid;
            if (next == ',') {
                p.i += 1;
                commas += 1;
            } else if (next != close) return error.Invalid;
        }
    }

    fn dict(p: *Literal, depth: usize) Error!Value {
        p.i += 1;
        p.nesting += 1;
        defer p.nesting -= 1;
        const o = try json.newObject(p.a);
        var keys: std.ArrayList(Value) = .empty;
        if (p.peek() == '}') {
            p.i += 1;
            return .{ .object = o };
        }
        while (true) {
            const key = try p.expr(depth + 1);
            if (p.peek() != ':') return error.Invalid; // a set is no JSON value
            p.i += 1;
            const v = try p.expr(depth + 1);
            const name: []const u8 = switch (key) {
                .string => |s| s,
                .int => |t| t,
                .bool => |b| if (b) "true" else "false",
                .null => "null",
                .float => |f| try json.stringify(p.a, .{ .float = f }, .{}),
                else => return error.Invalid,
            };
            var matched: ?usize = null;
            for (keys.items, 0..) |old, i| if (try sameKey(p.a, old, key)) {
                matched = i;
                break;
            };
            if (matched) |i| {
                o.values()[i] = v;
            } else {
                try keys.append(p.a, key);
                // Python keys retain their types: 1 and "1" coexist and serialize as duplicate JSON names.
                try o.entries.append(p.a, .{ .hash = std.array_hash_map.hashString(name), .key = name, .value = v });
            }
            const next = p.peek() orelse return error.Invalid;
            if (next == ',') {
                p.i += 1;
                if (p.peek() == '}') {
                    p.i += 1;
                    return .{ .object = o };
                }
            } else if (next == '}') {
                p.i += 1;
                return .{ .object = o };
            } else return error.Invalid;
        }
    }

    fn sameKey(a: Allocator, x: Value, y: Value) Allocator.Error!bool {
        if (x == .string or y == .string) return x == .string and y == .string and std.mem.eql(u8, x.string, y.string);
        if (x == .null or y == .null) return x == .null and y == .null;
        const xi = if (x == .bool) (if (x.bool) "1" else "0") else if (x == .int) x.int else null;
        const yi = if (y == .bool) (if (y.bool) "1" else "0") else if (y == .int) y.int else null;
        if (xi != null and yi != null) return std.mem.eql(u8, xi.?, yi.?);
        if (x == .float and y == .float) return x.float == y.float;
        const f = if (x == .float) x.float else y.float;
        if (!std.math.isFinite(f) or @trunc(f) != f) return false;
        return std.mem.eql(u8, (try numbers.integralFloat(a, f)).int, xi orelse yi.?);
    }

    /// One or more adjacent string literals, joined as Python's parser joins them.
    fn strings(p: *Literal) Error!Value {
        var out: std.ArrayList(u8) = .empty;
        var any = false;
        while (p.peek()) |c| {
            var raw = false;
            if (c == 'r' or c == 'R' or c == 'u' or c == 'U') {
                if (p.i + 1 >= p.s.len or (p.s[p.i + 1] != '\'' and p.s[p.i + 1] != '"')) break;
                raw = c == 'r' or c == 'R';
                p.i += 1;
            } else if (c != '\'' and c != '"') break;
            try p.string(&out, raw);
            any = true;
        }
        if (!any) return error.Invalid;
        return .{ .string = out.items };
    }

    fn string(p: *Literal, out: *std.ArrayList(u8), raw: bool) Error!void {
        const q = p.s[p.i];
        const triple = p.i + 2 < p.s.len and p.s[p.i + 1] == q and p.s[p.i + 2] == q;
        p.i += if (triple) 3 else 1;
        while (true) {
            if (p.i >= p.s.len) return error.Invalid;
            const c = p.s[p.i];
            if (c == q and (!triple or (p.i + 2 < p.s.len and p.s[p.i + 1] == q and p.s[p.i + 2] == q))) {
                p.i += if (triple) 3 else 1;
                return;
            }
            if (c == '\n' and !triple) return error.Invalid;
            if (c != '\\') {
                try out.append(p.a, c);
                p.i += 1;
                continue;
            }
            if (p.i + 1 >= p.s.len) return error.Invalid;
            const e = p.s[p.i + 1];
            if (raw) {
                try out.appendSlice(p.a, p.s[p.i .. p.i + 2]);
                p.i += 2;
                continue;
            }
            p.i += 2;
            const simple: ?u8 = switch (e) {
                'n' => '\n',
                't' => '\t',
                'r' => '\r',
                '\\' => '\\',
                '\'' => '\'',
                '"' => '"',
                'a' => 7,
                'b' => 8,
                'f' => 12,
                'v' => 11,
                '0'...'7' => blk: {
                    var v: u21 = e - '0';
                    var n: usize = 1;
                    while (n < 3 and p.i < p.s.len and p.s[p.i] >= '0' and p.s[p.i] <= '7') : (n += 1) {
                        v = v * 8 + (p.s[p.i] - '0');
                        p.i += 1;
                    }
                    try appendCode(p.a, out, v);
                    break :blk null;
                },
                'x', 'u', 'U' => blk: {
                    const n: usize = if (e == 'x') 2 else if (e == 'u') 4 else 8;
                    if (p.i + n > p.s.len) return error.Invalid;
                    for (p.s[p.i .. p.i + n]) |digit| if (!std.ascii.isHex(digit)) return error.Invalid;
                    const v = std.fmt.parseInt(u21, p.s[p.i .. p.i + n], 16) catch return error.Invalid;
                    p.i += n;
                    try appendCode(p.a, out, v);
                    break :blk null;
                },
                '\n' => null,
                'N' => blk: {
                    if (p.i >= p.s.len or p.s[p.i] != '{') return error.Invalid;
                    const start = p.i + 1;
                    const end = std.mem.indexOfScalarPos(u8, p.s, start, '}') orelse return error.Invalid;
                    const v = (try p.names.lookup(p.a, p.s[start..end])) orelse return error.Invalid;
                    p.i = end + 1;
                    try appendCode(p.a, out, v);
                    break :blk null;
                },
                else => blk: {
                    try out.appendSlice(p.a, &.{ '\\', e });
                    break :blk null;
                },
            };
            if (simple) |ch| try out.append(p.a, ch);
        }
    }

    fn appendCode(a: Allocator, out: *std.ArrayList(u8), v: u21) Error!void {
        if (v > 0x10ffff) return error.Invalid;
        var buf: [4]u8 = undefined;
        const n = @import("json").wtf8Encode(v, &buf);
        try out.appendSlice(a, buf[0..n]);
    }

    fn number(p: *Literal) Error!Value {
        var negative = false;
        if (p.peek()) |c| if (c == '-' or c == '+') {
            negative = c == '-';
            p.i += 1;
            p.space();
        };
        var grouped: usize = 0;
        while (p.peek() == '(') {
            p.i += 1;
            p.nesting += 1;
            grouped += 1;
        }
        const start = p.i;
        while (p.i < p.s.len and (std.ascii.isAlphanumeric(p.s[p.i]) or p.s[p.i] == '_' or p.s[p.i] == '.' or
            ((p.s[p.i] == '+' or p.s[p.i] == '-') and p.i > start and (p.s[p.i - 1] == 'e' or p.s[p.i - 1] == 'E')))) p.i += 1;
        const text = p.s[start..p.i];
        for (0..grouped) |_| {
            if (p.peek() != ')') return error.Invalid;
            p.i += 1;
            p.nesting -= 1;
        }
        if (text.len == 0) return error.Invalid;
        const sign: []const u8 = if (negative) "-" else "";
        if (text.len > 2 and text[0] == '0' and std.ascii.findIgnoreCase("xob", text[1..2]) != null) {
            const radix: u8 = switch (std.ascii.toLower(text[1])) {
                'x' => 16,
                'o' => 8,
                else => 2,
            };
            const part = if (text[2] == '_') text[3..] else text[2..];
            const ds = (try numbers.digits(p.a, part, radix)) orelse return error.Invalid;
            var big = try std.math.big.int.Managed.init(p.a);
            defer big.deinit();
            big.setString(radix, ds) catch |e| switch (e) {
                error.OutOfMemory => return error.OutOfMemory,
                else => return error.Invalid,
            };
            if (negative and !big.toConst().eqlZero()) big.negate();
            return .{ .int = big.toString(p.a, 10, .lower) catch |e| switch (e) {
                error.OutOfMemory => return error.OutOfMemory,
                else => unreachable,
            } };
        }
        if (std.mem.indexOfAny(u8, text, ".eE") == null) {
            const ds = (try numbers.digits(p.a, text, 10)) orelse return error.Invalid;
            if (ds.len > 1 and ds[0] == '0' and std.mem.trimStart(u8, ds, "0").len > 0) return error.Invalid;
            return .{ .int = (try numbers.integer(p.a, try std.mem.concat(p.a, u8, &.{ sign, text }))) orelse return error.Invalid };
        }
        const f = (try numbers.floating(p.a, text)) orelse return error.Invalid;
        return .{ .float = if (negative) -f else f };
    }
};
