//! LFM2's tool calls: ``[name(key=value, ...), ...]`` between ``<|tool_call_start|>`` and ``<|tool_call_end|>``, the
//! values Python literals (its chat template writes strings in single quotes, lists and dicts as JSON).
const std = @import("std");
const json = @import("json.zig");
const Value = json.Value;
const Allocator = std.mem.Allocator;

pub const Call = struct { name: []const u8, arguments: Value };
const Error = error{ Invalid, OutOfMemory };

const Parser = struct {
    a: Allocator,
    s: []const u8,
    i: usize = 0,

    fn space(p: *Parser) void {
        while (p.i < p.s.len and std.ascii.isWhitespace(p.s[p.i])) p.i += 1;
    }

    fn eat(p: *Parser, ch: u8) bool {
        p.space();
        if (p.i < p.s.len and p.s[p.i] == ch) {
            p.i += 1;
            return true;
        }
        return false;
    }

    fn name(p: *Parser) Error![]const u8 {
        p.space();
        const start = p.i;
        while (p.i < p.s.len and (std.ascii.isAlphanumeric(p.s[p.i]) or p.s[p.i] == '_' or p.s[p.i] == '.' or p.s[p.i] == '-' or p.s[p.i] >= 0x80)) p.i += 1;
        if (p.i == start) return error.Invalid;
        return p.s[start..p.i];
    }

    /// ``name(key=value, ...)``: positional values are not a call's (the template writes keys).
    fn call(p: *Parser) Error!Call {
        const n = try p.name();
        if (!p.eat('(')) return error.Invalid;
        const args = try json.newObject(p.a);
        if (!p.eat(')')) while (true) {
            const key = try p.name();
            if (!p.eat('=')) return error.Invalid;
            try args.put(p.a, key, try p.value());
            if (p.eat(')')) break;
            if (!p.eat(',')) return error.Invalid;
            if (p.eat(')')) break; // a trailing comma
        };
        return .{ .name = n, .arguments = .{ .object = args } };
    }

    fn value(p: *Parser) Error!Value {
        p.space();
        if (p.i >= p.s.len) return error.Invalid;
        const ch = p.s[p.i];
        if (ch == '\'' or ch == '"') return .{ .string = try p.string() };
        if (ch == '[' or ch == '(') {
            const close: u8 = if (ch == '[') ']' else ')';
            p.i += 1;
            var items: std.ArrayList(Value) = .empty;
            if (!p.eat(close)) while (true) {
                try items.append(p.a, try p.value());
                if (p.eat(close)) break;
                if (!p.eat(',')) return error.Invalid;
                if (p.eat(close)) break;
            };
            return .{ .array = items.items };
        }
        if (ch == '{') {
            p.i += 1;
            const obj = try json.newObject(p.a);
            if (!p.eat('}')) while (true) {
                p.space();
                const key = switch (try p.value()) {
                    .string => |k| k,
                    .int => |k| k,
                    else => return error.Invalid,
                };
                if (!p.eat(':')) return error.Invalid;
                try obj.put(p.a, key, try p.value());
                if (p.eat('}')) break;
                if (!p.eat(',')) return error.Invalid;
                if (p.eat('}')) break;
            };
            return .{ .object = obj };
        }
        const start = p.i;
        while (p.i < p.s.len and (std.ascii.isAlphanumeric(p.s[p.i]) or p.s[p.i] == '.' or p.s[p.i] == '-' or p.s[p.i] == '+' or p.s[p.i] == '_')) p.i += 1;
        const word = p.s[start..p.i];
        if (word.len == 0) return error.Invalid;
        const words = [_]struct { []const u8, Value }{
            .{ "True", .{ .bool = true } },   .{ "true", .{ .bool = true } },
            .{ "False", .{ .bool = false } }, .{ "false", .{ .bool = false } },
            .{ "None", .null },               .{ "null", .null },
        };
        for (words) |w| if (std.mem.eql(u8, word, w[0])) return w[1];
        const digits = std.mem.trimStart(u8, word, "+-");
        if (digits.len > 0 and std.mem.indexOfNone(u8, digits, "0123456789_") == null) {
            var text: std.ArrayList(u8) = .empty;
            if (word[0] == '-') try text.append(p.a, '-');
            for (digits) |d| if (d != '_') try text.append(p.a, d);
            const lead = std.mem.trimStart(u8, text.items[@intFromBool(word[0] == '-')..], "0");
            const canonical = if (lead.len == 0) "0" else try std.mem.concat(p.a, u8, &.{ if (word[0] == '-') "-" else "", lead });
            return .{ .int = canonical };
        }
        const f = std.fmt.parseFloat(f64, word) catch return error.Invalid;
        if (!std.math.isFinite(f)) return error.Invalid;
        return .{ .float = f };
    }

    /// A quoted string (single, double or tripled quotes) with Python's escapes; a prefix like r'' is not read.
    fn string(p: *Parser) Error![]const u8 {
        const q = p.s[p.i];
        const triple = p.i + 2 < p.s.len and p.s[p.i + 1] == q and p.s[p.i + 2] == q;
        p.i += if (triple) 3 else 1;
        var out: std.ArrayList(u8) = .empty;
        while (p.i < p.s.len) {
            const ch = p.s[p.i];
            if (ch == q and (!triple or (p.i + 2 < p.s.len and p.s[p.i + 1] == q and p.s[p.i + 2] == q))) {
                p.i += if (triple) 3 else 1;
                return out.items;
            }
            if (ch != '\\') {
                try out.append(p.a, ch);
                p.i += 1;
                continue;
            }
            if (p.i + 1 >= p.s.len) return error.Invalid;
            const e = p.s[p.i + 1];
            p.i += 2;
            switch (e) {
                'n' => try out.append(p.a, '\n'),
                't' => try out.append(p.a, '\t'),
                'r' => try out.append(p.a, '\r'),
                '0' => try out.append(p.a, 0),
                '\\', '\'', '"' => try out.append(p.a, e),
                '\n' => {},
                'u', 'x', 'U' => {
                    const width: usize = switch (e) {
                        'x' => 2,
                        'u' => 4,
                        else => 8,
                    };
                    if (p.i + width > p.s.len) return error.Invalid;
                    var cp = std.fmt.parseInt(u21, p.s[p.i .. p.i + width], 16) catch return error.Invalid;
                    p.i += width;
                    // a JSON-style surrogate pair, as tojson writes characters past the BMP
                    if (cp >= 0xd800 and cp < 0xdc00 and p.i + 6 <= p.s.len and p.s[p.i] == '\\' and p.s[p.i + 1] == 'u') {
                        const lo = std.fmt.parseInt(u21, p.s[p.i + 2 .. p.i + 6], 16) catch return error.Invalid;
                        if (lo >= 0xdc00 and lo < 0xe000) {
                            cp = 0x10000 + ((cp - 0xd800) << 10) + (lo - 0xdc00);
                            p.i += 6;
                        }
                    }
                    var buf: [4]u8 = undefined;
                    const n = std.unicode.utf8Encode(cp, &buf) catch return error.Invalid;
                    try out.appendSlice(p.a, buf[0..n]);
                },
                else => try out.appendSlice(p.a, &.{ '\\', e }),
            }
        }
        return error.Invalid;
    }
};

/// The calls of a block's inside: ``[f(a=1), g()]``, or one call without the brackets; Invalid when it is neither.
pub fn calls(a: Allocator, block: []const u8) Error![]Call {
    var p: Parser = .{ .a = a, .s = block };
    var out: std.ArrayList(Call) = .empty;
    const listed = p.eat('[');
    if (!(listed and p.eat(']'))) while (true) {
        try out.append(a, try p.call());
        if (!listed) break;
        if (p.eat(']')) break;
        if (!p.eat(',')) return error.Invalid;
    };
    p.space();
    if (p.i != p.s.len) return error.Invalid;
    return out.items;
}

test "LFM2's calls read as names and JSON arguments" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const got = try calls(a, "[get_weather(city='Paris', days=3, units=\"metric\"), search(q='it\\'s', tags=[\"a\", 'b'], opts={\"deep\": true, 'n': None}), noop()]");
    try std.testing.expectEqual(@as(usize, 3), got.len);
    try std.testing.expectEqualStrings("get_weather", got[0].name);
    try std.testing.expectEqualStrings("{\"city\":\"Paris\",\"days\":3,\"units\":\"metric\"}", try json.stringify(a, got[0].arguments, .{ .ascii = false, .compact = true }));
    try std.testing.expectEqualStrings("{\"q\":\"it's\",\"tags\":[\"a\",\"b\"],\"opts\":{\"deep\":true,\"n\":null}}", try json.stringify(a, got[1].arguments, .{ .ascii = false, .compact = true }));
    try std.testing.expectEqualStrings("{}", try json.stringify(a, got[2].arguments, .{ .ascii = false, .compact = true }));
    const one = try calls(a, " run(cmd='ls -la\\nwc -l', timeout=2.5, n=-007) ");
    try std.testing.expectEqualStrings("{\"cmd\":\"ls -la\\nwc -l\",\"timeout\":2.5,\"n\":-7}", try json.stringify(a, one[0].arguments, .{ .ascii = false, .compact = true }));
    try std.testing.expectError(error.Invalid, calls(a, "[f(1)]"));
    try std.testing.expectError(error.Invalid, calls(a, "[f(a=1)] trailing"));
    try std.testing.expectError(error.Invalid, calls(a, "plain words"));
}
