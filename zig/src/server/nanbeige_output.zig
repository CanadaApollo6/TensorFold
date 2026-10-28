const std = @import("std");
const json = @import("json");
const numbers = @import("nanbeige_numbers.zig");
const Allocator = std.mem.Allocator;
const Value = json.Value;
const strip = @import("reply_text.zig").pyStrip;

pub const Parsed = struct { content: []const u8, calls: ?[]Value };
pub const Split = @import("reply_text.zig").Split;

pub fn split(_: Allocator, text: []const u8) !Split {
    const close = std.mem.indexOf(u8, text, "</think>");
    const thought = if (close) |c| text[0..c] else text;
    const reasoning = std.mem.trim(u8, if (std.mem.startsWith(u8, thought, "<think>")) thought[7..] else thought, "\n");
    return .{ .reasoning = reasoning, .answer = if (close) |c| text[c + 8 ..] else "" };
}

pub fn typed(a: Allocator, value: []const u8, schema: Value) Allocator.Error!Value {
    var kind = schema.get("type") orelse Value{ .string = "string" };
    if (kind == .array) kind = for (kind.array) |k| {
        if (k != .string or !std.mem.eql(u8, k.string, "null")) break k;
    } else Value{ .string = "string" };
    const name = if (kind == .string) kind.string else "";
    if (std.ascii.eqlIgnoreCase(value, "null") and !std.mem.eql(u8, name, "string")) return .null;
    if (is(name, &.{ "string", "str", "text" })) return .{ .string = value };
    if (is(name, &.{ "integer", "int" }))
        return .{ .int = (try numbers.integer(a, value)) orelse return .{ .string = value } };
    if (is(name, &.{ "number", "float" })) {
        const f = (try numbers.floating(a, value)) orelse return .{ .string = value };
        if (std.math.isFinite(f) and @trunc(f) == f and std.mem.indexOfScalar(u8, value, '.') == null)
            return numbers.integralFloat(a, f);
        return .{ .float = f };
    }
    if (is(name, &.{ "boolean", "bool" })) return .{ .bool = std.ascii.eqlIgnoreCase(strip(value), "true") };
    return switch (try json.parseText(a, value)) {
        .ok => |v| v,
        .err => (try @import("nanbeige_literal.zig").parse(a, value)) orelse .{ .string = value },
    };
}

fn is(name: []const u8, names: []const []const u8) bool {
    for (names) |n| if (std.mem.eql(u8, name, n)) return true;
    return false;
}

pub fn calls(a: Allocator, text: []const u8, tools: []const Value) Allocator.Error!Parsed {
    const first = if (tools.len > 0) std.mem.indexOf(u8, text, "<tool_call>") else null;
    const schemas = try json.newObject(a);
    for (tools) |tool| {
        if (tool.get("type")) |t| if (t != .string or !std.mem.eql(u8, t.string, "function")) continue;
        const f = tool.get("function") orelse continue;
        const name = f.strField("name") orelse continue;
        const props = if (f.get("parameters")) |p| p.get("properties") orelse .null else Value.null;
        try schemas.put(a, name, props);
    }
    var out: std.ArrayList(Value) = .empty;
    if (first) |start| {
        var pos = start;
        while (std.mem.indexOfPos(u8, text, pos, "<tool_call>")) |at| {
            const end = std.mem.indexOfPos(u8, text, at + 11, "</tool_call>") orelse pyEnd(text);
            const block = text[at + 11 .. end];
            pos = @min(end + 12, text.len);
            const f = header(block, "<function=") orelse continue;
            const name = strip(block[f.name..f.open]);
            const close = std.mem.indexOfPos(u8, block, f.open + 1, "</function>") orelse pyEnd(block);
            const body = block[f.open + 1 .. close];
            const args = try json.newObject(a);
            var ppos: usize = 0;
            while (header(body[ppos..], "<parameter=")) |p| {
                const key = strip(body[ppos + p.name .. ppos + p.open]);
                const begin = ppos + p.open + 1;
                const closed = std.mem.indexOfPos(u8, body, begin, "</parameter>");
                const next = std.mem.indexOfPos(u8, body, begin, "<parameter=");
                const endp = @min(@min(closed orelse body.len, next orelse body.len), pyEnd(body));
                var v = body[begin..endp];
                if (std.mem.startsWith(u8, v, "\n")) v = v[1..];
                if (std.mem.endsWith(u8, v, "\n")) v = v[0 .. v.len - 1];
                const props = schemas.get(name) orelse Value.null;
                try args.put(a, key, try typed(a, v, props.get(key) orelse .null));
                ppos = if (closed == endp) endp + 12 else endp;
            }
            const function = try json.newObject(a);
            try function.put(a, "name", .{ .string = name });
            try function.put(a, "arguments", .{ .string = try json.stringify(a, .{ .object = args }, .{ .ascii = false }) });
            const call = try json.newObject(a);
            try call.put(a, "id", .{ .string = try @import("nanbeige_response.zig").id(a, true) });
            try call.put(a, "type", .{ .string = "function" });
            try call.put(a, "function", .{ .object = function });
            try out.append(a, .{ .object = call });
        }
    }
    return .{ .content = strip(if (out.items.len > 0) text[0..first.?] else text), .calls = if (out.items.len > 0) out.items else null };
}

// Python's lazy .*? reaches `$` before a final newline, even with re.S.
fn pyEnd(s: []const u8) usize {
    return if (std.mem.endsWith(u8, s, "\n")) s.len - 1 else s.len;
}

fn header(block: []const u8, tag: []const u8) ?struct { name: usize, open: usize } {
    var from: usize = 0;
    while (std.mem.indexOfPos(u8, block, from, tag)) |at| {
        const name = at + tag.len;
        const stop = std.mem.indexOfAnyPos(u8, block, name, ">\n") orelse return null;
        if (stop > name and block[stop] == '>') return .{ .name = name, .open = stop };
        from = at + 1;
    }
    return null;
}
