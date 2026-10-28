const std = @import("std");
const json = @import("json");
const parse = @import("tool_parse.zig");
const params = @import("tool_params.zig");
const reply = @import("reply_text.zig");

pub fn calls(a: std.mem.Allocator, text: []const u8, tools: []const json.Value, max: ?usize) !parse.Parsed {
    const first = if (tools.len > 0) std.mem.indexOf(u8, text, "<tool_call>") else null;
    var out: std.ArrayList(json.Value) = .empty;
    if (first) |start| {
        const schemas = try params.schemasExact(a, tools);
        var pos = start;
        while (std.mem.indexOfPos(u8, text, pos, "<tool_call>")) |at| {
            const end = std.mem.indexOfPos(u8, text, at + 11, "</tool_call>") orelse pyEnd(text);
            const block = text[at + 11 .. end];
            pos = @min(end + 12, text.len);
            if (max) |m| if (out.items.len >= m) break;
            const function = header(block) orelse continue;
            const open = std.mem.indexOfScalarPos(u8, block, function, '>').? + 1;
            const close = std.mem.indexOfPos(u8, block, open, "</function>");
            const inner = block[open..(close orelse pyEnd(block))];
            const body = try std.mem.concat(a, u8, &.{ block[function..open], inner[0..pyEnd(inner)], "</function>" });
            const call = parse.functionBlock(a, body, &schemas, false, .qwen3_coder) catch |e| switch (e) {
                error.OutOfMemory => return error.OutOfMemory,
                error.Invalid => continue,
            } orelse continue;
            try out.append(a, try parse.openaiCall(a, call.name, call.arguments));
        }
    }
    return .{ .content = reply.pyStrip(if (out.items.len > 0) text[0..first.?] else text), .calls = if (out.items.len > 0) out.items else null };
}

// Where the pod's regex `$` matches when nothing closes a span: before a final newline, else at the end.
fn pyEnd(s: []const u8) usize {
    return if (std.mem.endsWith(u8, s, "\n")) s.len - 1 else s.len;
}

// The pod's `<function=([^>\n]+)>` search: skip headers that are empty or break at a newline before `>`.
fn header(block: []const u8) ?usize {
    var from: usize = 0;
    while (std.mem.indexOfPos(u8, block, from, "<function=")) |at| {
        const name = at + "<function=".len;
        const stop = std.mem.indexOfAnyPos(u8, block, name, ">\n") orelse return null;
        if (stop > name and block[stop] == '>') return at;
        from = at + 1;
    }
    return null;
}

pub fn deltas(a: std.mem.Allocator, parsed: []const json.Value) ![]json.Value {
    const out = try a.alloc(json.Value, parsed.len);
    for (parsed, out, 0..) |call, *dst, i| {
        const copy = try json.copyObject(a, call.object);
        try copy.put(a, "index", try json.intValue(a, i));
        dst.* = try parse.wrapCalls(a, .{ .object = copy });
    }
    return out;
}
