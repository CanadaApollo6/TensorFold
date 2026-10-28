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
            const end = std.mem.indexOfPos(u8, text, at + 11, "</tool_call>") orelse text.len;
            const block = text[at + 11 .. end];
            pos = @min(end + 12, text.len);
            if (max) |m| if (out.items.len >= m) break;
            const function = std.mem.indexOf(u8, block, "<function=") orelse continue;
            var body = block[function..];
            if (std.mem.indexOf(u8, body, "</function>")) |close| body = body[0 .. close + 11];
            if (!std.mem.endsWith(u8, body, "</function>")) body = try std.mem.concat(a, u8, &.{ body, "</function>" });
            const call = parse.functionBlock(a, body, &schemas, false, .qwen3_coder) catch |e| switch (e) {
                error.OutOfMemory => return error.OutOfMemory,
                error.Invalid => continue,
            } orelse continue;
            try out.append(a, try parse.openaiCall(a, call.name, call.arguments));
        }
    }
    return .{ .content = reply.pyStrip(if (out.items.len > 0) text[0..first.?] else text), .calls = if (out.items.len > 0) out.items else null };
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
