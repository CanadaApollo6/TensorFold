const std = @import("std");
const json = @import("json");
const Cx = @import("errors.zig").Cx;
const Value = json.Value;

pub fn defaults(a: std.mem.Allocator, configured: ?Value) !Value {
    const out = try json.newObject(a);
    try out.put(a, "temperature", .{ .float = 1 });
    try out.put(a, "top_p", .{ .float = 0.95 });
    try out.put(a, "top_k", .{ .int = "0" });
    try out.put(a, "min_p", .{ .float = 0 });
    if (configured) |cfg| if (cfg == .object) {
        for (cfg.object.keys(), cfg.object.values()) |key, value| if (value != .null) try out.put(a, key, value);
    };
    return .{ .object = out };
}

pub fn prepare(cx: *Cx, messages: ?Value) !?Value {
    const list = messages orelse return null;
    if (list != .array) return list;
    const out = try cx.a.alloc(Value, list.array.len);
    for (list.array, out) |message, *dst| {
        dst.* = message;
        if (message != .object) continue;
        const copy = try json.copyObject(cx.a, message.object);
        if (message.get("content")) |content| if (content == .array) {
            var text: std.ArrayList(u8) = .empty;
            for (content.array, 0..) |part, i| {
                const piece = if (part == .string) part.string else if (part.strField("type")) |kind|
                    (if (std.mem.eql(u8, kind, "text")) part.strField("text") orelse "" else return cx.refuse("Nanbeige is not a multimodal model"))
                else
                    return cx.refuse("Nanbeige is not a multimodal model");
                if (i > 0) try text.append(cx.a, '\n');
                try text.appendSlice(cx.a, piece);
            }
            try copy.put(cx.a, "content", .{ .string = text.items });
        };
        if (message.strField("role")) |role| if (std.mem.eql(u8, role, "assistant")) {
            if (message.get("tool_calls")) |calls| if (calls == .array) {
                const normalized = try cx.a.dupe(Value, calls.array);
                for (calls.array, normalized) |call, *slot| {
                    const function = call.get("function") orelse continue;
                    const args = function.strField("arguments") orelse continue;
                    if ((try json.parseText(cx.a, args)) == .ok) continue;
                    const f = try json.copyObject(cx.a, function.object);
                    try f.put(cx.a, "arguments", .{ .object = try json.newObject(cx.a) });
                    const c = try json.copyObject(cx.a, call.object);
                    try c.put(cx.a, "function", .{ .object = f });
                    slot.* = .{ .object = c };
                }
                try copy.put(cx.a, "tool_calls", .{ .array = normalized });
            };
            const reasoning = message.field("reasoning_content");
            if (reasoning == null or !reasoning.?.truthy()) try copy.put(cx.a, "reasoning_content", message.get("reasoning") orelse .null);
        };
        dst.* = .{ .object = copy };
    }
    return .{ .array = out };
}

// The pod's `text.partition("</think>")`: the answer keeps its whitespace for the tool parser, which strips content.
pub fn split(_: std.mem.Allocator, text: []const u8) !@import("reply_text.zig").Split {
    const close = std.mem.indexOf(u8, text, "</think>");
    const thought = if (close) |c| text[0..c] else text;
    const reasoning = std.mem.trim(u8, if (std.mem.startsWith(u8, thought, "<think>")) thought[7..] else thought, "\n");
    return .{ .reasoning = reasoning, .answer = if (close) |c| text[c + 8 ..] else "" };
}

pub fn stream(g: anytype, text: []const u8, finished: bool) !void {
    const sink = g.sink orelse return;
    const close = std.mem.indexOf(u8, text, "</think>");
    var end = close orelse text.len;
    var held: usize = 0;
    if (close == null and !finished) while (end > 0 and held < 7) : (held += 1) {
        end -= 1;
        while (end > 0 and text[end] & 0xc0 == 0x80) end -= 1;
    };
    const sent = g.streamed_reasoning.text.items.len;
    if (end <= sent) return;
    const delta = try @import("chat.zig").deltaOf(g.a, "reasoning_content", text[sent..end]);
    try delta.object.put(g.a, "reasoning", .{ .string = text[sent..end] });
    sink.call(sink.ctx, delta) catch return error.Cancelled;
    try g.streamed_reasoning.text.appendSlice(g.a, text[sent..end]);
}

pub fn finishStream(g: anytype, text: []const u8, content: []const u8) !void {
    try stream(g, text, true);
    if (g.tools.len == 0 and content.len > 0) if (g.sink) |sink| {
        sink.call(sink.ctx, .{ .string = content }) catch return error.Cancelled;
    };
}

pub fn attach(a: std.mem.Allocator, reply: *@import("chat.zig").Reply, tools: []const Value, max: ?usize) !?[]Value {
    const parsed = try @import("tool_qwen.zig").calls(a, reply.content, tools, max);
    reply.content = parsed.content;
    if (parsed.calls != null) reply.finish_reason = "tool_calls";
    return parsed.calls;
}

pub fn sampling(cx: *Cx, f: Value) !void {
    if (f.field("repetition_penalty")) |value| {
        const n = switch (value) {
            .float => value.float,
            .int => std.fmt.parseFloat(f64, value.int) catch 0,
            else => 0,
        };
        if (n != 1) return cx.refuse("Nanbeige does not support repetition_penalty; send 1 or omit it");
    }
    if (f.field("temperature")) |value| if (value == .float and value.float < 0)
        return cx.refuse("temperature must be nonnegative");
    if (f.field("top_p")) |value| if (value == .float and (value.float < 0 or value.float > 1))
        return cx.refuse("top_p must be between 0 and 1");
}
