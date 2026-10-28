const std = @import("std");
const json = @import("json");
const Cx = @import("errors.zig").Cx;
const Value = json.Value;

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
            const reasoning = message.field("reasoning_content");
            if (reasoning == null or !reasoning.?.truthy()) try copy.put(cx.a, "reasoning_content", message.get("reasoning") orelse .null);
        };
        dst.* = .{ .object = copy };
    }
    return .{ .array = out };
}

pub fn split(a: std.mem.Allocator, text: []const u8) !@import("reply_text.zig").Split {
    return @import("reply_text.zig").splitThinkingAt(a, text, true, .{ .open = "", .close = "</think>", .strict = true }, .spelled);
}

pub fn stream(g: anytype, text: []const u8, finished: bool) !void {
    const sink = g.sink orelse return;
    const end = std.mem.indexOf(u8, text, "</think>") orelse if (finished) text.len else text.len -| 7;
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
