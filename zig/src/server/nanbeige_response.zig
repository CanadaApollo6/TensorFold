const std = @import("std");
const json = @import("json");
const Allocator = std.mem.Allocator;
const Value = json.Value;

pub const Envelope = struct {
    a: Allocator,
    id: []const u8,
    created: i64,
    model: []const u8,

    fn base(e: Envelope, streaming: bool) !*json.Object {
        const o = try json.newObject(e.a);
        try o.put(e.a, "id", .{ .string = e.id });
        try o.put(e.a, "object", .{ .string = if (streaming) "chat.completion.chunk" else "chat.completion" });
        try o.put(e.a, "created", try json.intValue(e.a, e.created));
        try o.put(e.a, "model", .{ .string = e.model });
        return o;
    }

    pub fn chunk(e: Envelope, delta: ?Value, reason: ?[]const u8) !Value {
        const o = try e.base(true);
        const choice = try json.newObject(e.a);
        try choice.put(e.a, "index", .{ .int = "0" });
        const d = delta orelse Value.null;
        try choice.put(e.a, "delta", if (d == .object) d else if (d == .string and d.string.len > 0)
            try @import("chat.zig").deltaOf(e.a, "content", d.string)
        else
            .{ .object = try json.newObject(e.a) });
        try choice.put(e.a, "logprobs", .null);
        try choice.put(e.a, "finish_reason", if (reason) |f| .{ .string = f } else .null);
        const list = try e.a.alloc(Value, 1);
        list[0] = .{ .object = choice };
        try o.put(e.a, "choices", .{ .array = list });
        return .{ .object = o };
    }

    pub fn whole(e: Envelope, reply: anytype, calls: ?[]Value) !Value {
        const o = try e.base(false);
        const message = try json.newObject(e.a);
        try message.put(e.a, "role", .{ .string = "assistant" });
        try message.put(e.a, "content", optional(reply.content));
        try message.put(e.a, "reasoning", optional(reply.reasoning orelse ""));
        try message.put(e.a, "reasoning_content", optional(reply.reasoning orelse ""));
        try message.put(e.a, "tool_calls", .{ .array = calls orelse &.{} });
        const choice = try json.newObject(e.a);
        try choice.put(e.a, "index", .{ .int = "0" });
        try choice.put(e.a, "message", .{ .object = message });
        try choice.put(e.a, "logprobs", .null);
        try choice.put(e.a, "finish_reason", .{ .string = reply.finish_reason });
        const list = try e.a.alloc(Value, 1);
        list[0] = .{ .object = choice };
        try o.put(e.a, "choices", .{ .array = list });
        try o.put(e.a, "usage", try usage(e.a, reply));
        return .{ .object = o };
    }

    pub fn finish(e: Envelope, out: anytype, reply: anytype, calls: ?[]Value, include_usage: bool) !void {
        if (reply.content.len > 0) try out.vt.event(out.ctx, try e.chunk(.{ .string = reply.content }, null));
        for (calls orelse &.{}, 0..) |call, i| {
            const c = try json.newObject(e.a);
            try c.put(e.a, "index", try json.intValue(e.a, i));
            for (call.object.keys(), call.object.values()) |key, value| try c.put(e.a, key, value);
            const list = try e.a.alloc(Value, 1);
            list[0] = .{ .object = c };
            const delta = try json.newObject(e.a);
            try delta.put(e.a, "tool_calls", .{ .array = list });
            try out.vt.event(out.ctx, try e.chunk(.{ .object = delta }, null));
        }
        try out.vt.event(out.ctx, try e.chunk(null, reply.finish_reason));
        if (include_usage) {
            const u = try e.chunk(null, null);
            try u.object.put(e.a, "choices", .{ .array = &.{} });
            try u.object.put(e.a, "usage", try usage(e.a, reply));
            try out.vt.event(out.ctx, u);
        }
        try out.vt.event(out.ctx, null);
    }
};

fn optional(text: []const u8) Value {
    return if (text.len > 0) .{ .string = text } else .null;
}

pub fn role(a: Allocator) !Value {
    const d = try json.newObject(a);
    try d.put(a, "role", .{ .string = "assistant" });
    try d.put(a, "content", .{ .string = "" });
    return .{ .object = d };
}

pub fn usage(a: Allocator, reply: anytype) !Value {
    const o = try json.newObject(a);
    try o.put(a, "prompt_tokens", try json.intValue(a, reply.prompt_tokens));
    try o.put(a, "completion_tokens", try json.intValue(a, reply.completion_tokens));
    try o.put(a, "total_tokens", try json.intValue(a, reply.prompt_tokens + reply.completion_tokens));
    return .{ .object = o };
}

pub fn bytes(a: Allocator, payload: Value, streaming: bool) ![]const u8 {
    const text = try json.stringify(a, payload, .{ .ascii = streaming and payload.has("error"), .compact = !streaming });
    return if (streaming) std.mem.concat(a, u8, &.{ "data: ", text, "\n\n" }) else text;
}

pub fn id(a: Allocator, tool: bool) ![]const u8 {
    const prefix: []const u8 = if (tool) "chatcmpl-tool-" else "chatcmpl-";
    const text = @constCast(try @import("ids.zig").make(a, prefix, if (tool) 16 else 32));
    text[prefix.len + 12] = '4';
    if (!tool) {
        const nibble = std.fmt.charToDigit(text[prefix.len + 16], 16) catch unreachable;
        text[prefix.len + 16] = "89ab"[nibble & 3];
    }
    return text;
}

pub fn failed(a: Allocator, out: anytype, cx: anytype, failure: anyerror) void {
    const message = if (failure == error.Refused) cx.message else if (failure == error.OutOfMemory) "out of memory" else "the reply failed";
    const e = json.newObject(a) catch return;
    e.put(a, "message", .{ .string = message }) catch return;
    e.put(a, "type", .{ .string = "InternalServerError" }) catch return;
    const payload = json.newObject(a) catch return;
    payload.put(a, "error", .{ .object = e }) catch return;
    out.vt.event(out.ctx, .{ .object = payload }) catch {};
}
