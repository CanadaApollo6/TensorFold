const std = @import("std");
const json = @import("json");
const Cx = @import("errors.zig").Cx;
const Value = json.Value;

pub fn input(cx: *Cx, body: Value, fallback: []const u8) @import("errors.zig").Refused!@import("chat.zig").Input {
    try @import("fields.zig").requireObject(cx, body);
    const prepared = @import("nanbeige.zig").prepare(cx, body.get("messages") orelse .{ .array = &.{} }) catch |e| {
        if (std.mem.eql(u8, cx.message, "Nanbeige is not a multimodal model"))
            cx.message = try std.fmt.allocPrint(cx.a, "{s} is not a multimodal model", .{model(body, fallback)});
        return e;
    };
    const tools = body.get("tools") orelse Value.null;
    const sampling = try @import("nanbeige.zig").defaults(cx.a, null);
    for ([_][]const u8{ "temperature", "top_p", "top_k", "min_p", "seed", "chat_template_kwargs" }) |key| {
        if (body.field(key)) |v| if (!std.mem.eql(u8, key, "chat_template_kwargs") or v.truthy())
            try sampling.object.put(cx.a, key, v);
    }
    const numeric = try @import("fields.zig").parseNumbers(cx, sampling);
    if (numeric.field("chat_template_kwargs")) |kw| if (kw != .object) return cx.refuse("chat_template_kwargs must be an object");
    const max = body.firstTruthy(&.{ "max_completion_tokens", "max_tokens" });
    return .{
        .messages = prepared orelse .{ .array = &.{} },
        .tools = if (tools == .array) tools.array else &.{},
        .max_tokens = if (max) |v| v.int64() orelse return cx.refuse("max_tokens must be an integer") else null,
        .fields = numeric,
    };
}

pub fn model(body: Value, fallback: []const u8) []const u8 {
    const name = body.strField("model") orelse return fallback;
    return if (name.len > 0) name else fallback;
}

pub fn errorBody(a: std.mem.Allocator, status: u16, message: []const u8) !Value {
    const o = try json.newObject(a);
    try o.put(a, "message", .{ .string = message });
    try o.put(a, "type", .{ .string = "BadRequestError" });
    try o.put(a, "param", .null);
    try o.put(a, "code", try json.intValue(a, status));
    return @import("openai.zig").wrapError(a, .{ .object = o });
}

pub fn limit(cx: *Cx, wanted: ?i64, room: i64, prompt: i64, window: i64) @import("errors.zig").Refused!i64 {
    const want = wanted orelse room;
    if (room <= 0 or want > room) return cx.fail(.context_length, "This model's maximum context length is {d} tokens. However, you requested {d} tokens ({d} in the messages, {d} in the completion).", .{ window, prompt + want, prompt, want });
    return @max(1, want);
}

pub fn templateFailure(a: std.mem.Allocator, body: Value) std.mem.Allocator.Error!bool {
    if (body.get("tools")) |tools| {
        if (tools.truthy() and tools != .array) return true;
        if (tools == .array) for (tools.array) |tool| {
            if (tool != .object) return true;
        };
    }
    const kwargs = body.get("chat_template_kwargs") orelse Value.null;
    if (kwargs.strField("tool_call_format")) |f| if (std.mem.eql(u8, f, "json")) return false;
    const messages = body.get("messages") orelse return false;
    if (messages != .array) return false;
    for (messages.array) |message| {
        const role = message.strField("role") orelse continue;
        if (!std.mem.eql(u8, role, "assistant")) continue;
        const calls = message.get("tool_calls") orelse continue;
        if (calls != .array) continue;
        for (calls.array) |call| {
            const f = call.get("function") orelse call;
            var args = f.get("arguments") orelse continue;
            if (args == .string) {
                const parsed = try json.parseText(a, args.string);
                if (parsed == .err) continue;
                args = parsed.ok;
            }
            if (args != .object) return true;
        }
    }
    return false;
}
