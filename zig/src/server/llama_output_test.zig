const std = @import("std");
const json = @import("json");
const pod = @import("nanbeige.zig");
const qwen = @import("tool_qwen.zig");
const api = @import("engine_api");
const server = @import("server.zig");
const openai = @import("openai.zig");
const Conn = @import("http_conn.zig").Conn;
const Allocator = std.mem.Allocator;
const Value = json.Value;

fn expectCalls(a: Allocator, expected: []const Value, calls: ?[]Value) !void {
    const got = calls orelse &.{};
    try std.testing.expectEqual(expected.len, got.len);
    for (expected, got) |e, g| {
        const f = g.get("function").?;
        try std.testing.expectEqualStrings(e.get("name").?.string, f.get("name").?.string);
        const args = (try json.parse(a, f.get("arguments").?.string)).ok;
        try std.testing.expect(json.equal(e.get("arguments").?, args));
    }
}

test "Llama serve: reasoning and XML calls equal the scored pod parser" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    for ((try json.parse(a, @embedFile("fixtures/llama/outputs.json"))).ok.array) |case| {
        const split = try pod.split(a, case.get("raw").?.string);
        const calls = try qwen.calls(a, split.answer, case.get("tools").?.array, null);
        const expected = case.get("expected").?;
        try std.testing.expectEqualStrings(expected.get("reasoning").?.string, split.reasoning);
        try std.testing.expectEqualStrings(expected.get("content").?.string, calls.content);
        try expectCalls(a, expected.get("calls").?.array, calls.calls);
        std.debug.print("OUTPUT {s} equal\n", .{case.get("name").?.string});
    }
}

const Script = struct {
    reply: []const u32,
    request: ?api.Request = null,
    pub fn engine(e: *@This()) api.Engine {
        return .{ .ctx = e, .vtable = &.{ .info = info, .submit = submit, .cancel = cancel, .status = status, .memory = memory } };
    }
    fn info(_: *anyopaque) api.Info {
        return .{ .context_window = 4096, .greedy_only = true };
    }
    fn submit(ctx: *anyopaque, id: api.Id, req: *const api.Request, sink: api.Sink) api.SubmitError!void {
        const e: *Script = @ptrCast(@alignCast(ctx));
        e.request = req.*;
        sink.event(sink.ctx, id, &.{ .prefilled = 0 });
        for (e.reply[0..@min(e.reply.len, req.max_tokens)]) |*token| sink.event(sink.ctx, id, &.{ .tokens = token[0..1] });
        sink.event(sink.ctx, id, &.{ .finished = .{ .reason = if (e.reply.len > req.max_tokens) .length else .stop } });
    }
    fn cancel(_: *anyopaque, _: api.Id) void {}
    fn status(_: *anyopaque, out: *api.Status, _: []u32) void {
        out.* = .{};
    }
    fn memory(_: *anyopaque, _: bool) ?api.Memory {
        return null;
    }
};

const Got = struct {
    a: Allocator,
    status: u16 = 0,
    body: Value = .null,
    events: std.ArrayList(Value) = .empty,
    fn out(g: *Got) openai.Out {
        return .{ .ctx = g, .vt = &.{ .open = open, .event = event, .reply = reply } };
    }
    fn open(ctx: *anyopaque, _: usize) error{Closed}!void {
        const g: *Got = @ptrCast(@alignCast(ctx));
        g.status = 200;
    }
    fn event(ctx: *anyopaque, value: ?Value) error{Closed}!void {
        const g: *Got = @ptrCast(@alignCast(ctx));
        if (value) |v| {
            const bytes = json.stringify(g.a, v, .{ .ascii = false }) catch return error.Closed;
            g.events.append(g.a, (json.parse(g.a, bytes) catch return error.Closed).ok) catch return error.Closed;
        }
    }
    fn reply(ctx: *anyopaque, code: u16, value: Value) void {
        const g: *Got = @ptrCast(@alignCast(ctx));
        g.status = code;
        g.body = value;
    }
};

fn run(srv: *server.Server, a: Allocator, body: Value, chat: bool) Got {
    var conn: Conn = .{ .fd = -1, .peer = "", .buf = &.{}, .gpa = a };
    var got: Got = .{ .a = a };
    openai.run(srv, a, got.out(), .{ .conn = &conn }, chat, body);
    return got;
}

test "Llama serve: actual JSON and SSE replies match pod output with real tokenizer" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const text = try @import("llama_prompt_test.zig").load(a);
    defer text.deinit();
    var script: Script = .{ .reply = &.{} };
    const srv = try server.Server.init(std.testing.allocator, std.testing.io, script.engine(), text.text(), .{
        .served_name = "nanbeige",
        .model_type = "llama",
        .model_ids = &.{"nanbeige"},
        .use_drafts = false,
    }, null);
    defer srv.deinit();
    for ((try json.parse(a, @embedFile("fixtures/llama/outputs.json"))).ok.array) |case| {
        const raw = case.get("raw").?.string;
        script.reply = try text.text().encode(a, raw, false);
        const body = (try json.parse(a, "{\"messages\":[{\"role\":\"user\",\"content\":\"x\"}],\"max_tokens\":512}")).ok;
        try body.object.put(a, "tools", case.get("tools").?);
        const expected = case.get("expected").?;
        const whole = run(srv, a, body, true);
        try std.testing.expectEqual(@as(u16, 200), whole.status);
        const message = whole.body.get("choices").?.array[0].get("message").?;
        try std.testing.expectEqualStrings(expected.get("content").?.string, message.strField("content") orelse "");
        try std.testing.expectEqualStrings(expected.get("reasoning").?.string, message.strField("reasoning_content") orelse "");
        try expectCalls(a, expected.get("calls").?.array, if (message.get("tool_calls")) |c| c.array else null);
        try body.object.put(a, "stream", .{ .bool = true });
        const streamed = run(srv, a, body, true);
        try std.testing.expectEqual(@as(u16, 200), streamed.status);
        var thought: std.ArrayList(u8) = .empty;
        var content: std.ArrayList(u8) = .empty;
        var calls: std.ArrayList(Value) = .empty;
        var saw_call = false;
        for (streamed.events.items) |event| {
            for (event.get("choices").?.array) |choice| {
                const delta = choice.get("delta").?;
                if (delta.strField("reasoning")) |r| try thought.appendSlice(a, r);
                if (delta.strField("content")) |c| {
                    if (c.len > 0) try std.testing.expect(!saw_call);
                    try content.appendSlice(a, c);
                }
                if (delta.get("tool_calls")) |c| {
                    saw_call = true;
                    try calls.appendSlice(a, c.array);
                }
            }
        }
        const end = std.mem.indexOf(u8, raw, "</think>") orelse raw.len;
        try std.testing.expectEqualStrings(raw[0..end], thought.items);
        try std.testing.expectEqualStrings(expected.get("content").?.string, content.items);
        try expectCalls(a, expected.get("calls").?.array, calls.items);
        std.debug.print("ROUTE {s} JSON+SSE equal\n", .{case.get("name").?.string});
    }
}

test "Llama serve: every split of the thinking closer streams without leaking markup" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const Capture = struct {
        a: Allocator,
        sink: ?@import("chat.zig").Sink,
        streamed_reasoning: @import("reply_text.zig").Shown = .{},
        out: std.ArrayList(u8) = .empty,
        fn call(ctx: *anyopaque, v: Value) error{Closed}!void {
            const c: *@This() = @ptrCast(@alignCast(ctx));
            c.out.appendSlice(c.a, v.strField("reasoning_content").?) catch return error.Closed;
        }
    };
    const raw = "Plan.\n</think>\nAnswer";
    for (1..raw.len) |cut| {
        var c: Capture = .{ .a = a, .sink = null };
        c.sink = .{ .ctx = &c, .call = Capture.call };
        try pod.stream(&c, raw[0..cut], false);
        try pod.stream(&c, raw, true);
        try std.testing.expectEqualStrings("Plan.\n", c.out.items);
    }
}

test "Llama serve: unsupported sampling refuses before SSE and completions add BOS once" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const text = try @import("llama_prompt_test.zig").load(a);
    defer text.deinit();
    var script: Script = .{ .reply = try text.text().encode(a, "Plan.</think>answerSTOPtail", false) };
    const srv = try server.Server.init(std.testing.allocator, std.testing.io, script.engine(), text.text(), .{
        .served_name = "nanbeige",
        .model_type = "llama",
        .model_ids = &.{"nanbeige"},
        .use_drafts = false,
    }, null);
    defer srv.deinit();
    for ([_][]const u8{ "\"temperature\":1", "\"presence_penalty\":1", "\"frequency_penalty\":1", "\"repetition_penalty\":1.1", "\"top_p\":2", "\"chat_template_kwargs\":[]" }) |field| {
        const body = (try json.parse(a, try std.fmt.allocPrint(a, "{{\"messages\":[{{\"role\":\"user\",\"content\":\"x\"}}],\"stream\":true,{s}}}", .{field}))).ok;
        const got = run(srv, a, body, true);
        try std.testing.expectEqual(@as(u16, 400), got.status);
        try std.testing.expectEqual(@as(usize, 0), got.events.items.len);
    }
    const body = (try json.parse(a, "{\"messages\":[{\"role\":\"user\",\"content\":\"x\"}],\"temperature\":0,\"top_p\":0.95,\"top_k\":20,\"presence_penalty\":0,\"repetition_penalty\":1,\"max_tokens\":128,\"stop\":\"STOP\"}")).ok;
    const got = run(srv, a, body, true);
    try std.testing.expectEqual(@as(u16, 200), got.status);
    try std.testing.expectEqual(@as(u32, 128), script.request.?.max_tokens);
    try std.testing.expect(script.request.?.sampling == null);
    const message = got.body.get("choices").?.array[0].get("message").?;
    try std.testing.expectEqualStrings("answer", message.get("content").?.string);
    const end_text = try text.text().encode(a, "Plan.</think>answer", false);
    script.reply = try std.mem.concat(a, u32, &.{ end_text, &.{166102} });
    const ended = run(srv, a, body, true);
    try std.testing.expectEqualStrings("answer", ended.body.get("choices").?.array[0].get("message").?.get("content").?.string);
    try std.testing.expectEqualStrings("stop", ended.body.get("choices").?.array[0].get("finish_reason").?.string);
    try body.object.put(a, "max_completion_tokens", .{ .int = "7" });
    _ = run(srv, a, body, true);
    try std.testing.expectEqual(@as(u32, 7), script.request.?.max_tokens);
    for ([_][]const u8{ "hello", "<|im_start|>user\nhello" }) |input| {
        const completion = try json.newObject(a);
        try completion.put(a, "prompt", .{ .string = input });
        try completion.put(a, "max_tokens", .{ .int = "2" });
        const reply = run(srv, a, .{ .object = completion }, false);
        try std.testing.expectEqual(@as(u16, 200), reply.status);
        const ids = script.request.?.prompt;
        try std.testing.expectEqual(@as(u32, 166100), ids[0]);
        try std.testing.expect(ids[1] != 166100);
    }
}
