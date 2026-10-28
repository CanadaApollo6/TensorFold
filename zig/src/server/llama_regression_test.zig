const std = @import("std");
const json = @import("json");
const server = @import("server.zig");
const support = @import("llama_output_test.zig");
const HfText = @import("hf_text.zig").HfText;

test "Llama serve: resolved template controls retain upstream precedence outside Nanbeige" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    try tmp.dir.writeFile(std.testing.io, .{ .sub_path = "tokenizer.json", .data =
        \\{"model":{"type":"BPE","vocab":{"H":0},"merges":[]},"added_tokens":[]}
    });
    try tmp.dir.writeFile(std.testing.io, .{ .sub_path = "chat_template.jinja", .data =
        \\{{ reasoning_effort }}|{{ enable_thinking }}|{{ clear_thinking }}|{{ custom }}
    });
    const dir = try std.fmt.allocPrint(a, ".zig-cache/tmp/{s}", .{tmp.sub_path});
    var problem: []const u8 = "";
    const text = try HfText.load(std.testing.allocator, std.testing.io, dir, null, a, &problem);
    defer text.deinit();
    text.clear_thinking = false;
    const kwargs = (try json.parse(a, "{\"reasoning_effort\":\"low\",\"enable_thinking\":false,\"clear_thinking\":true,\"custom\":\"kept\"}")).ok;
    const rendered = try text.text().render(a, .{ .array = &.{} }, .{
        .reasoning_effort = "high", .enable_thinking = true, .clear_thinking = false, .template_kwargs = kwargs,
    }, &problem);
    try std.testing.expectEqualStrings("high|True|False|kept", rendered.text);
    text.nanbeige = true;
    const raw = try text.text().render(a, .{ .array = &.{} }, .{
        .reasoning_effort = "high", .enable_thinking = true, .clear_thinking = false, .template_kwargs = kwargs,
    }, &problem);
    try std.testing.expectEqualStrings("low|False|True|kept", raw.text);
}

test "Llama serve: context overflow is HTTP 400 before SSE with and without compaction" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const text = try @import("llama_prompt_test.zig").load(a);
    defer text.deinit();
    var script: support.Script = .{ .reply = &.{} };
    const srv = try server.Server.init(std.testing.allocator, std.testing.io, script.engine(), text.text(), .{
        .served_name = "nanbeige", .model_type = "llama", .model_ids = &.{"nanbeige"}, .use_drafts = false,
    }, null);
    defer srv.deinit();
    for ([_]?server.CompactAt{ null, .{ .fraction = 0.9 } }) |compact| {
        srv.config.compact_at = compact;
        for ([_]bool{ true, false }) |chat| {
            const body = (try json.parse(a, if (chat)
                "{\"messages\":[{\"role\":\"user\",\"content\":\"x\"}],\"max_tokens\":4096,\"stream\":true}"
            else
                "{\"prompt\":\"x\",\"max_tokens\":4096,\"stream\":true}")).ok;
            const got = support.run(srv, a, body, chat);
            try std.testing.expectEqual(@as(u16, 400), got.status);
            try std.testing.expectEqual(@as(usize, 0), got.events.items.len);
            try std.testing.expectEqualStrings("context_length_exceeded", got.body.get("error").?.strField("code").?);
            try std.testing.expect(script.request == null);
        }
    }
}

test "Llama serve: Unicode stops end generation across more tokens than characters" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const text = try @import("llama_prompt_test.zig").load(a);
    defer text.deinit();
    const stop = "💡💡💡💡💡💡💡💡💡💡";
    const trailing: [1000]u8 = @splat('x');
    const ids = try text.text().encode(a, try std.mem.concat(a, u8, &.{ "prefix", stop, &trailing }), false);
    const stop_ids = try text.text().encode(a, stop, false);
    try std.testing.expect(stop_ids.len > 18);
    var script: support.Script = .{ .reply = ids };
    const srv = try server.Server.init(std.testing.allocator, std.testing.io, script.engine(), text.text(), .{
        .served_name = "nanbeige", .model_type = "llama", .model_ids = &.{"nanbeige"}, .use_drafts = false,
    }, null);
    defer srv.deinit();
    var hook: @import("stop_hook.zig").StopHook = .{ .srv = srv, .stops = .{ .strings = &.{stop} } };
    srv.nanbeige = false;
    for ([_]bool{ false, true }) |stream| {
        const body = (try json.parse(a, "{\"prompt\":\"x\",\"temperature\":0,\"max_tokens\":60}")).ok;
        try body.object.put(a, "stop", .{ .string = stop });
        try body.object.put(a, "stream", .{ .bool = stream });
        const got = support.run(srv, a, body, false);
        try std.testing.expectEqual(@as(u16, 200), got.status);
        const choice = if (stream) got.events.items[got.events.items.len - 1].get("choices").?.array[0]
            else got.body.get("choices").?.array[0];
        try std.testing.expectEqualStrings("stop", choice.strField("finish_reason").?);
        if (!stream) {
            try std.testing.expectEqualStrings("prefix", choice.strField("text").?);
            try std.testing.expect(got.body.get("usage").?.get("completion_tokens").?.int64().? < 60);
        }
    }
    try std.testing.expect(@import("stop_hook.zig").StopHook.check(&hook, stop_ids));
}
