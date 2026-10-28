const std = @import("std");
const json = @import("json");
const HfText = @import("hf_text.zig").HfText;
const server = @import("server.zig");
const api = @import("engine_api");
const prompt = @import("prompt.zig");
const openai = @import("openai.zig");
const Cx = @import("errors.zig").Cx;

pub fn load(a: std.mem.Allocator) !*HfText {
    const dir = std.c.getenv("TF_LLAMA_MODEL") orelse return error.SkipZigTest;
    var problem: []const u8 = "";
    return HfText.load(std.testing.allocator, std.testing.io, std.mem.span(dir), null, a, &problem) catch |e| {
        std.debug.print("HfText: {s}\n", .{problem});
        return e;
    };
}

pub const Empty = struct {
    pub fn engine(e: *@This()) api.Engine {
        return .{ .ctx = e, .vtable = &.{ .info = info, .submit = submit, .cancel = cancel, .status = status, .memory = memory } };
    }
    fn info(_: *anyopaque) api.Info {
        return .{ .context_window = 4096 };
    }
    fn submit(_: *anyopaque, _: api.Id, _: *const api.Request, _: api.Sink) api.SubmitError!void {
        return error.Busy;
    }
    fn cancel(_: *anyopaque, _: api.Id) void {}
    fn status(_: *anyopaque, out: *api.Status, _: []u32) void {
        out.* = .{};
    }
    fn memory(_: *anyopaque, _: bool) ?api.Memory {
        return null;
    }
};

fn checkPrompts(prefix: []const u8) !void {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const t = try load(a);
    defer t.deinit();
    var engine: Empty = .{};
    const srv = try server.Server.init(std.testing.allocator, std.testing.io, engine.engine(), t.text(), .{
        .served_name = "nanbeige",
        .model_type = "llama",
        .model_ids = &.{"nanbeige"},
        .use_drafts = false,
    }, null);
    defer srv.deinit();
    try std.testing.expectEqualSlices(u32, &.{ 166101, 166102 }, t.text().eosIds());
    const cases = (try json.parse(a, @embedFile("fixtures/llama/prompts.json"))).ok.array;
    var failures: usize = 0;
    for (cases) |case| {
        if (!std.mem.startsWith(u8, case.get("name").?.string, prefix)) continue;
        var cx: Cx = .{ .a = a };
        const input = try openai.chatInput(srv, &cx, case.get("request").?);
        cx.template_kwargs = input.fields.field("chat_template_kwargs");
        const thinking = if (input.fields.get("enable_thinking")) |v| v.bool else true;
        const got = prompt.prepare(srv, &cx, input, thinking, null) catch |e| {
            std.debug.print("case {s}: {s}\n", .{ case.get("name").?.string, cx.message });
            return e;
        };
        const expected = case.get("ids").?.array;
        var equal = expected.len == got.ids.len;
        for (expected[0..@min(expected.len, got.ids.len)], got.ids[0..@min(expected.len, got.ids.len)]) |id, token| {
            equal = equal and id.int64().? == token;
        }
        if (!equal) {
            std.debug.print("prompt {s}: HF={d} TF={d}\n{s}\n", .{ case.get("name").?.string, expected.len, got.ids.len, try t.text().decode(a, got.ids) });
            failures += 1;
            continue;
        }
        try std.testing.expectEqual(@as(u32, 166100), got.ids[0]);
        try std.testing.expect(got.ids[1] != 166100);
        std.debug.print("PROMPT {s} tokens={d} equal\n", .{ case.get("name").?.string, got.ids.len });
    }
    try std.testing.expectEqual(@as(usize, 0), failures);
}

test "Llama serve: production prompt tokens equal HF for every conversation" {
    try checkPrompts("");
}

test "Llama serve: literal special-token text matches HF" {
    try checkPrompts("special-");
}

test "Llama serve: developer roles match HF through both normalizations" {
    try checkPrompts("developer-");
}

test "Llama serve: raw template kwargs match the pod" {
    try checkPrompts("kwargs-");
}
