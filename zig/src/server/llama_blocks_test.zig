const std = @import("std");
const json = @import("json");
const support = @import("llama_output_test.zig");
const bytes = @import("llama_bytes_test.zig");

pub fn canonical(a: std.mem.Allocator, data: []const u8, chat: bool) ![]const u8 {
    if (chat) return data;
    var out: std.ArrayList(u8) = .empty;
    var frames = std.mem.splitSequence(u8, data, "\n\n");
    while (frames.next()) |frame| {
        if (frame.len == 0) continue;
        if (std.mem.eql(u8, frame, "data: [DONE]")) {
            try out.appendSlice(a, "data: [DONE]\n\n");
            continue;
        }
        const streaming = std.mem.startsWith(u8, frame, "data: ");
        const value = (try json.parse(a, if (streaming) frame[6..] else frame)).ok;
        const fields = try json.newObject(a);
        for (value.object.keys(), value.object.values()) |key, item| {
            if (std.mem.eql(u8, key, "id") or std.mem.eql(u8, key, "tensorfold") or std.mem.eql(u8, key, "speculative")) continue;
            try fields.put(a, key, item);
        }
        if (streaming) try out.appendSlice(a, "data: ");
        try out.appendSlice(a, try json.stringify(a, .{ .object = fields }, .{}));
        if (streaming) try out.appendSlice(a, "\n\n");
    }
    return out.items;
}

fn wire(srv: *@import("server.zig").Server, a: std.mem.Allocator, body: json.Value, chat: bool) ![]const u8 {
    return canonical(a, try bytes.wireAt(srv, a, body, 200, chat), chat);
}

test "Llama serve: stop, EOS and length at every accepted block offset preserve bytes and usage" {
    var outer = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer outer.deinit();
    const b = outer.allocator();
    const text = try @import("llama_prompt_test.zig").load(b);
    defer text.deinit();
    const aa = try text.text().encode(b, "a", false);
    const xx = try text.text().encode(b, "X", false);
    try std.testing.expectEqual(@as(usize, 1), aa.len);
    try std.testing.expectEqual(@as(usize, 1), xx.len);
    var script: support.Script = .{ .reply = &.{}, .clip = false };
    const srv = try @import("server.zig").Server.init(std.testing.allocator, std.testing.io, script.engine(), text.text(), .{
        .served_name = "nanbeige",
        .model_type = "llama",
        .model_ids = &.{"nanbeige"},
        .use_drafts = false,
    }, null);
    defer srv.deinit();
    for (0..3) |kind| for (1..9) |cut| {
        var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
        defer arena.deinit();
        const a = arena.allocator();
        var ids: [24]u32 = @splat(aa[0]);
        if (kind != 2) ids[cut - 1] = if (kind == 0) xx[0] else srv.eos[0];
        script.reply = &ids;
        const body = (try json.parse(a, "{\"messages\":[{\"role\":\"user\",\"content\":\"x\"}],\"max_tokens\":24,\"stream_options\":{\"include_usage\":true}}")).ok;
        if (kind == 0) {
            try body.object.put(a, "stop", .{ .string = "X" });
            try body.object.put(a, "prompt", .{ .string = "x" });
        }
        if (kind == 2) try body.object.put(a, "max_tokens", try json.intValue(a, cut));
        script.block = 1;
        script.offset = 0;
        const expected = try wire(srv, a, body, kind != 0);
        const whole = (try json.parse(a, expected)).ok;
        try std.testing.expectEqualStrings(if (kind == 2) "length" else "stop", whole.get("choices").?.array[0].strField("finish_reason").?);
        try std.testing.expectEqual(@as(i64, @intCast(cut)), whole.get("usage").?.get("completion_tokens").?.int64().?);
        try body.object.put(a, "stream", .{ .bool = true });
        const sse = try wire(srv, a, body, kind != 0);
        try std.testing.expect(std.mem.endsWith(u8, sse, "data: [DONE]\n\n"));
        for (1..9) |block| for (0..block) |offset| {
            script.block = block;
            script.offset = offset;
            try body.object.put(a, "stream", .{ .bool = false });
            try std.testing.expectEqualStrings(expected, try wire(srv, a, body, kind != 0));
            try body.object.put(a, "stream", .{ .bool = true });
            try std.testing.expectEqualStrings(sse, try wire(srv, a, body, kind != 0));
        };
    };
}

test "Llama serve: GPU wire harness runs exact stop cases with CPU forwards" {
    const lanes = @import("lanes");
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const text = try @import("llama_prompt_test.zig").load(a);
    defer text.deinit();
    var fake: lanes.fake.Fake = .{ .gpa = a };
    defer fake.deinit();
    var vt = fake.backend().vtable.*;
    vt.tree = struct {
        fn tree(_: *anyopaque, _: *lanes.Stream, gpa: std.mem.Allocator) !?lanes.stream.Held {
            return .{ .count = 7, .tokens = try gpa.dupe(u32, &.{ 1, 1, 1, 1, 1, 1, 1 }) };
        }
    }.tree;
    var inner = fake.backend();
    inner.vtable = &vt;
    try @import("llama_gpu_wire.zig").checkCases(a, std.testing.io, text.text(), inner, .{ .exact_width = 8, .copy_drafts = false, .gpu_tokens = false, .mtp = true, .speculate = true, .drafts = 7, .batch_rows = 8, .max_streams = 1 });
}
