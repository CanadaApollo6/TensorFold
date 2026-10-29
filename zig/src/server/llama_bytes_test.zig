const std = @import("std");
const json = @import("json");
const Allocator = std.mem.Allocator;
const Value = json.Value;
const output = @import("nanbeige_output.zig");
const support = @import("llama_output_test.zig");
const Conn = @import("http_conn.zig").Conn;

pub fn normalize(a: Allocator, text: []const u8, value: Value) error{ OutOfMemory, TestUnexpectedResult }![]const u8 {
    var out = text;
    switch (value) {
        .array => |items| for (items) |v| {
            out = try normalize(a, out, v);
        },
        .object => |o| for (o.keys(), o.values()) |key, v| {
            if (std.mem.eql(u8, key, "id") and v == .string and std.mem.startsWith(u8, v.string, "chatcmpl-")) {
                const tool = std.mem.startsWith(u8, v.string, "chatcmpl-tool-");
                const prefix: []const u8 = if (tool) "chatcmpl-tool-" else "chatcmpl-";
                const n: usize = if (tool) 16 else 32;
                if (v.string.len != prefix.len + n) return error.TestUnexpectedResult;
                if (v.string[prefix.len + 12] != '4') return error.TestUnexpectedResult;
                if (!tool and std.mem.indexOfScalar(u8, "89ab", v.string[prefix.len + 16]) == null) return error.TestUnexpectedResult;
                for (v.string[prefix.len..]) |c| if (!std.ascii.isDigit(c) and (c < 'a' or c > 'f')) return error.TestUnexpectedResult;
                const id = try a.alloc(u8, v.string.len);
                @memcpy(id[0..prefix.len], prefix);
                @memset(id[prefix.len..], '0');
                out = try std.mem.replaceOwned(u8, a, out, v.string, id);
            } else if (std.mem.eql(u8, key, "created")) {
                for ([_][]const u8{ "", " " }) |space| {
                    const from = try std.fmt.allocPrint(a, "\"created\":{s}{s}", .{ space, v.int });
                    out = try std.mem.replaceOwned(u8, a, out, from, try std.mem.concat(a, u8, &.{ "\"created\":", space, "0" }));
                }
            } else out = try normalize(a, out, v);
        },
        else => {},
    }
    return out;
}

fn parserFixture(data: []const u8) !void {
    var fixture_arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer fixture_arena.deinit();
    const fixture = (try json.parse(fixture_arena.allocator(), data)).ok;
    for (fixture.get("cases").?.array) |case| {
        var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
        defer arena.deinit();
        const a = arena.allocator();
        const split = try output.split(a, case.get("raw").?.string);
        const calls = try output.calls(a, split.answer, if (case.get("tools").?.bool) fixture.get("tools").?.array else &.{});
        const got = try json.newObject(a);
        try got.put(a, "reasoning", .{ .string = split.reasoning });
        try got.put(a, "content", .{ .string = calls.content });
        try got.put(a, "calls", .{ .array = calls.calls orelse &.{} });
        const value: Value = .{ .object = got };
        errdefer std.debug.print("BYTE parser case {s}\n", .{case.strField("name").?});
        try std.testing.expectEqualStrings(case.strField("expected").?, try normalize(a, try json.stringify(a, value, .{ .ascii = false }), value));
    }
    std.debug.print("BYTE parser {d} cases equal\n", .{fixture.get("cases").?.array.len});
}

test "Llama serve: grammar compares exact pod argument bytes" {
    try parserFixture(@embedFile("fixtures/llama/outputs_grammar.json"));
}

test "Llama serve: typed schemas compare exact pod argument bytes" {
    try parserFixture(@embedFile("fixtures/llama/outputs_typed.json"));
}

test "Llama serve: Python named escapes, strict hex, and source NUL match pod bytes" {
    try parserFixture(@embedFile("fixtures/llama/outputs_escapes.json"));
}

const Wire = struct {
    a: Allocator,
    text: std.ArrayList(u8) = .empty,

    fn write(ctx: *anyopaque, bytes: []const u8) error{Closed}!void {
        const w: *Wire = @ptrCast(@alignCast(ctx));
        w.text.appendSlice(w.a, bytes) catch return error.Closed;
    }
};

pub fn wire(srv: *@import("server.zig").Server, a: Allocator, body: Value, status: u16) ![]const u8 {
    return wireAt(srv, a, body, status, true);
}

pub fn wireAt(srv: *@import("server.zig").Server, a: Allocator, body: Value, status: u16, chat: bool) ![]const u8 {
    var w: Wire = .{ .a = a };
    var conn: Conn = .{
        .fd = -1,
        .peer = "",
        .buf = &.{},
        .gpa = a,
        .version = "HTTP/1.1",
        .method = "POST",
        .write_hooks = .{ .ctx = &w, .write = Wire.write },
    };
    var client: @import("openai_client.zig").ClientOut = .{ .conn = &conn, .a = a, .pod = true };
    @import("openai.zig").run(srv, a, client.out(), .{ .conn = &conn }, chat, body);
    try std.testing.expect(std.mem.startsWith(u8, w.text.items, try std.fmt.allocPrint(a, "HTTP/1.1 {d} ", .{status})));
    const start = std.mem.indexOf(u8, w.text.items, "\r\n\r\n") orelse return error.TestUnexpectedResult;
    const raw = w.text.items[start + 4 ..];
    const data = if (std.mem.indexOf(u8, w.text.items[0..start], "Transfer-Encoding: chunked") != null) (try decodeWire(a, raw)).body else raw;
    if (!std.mem.startsWith(u8, data, "data: ")) return normalize(a, data, (try json.parse(a, data)).ok);
    var out: std.ArrayList(u8) = .empty;
    var frames = std.mem.splitSequence(u8, data, "\n\n");
    while (frames.next()) |frame| {
        if (frame.len == 0) continue;
        const text = if (std.mem.eql(u8, frame, "data: [DONE]")) frame else try normalize(a, frame, (try json.parse(a, frame[6..])).ok);
        try out.appendSlice(a, text);
        try out.appendSlice(a, "\n\n");
    }
    return out.items;
}

test "Llama serve: real HTTP writer matches complete pod JSON and SSE bytes" {
    var outer = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer outer.deinit();
    const b = outer.allocator();
    const text = try @import("llama_prompt_test.zig").load(b);
    defer text.deinit();
    var script: support.Script = .{ .reply = &.{} };
    const srv = try @import("server.zig").Server.init(std.testing.allocator, std.testing.io, script.engine(), text.text(), .{
        .served_name = "nanbeige",
        .model_type = "llama",
        .model_ids = &.{"nanbeige"},
        .use_drafts = false,
    }, null);
    defer srv.deinit();
    const cases = (try json.parse(b, @embedFile("fixtures/llama/responses.json"))).ok.array;
    for (1..9) |block| for (0..block) |offset| for (cases) |case| {
        script.block = block;
        script.offset = offset;
        var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
        defer arena.deinit();
        const a = arena.allocator();
        script.reply = try text.text().encode(a, case.strField("raw").?, false);
        script.request = null;
        const body = try json.deepCopy(a, case.get("request").?);
        errdefer std.debug.print("BYTE response case {s}\n", .{case.strField("name").?});
        try std.testing.expectEqualStrings(case.strField("json_bytes").?, try wire(srv, a, body, @intCast(case.get("status").?.int64().?)));
        try body.object.put(a, "stream", .{ .bool = true });
        try std.testing.expectEqualStrings(case.strField("sse_bytes").?, try wire(srv, a, body, @intCast(case.get("status").?.int64().?)));
        const params = case.get("request_params").?;
        if (params == .null) {
            try std.testing.expect(script.request == null);
            continue;
        }
        try std.testing.expectEqual(@as(u32, @intCast(params.get("max_tokens").?.int64().?)), script.request.?.max_tokens);
        try std.testing.expectEqual(@as(f32, @floatCast(params.get("sampler").?.get("temperature").?.float64().?)), script.request.?.sampling.?.temperature);
        try std.testing.expect(!script.request.?.penalties.active());
        try std.testing.expect(script.request.?.structure == null);
    };
    std.debug.print("BYTE HTTP {d} JSON + {d} SSE cases equal\n", .{ cases.len, cases.len });
}

test "Llama serve: prepare equals the pod before template rendering" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    for ((try json.parse(a, @embedFile("fixtures/llama/prepared.json"))).ok.array) |case| {
        var cx: @import("errors.zig").Cx = .{ .a = a };
        const got = (try @import("nanbeige.zig").prepare(&cx, case.get("messages"))).?;
        try std.testing.expectEqualStrings(case.strField("expected").?, try json.stringify(a, got, .{ .ascii = false }));
    }
}

pub const DecodedWire = struct { body: []const u8, complete: bool };

pub fn decodeWire(a: Allocator, data: []const u8) !DecodedWire {
    var out: std.ArrayList(u8) = .empty;
    var at: usize = 0;
    while (at < data.len) {
        const end = std.mem.indexOfPos(u8, data, at, "\r\n") orelse return error.TestUnexpectedResult;
        const n = try std.fmt.parseInt(usize, data[at..end], 16);
        at = end + 2;
        if (n == 0) {
            try std.testing.expectEqualStrings("\r\n", data[at..]);
            return .{ .body = out.items, .complete = true };
        }
        if (at + n + 2 > data.len) return error.TestUnexpectedResult;
        try out.appendSlice(a, data[at .. at + n]);
        try std.testing.expectEqualStrings("\r\n", data[at + n .. at + n + 2]);
        at += n + 2;
    }
    return .{ .body = out.items, .complete = false };
}
