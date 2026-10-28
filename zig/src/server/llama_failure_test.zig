const std = @import("std");
const json = @import("json");
const Allocator = std.mem.Allocator;
const Value = json.Value;
const bytes_test = @import("llama_bytes_test.zig");
const Conn = @import("http_conn.zig").Conn;

const Wire = struct {
    a: Allocator,
    text: std.ArrayList(u8) = .empty,
    fn write(ctx: *anyopaque, bytes: []const u8) error{Closed}!void {
        const w: *Wire = @ptrCast(@alignCast(ctx));
        w.text.appendSlice(w.a, bytes) catch return error.Closed;
    }
};

fn wire(srv: *@import("server.zig").Server, a: Allocator, body: Value, expected: Value) !void {
    var w: Wire = .{ .a = a };
    var conn: Conn = .{ .fd = -1, .peer = "", .buf = &.{}, .gpa = a, .version = "HTTP/1.1", .method = "POST", .close = false, .write_hooks = .{ .ctx = &w, .write = Wire.write } };
    var client: @import("openai_client.zig").ClientOut = .{ .conn = &conn, .a = a, .pod = true };
    @import("openai.zig").run(srv, a, client.out(), .{ .conn = &conn }, true, body);
    try std.testing.expect(std.unicode.utf8ValidateSlice(w.text.items));
    try std.testing.expect(std.mem.startsWith(u8, w.text.items, try std.fmt.allocPrint(a, "HTTP/1.1 {d} ", .{expected.get("status").?.int64().?})));
    const start = std.mem.indexOf(u8, w.text.items, "\r\n\r\n").?;
    const headers = w.text.items[0..start];
    const content_type = expected.get("headers").?.strField("content-type").?;
    try std.testing.expect(std.mem.indexOf(u8, headers, try std.mem.concat(a, u8, &.{ "Content-Type: ", content_type })) != null);
    const raw = w.text.items[start + 4 ..];
    const chunked = std.mem.indexOf(u8, headers, "Transfer-Encoding: chunked") != null;
    const decoded: bytes_test.DecodedWire = if (chunked) try bytes_test.decodeWire(a, raw) else .{ .body = raw, .complete = true };
    try std.testing.expectEqual(expected.get("complete").?.bool, decoded.complete);
    const data = decoded.body;
    var got: std.ArrayList(u8) = .empty;
    if (std.mem.startsWith(u8, content_type, "text/event-stream")) {
        var frames = std.mem.splitSequence(u8, data, "\n\n");
        while (frames.next()) |frame| {
            if (frame.len == 0) continue;
            const text = if (std.mem.eql(u8, frame, "data: [DONE]")) frame else try bytes_test.normalize(a, frame, (try json.parse(a, frame[6..])).ok);
            try got.appendSlice(a, text);
            try got.appendSlice(a, "\n\n");
        }
        if (!expected.get("complete").?.bool) {
            try std.testing.expect(conn.close);
            try std.testing.expect(!conn.response_complete);
        }
    } else if (std.mem.eql(u8, content_type, "application/json")) {
        try got.appendSlice(a, try bytes_test.normalize(a, data, (try json.parse(a, data)).ok));
    } else {
        try got.appendSlice(a, data);
        try std.testing.expect(conn.response_complete);
        try std.testing.expect(conn.close);
    }
    try std.testing.expectEqualStrings(expected.strField("body").?, got.items);
}

test "Llama serve: pod failures preserve HTTP bytes and the exact SSE prefix" {
    var outer = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer outer.deinit();
    const b = outer.allocator();
    const text = try @import("llama_prompt_test.zig").load(b);
    defer text.deinit();
    var script: @import("llama_output_test.zig").Script = .{ .reply = &.{}, .context_window = 262144 };
    const srv = try @import("server.zig").Server.init(std.testing.allocator, std.testing.io, script.engine(), text.text(), .{
        .served_name = "nanbeige",
        .model_type = "llama",
        .model_ids = &.{"nanbeige"},
        .use_drafts = false,
    }, null);
    defer srv.deinit();
    const cases = (try json.parse(b, @embedFile("fixtures/llama/failures.json"))).ok.array;
    for (cases) |case| {
        var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
        defer arena.deinit();
        const a = arena.allocator();
        script.reply = try text.text().encode(a, case.strField("raw").?, false);
        const body = try json.deepCopy(a, case.get("request").?);
        errdefer std.debug.print("FAILURE wire case {s}\n", .{case.strField("name").?});
        try wire(srv, a, body, case.get("json").?);
        try body.object.put(a, "stream", .{ .bool = true });
        try wire(srv, a, body, case.get("sse").?);
    }
    std.debug.print("FAILURE HTTP {d} JSON + {d} SSE cases equal\n", .{ cases.len, cases.len });
}

test "Llama serve: output allocation failure returns plain 500 or an incomplete stream" {
    var outer = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer outer.deinit();
    const a = outer.allocator();
    var storage: [0]u8 = .{};
    var limited = std.heap.FixedBufferAllocator.init(&storage);
    const schema = try json.newObject(a);
    try schema.put(a, "type", .{ .string = "integer" });
    try std.testing.expectError(error.OutOfMemory, @import("nanbeige_output.zig").typed(limited.allocator(), "1", .{ .object = schema }));
    for ([_]bool{ false, true }) |stream| {
        var w: Wire = .{ .a = a };
        var conn: Conn = .{ .fd = -1, .peer = "", .buf = &.{}, .gpa = a, .version = "HTTP/1.1", .method = "POST", .write_hooks = .{ .ctx = &w, .write = Wire.write } };
        var client: @import("openai_client.zig").ClientOut = .{ .conn = &conn, .a = limited.allocator(), .pod = true };
        const out = client.out();
        if (stream) {
            try out.vt.open(out.ctx, 0);
            try std.testing.expectError(error.Closed, out.vt.event(out.ctx, .{ .string = "answer" }));
            const start = std.mem.indexOf(u8, w.text.items, "\r\n\r\n").?;
            const decoded = try bytes_test.decodeWire(a, w.text.items[start + 4 ..]);
            try std.testing.expect(!decoded.complete);
            try std.testing.expectEqual(@as(usize, 0), decoded.body.len);
            try std.testing.expect(!conn.response_complete);
        } else {
            out.vt.reply(out.ctx, 200, .{ .string = "answer" });
            try std.testing.expect(std.mem.startsWith(u8, w.text.items, "HTTP/1.1 500 "));
            try std.testing.expect(std.mem.endsWith(u8, w.text.items, "\r\n\r\nInternal Server Error"));
            try std.testing.expect(std.mem.indexOf(u8, w.text.items, "Content-Type: text/plain; charset=utf-8") != null);
        }
    }
}

test "Llama serve: injected pod MemoryErrors retain their generator boundary" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const fixtures = (try json.parse(a, @embedFile("fixtures/llama/failure_faults.json"))).ok.array;
    const response = @import("nanbeige_response.zig");
    const e: response.Envelope = .{ .a = a, .id = try response.id(a, false), .created = 1234, .model = "nanbeige" };
    const role = try e.chunk(try response.role(a), null);
    const delta = try @import("chat.zig").deltaOf(a, "reasoning", "Plan.");
    try delta.object.put(a, "reasoning_content", .{ .string = "Plan." });
    const thought = try e.chunk(delta, null);
    for (fixtures) |fixture| {
        const name = fixture.strField("name").?;
        var w: Wire = .{ .a = a };
        var conn: Conn = .{ .fd = -1, .peer = "", .buf = &.{}, .gpa = a, .version = "HTTP/1.1", .method = "POST", .write_hooks = .{ .ctx = &w, .write = Wire.write } };
        var client: @import("openai_client.zig").ClientOut = .{ .conn = &conn, .a = a, .pod = true };
        const out = client.out();
        var storage: [0]u8 = .{};
        var limited = std.heap.FixedBufferAllocator.init(&storage);
        if (std.mem.eql(u8, name, "memory-whole")) {
            client.a = limited.allocator();
            out.vt.reply(out.ctx, 200, role);
            const start = std.mem.indexOf(u8, w.text.items, "\r\n\r\n").?;
            try std.testing.expectEqualStrings(fixture.get("json").?.strField("body").?, w.text.items[start + 4 ..]);
            continue;
        }
        try out.vt.open(out.ctx, 0);
        if (!std.mem.eql(u8, name, "memory-role")) try out.vt.event(out.ctx, role);
        if (std.mem.eql(u8, name, "memory-role") or std.mem.eql(u8, name, "memory-reasoning")) {
            client.a = limited.allocator();
            try std.testing.expectError(error.Closed, out.vt.event(out.ctx, if (std.mem.eql(u8, name, "memory-role")) role else thought));
        } else {
            try out.vt.event(out.ctx, thought);
            if (std.mem.eql(u8, name, "memory-producer-runtime")) response.failed(a, out, &@import("errors.zig").Cx{ .a = a, .message = "generation failed" }, error.Refused) else @import("nanbeige_failure.zig").notify(out, .stream);
        }
        const start = std.mem.indexOf(u8, w.text.items, "\r\n\r\n").?;
        const decoded = try bytes_test.decodeWire(a, w.text.items[start + 4 ..]);
        const expected = fixture.get("sse").?;
        errdefer std.debug.print("FAULT wire case {s}\n", .{name});
        try std.testing.expectEqual(expected.get("complete").?.bool, decoded.complete);
        try std.testing.expectEqualStrings(expected.strField("body").?, try bytes_test.normalize(a, decoded.body, role));
    }
    std.debug.print("FAULT {d} pod-labelled allocation boundaries equal\n", .{fixtures.len});
}
