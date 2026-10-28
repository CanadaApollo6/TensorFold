const std = @import("std");
const json = @import("json");
const sse = @import("sse.zig");
const routes = @import("routes.zig");
const Conn = @import("http_conn.zig").Conn;
const Value = json.Value;
const Allocator = std.mem.Allocator;
const Out = @import("openai.zig").Out;

/// A chat reply written straight to the client.
pub const ClientOut = struct {
    conn: *Conn,
    a: Allocator,
    pod: bool = false,

    pub fn out(c: *ClientOut) Out {
        return .{ .ctx = c, .vt = &.{ .open = open, .event = event, .reply = reply, .keepalive = keepalive, .pod_failure = podFailure } };
    }

    fn open(ctx: *anyopaque, _: usize) error{Closed}!void {
        const c: *ClientOut = @ptrCast(@alignCast(ctx));
        if (!c.pod) return sse.open(c.conn);
        c.conn.startResponse(200, null) catch return error.Closed;
        c.conn.addHeader("Content-Type", "text/event-stream; charset=utf-8") catch return error.Closed;
        c.conn.addHeader("Connection", "close") catch return error.Closed;
        c.conn.addHeader("Transfer-Encoding", "chunked") catch return error.Closed;
        try c.conn.finish("");
        c.conn.response_complete = false;
    }

    fn event(ctx: *anyopaque, payload: ?Value) error{Closed}!void {
        const c: *ClientOut = @ptrCast(@alignCast(ctx));
        if (c.pod) {
            if (payload) |p| return c.frame(@import("nanbeige_response.zig").bytes(c.a, p, true) catch |e| {
                const choices = p.get("choices") orelse Value.null;
                const delta = if (choices == .array and choices.array.len > 0) choices.array[0].get("delta") orelse Value.null else Value.null;
                podFailure(ctx, if (e == error.OutOfMemory and delta.has("reasoning")) .producer_memory else .stream);
                return error.Closed;
            });
            try c.frame("data: [DONE]\n\n");
            try c.conn.writeAll("0\r\n\r\n");
            c.conn.response_complete = true;
            return;
        }
        return if (payload) |p| sse.data(c.conn, c.a, p) else sse.done(c.conn);
    }

    fn reply(ctx: *anyopaque, status: u16, payload: Value) void {
        const c: *ClientOut = @ptrCast(@alignCast(ctx));
        if (c.pod) return c.conn.sendJson(status, @import("nanbeige_response.zig").bytes(c.a, payload, false) catch return podFailure(ctx, .whole));
        routes.sendValue(c.conn, c.a, status, payload);
    }

    fn frame(c: *ClientOut, data: []const u8) error{Closed}!void {
        var head: [32]u8 = undefined;
        try c.conn.writeAll(std.fmt.bufPrint(&head, "{x}\r\n", .{data.len}) catch unreachable);
        try c.conn.writeAll(data);
        try c.conn.writeAll("\r\n");
    }

    fn podFailure(ctx: *anyopaque, kind: @import("nanbeige_failure.zig").PodFailure) void {
        const c: *ClientOut = @ptrCast(@alignCast(ctx));
        @import("nanbeige_failure.zig").write(c.conn, kind);
    }

    fn keepalive(ctx: *anyopaque) error{Closed}!void {
        const c: *ClientOut = @ptrCast(@alignCast(ctx));
        return if (c.pod) c.frame(": ping\n\n") else sse.comment(c.conn);
    }
};
