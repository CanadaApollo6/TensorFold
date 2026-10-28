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

    pub fn out(c: *ClientOut) Out {
        return .{ .ctx = c, .vt = &.{ .open = open, .event = event, .reply = reply, .keepalive = keepalive } };
    }

    fn open(ctx: *anyopaque, _: usize) error{Closed}!void {
        const c: *ClientOut = @ptrCast(@alignCast(ctx));
        return sse.open(c.conn);
    }

    fn event(ctx: *anyopaque, payload: ?Value) error{Closed}!void {
        const c: *ClientOut = @ptrCast(@alignCast(ctx));
        return if (payload) |p| sse.data(c.conn, c.a, p) else sse.done(c.conn);
    }

    fn reply(ctx: *anyopaque, status: u16, payload: Value) void {
        const c: *ClientOut = @ptrCast(@alignCast(ctx));
        routes.sendValue(c.conn, c.a, status, payload);
    }

    fn keepalive(ctx: *anyopaque) error{Closed}!void {
        const c: *ClientOut = @ptrCast(@alignCast(ctx));
        return sse.comment(c.conn);
    }
};
