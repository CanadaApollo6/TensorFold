const std = @import("std");
const json = @import("json");
const Conn = @import("http_conn.zig").Conn;
pub const Error = error{ PodFailure, OutOfMemory };
pub const Mode = enum { whole, stream };
pub const PodFailure = enum { whole, stream, producer_memory, producer_error };

pub fn arguments(v: json.Value, mode: Mode, depth: usize) error{PodFailure}!void {
    switch (v) {
        .array => |items| {
            if (depth >= (if (mode == .stream) @as(usize, 9991) else 9993)) return error.PodFailure;
            for (items) |item| try arguments(item, mode, depth + 1);
        },
        .object => |o| {
            if (depth >= (if (mode == .stream) @as(usize, 9991) else 9993)) return error.PodFailure;
            for (o.values()) |item| try arguments(item, mode, depth + 1);
        },
        .int => |text| if (std.mem.trimStart(u8, text, "-").len > 4300) return error.PodFailure,
        else => {},
    }
}

pub fn notify(out: anytype, kind: PodFailure) void {
    if (out.vt.pod_failure) |failed| failed(out.ctx, kind);
}

pub fn write(conn: *Conn, kind: PodFailure) void {
    if (kind == .producer_error) {
        conn.writeAll("0\r\n\r\n") catch return;
        conn.response_complete = true;
        return;
    }
    if (kind == .producer_memory) {
        const frame = "data: {\"error\": {\"message\": \"\", \"type\": \"InternalServerError\"}}\n\n";
        var head: [32]u8 = undefined;
        conn.writeAll(std.fmt.bufPrint(&head, "{x}\r\n", .{frame.len}) catch unreachable) catch return;
        conn.writeAll(frame) catch return;
        conn.writeAll("\r\n0\r\n\r\n") catch return;
        conn.response_complete = true;
        return;
    }
    if (kind == .stream) {
        conn.close = true;
        conn.response_complete = false;
        return;
    }
    conn.close = true;
    conn.startResponse(500, null) catch return;
    conn.addHeader("Content-Type", "text/plain; charset=utf-8") catch return;
    conn.addHeader("Content-Length", "21") catch return;
    if (conn.close) conn.addHeader("Connection", "close") catch return;
    conn.finish("Internal Server Error") catch {};
}

pub fn literalSource(text: []const u8) error{PodFailure}!void {
    var i: usize = 0;
    var unary: usize = 0;
    var binary: usize = 0;
    var powers: usize = 0;
    var conditionals: usize = 0;
    var lambdas: usize = 0;
    var nesting: usize = 0;
    var operand = true;
    while (i < text.len) {
        const c = text[i];
        if (std.ascii.isWhitespace(c)) {
            i += 1;
            continue;
        }
        if (c == '#') {
            while (i < text.len and text[i] != '\n') i += 1;
            continue;
        }
        if (c == '\'' or c == '"') {
            const width: usize = if (i + 2 < text.len and text[i + 1] == c and text[i + 2] == c) 3 else 1;
            i += width;
            while (i < text.len) {
                if (text[i] == '\\') {
                    i = @min(i + 2, text.len);
                } else if (text[i] == c and (width == 1 or (i + 2 < text.len and text[i + 1] == c and text[i + 2] == c))) {
                    i += width;
                    break;
                } else i += 1;
            }
            operand = false;
            unary = 0;
            continue;
        }
        if (c == '(' or c == '[' or c == '{') {
            nesting += 1;
            if (nesting > 200) return;
            operand = true;
        } else if (c == ')' or c == ']' or c == '}') {
            nesting -|= 1;
            operand = false;
        } else if (c == ',' or c == ':') {
            binary = 0;
            powers = 0;
            conditionals = 0;
            unary = 0;
            operand = true;
        } else if (std.mem.indexOfScalar(u8, "+-~*/%@|^&<>", c) != null) {
            const repeated = i + 1 < text.len and text[i + 1] == c and std.mem.indexOfScalar(u8, "*/<>", c) != null;
            if (repeated) i += 1;
            if (c == '*' and repeated) {
                powers += 1;
                if (powers >= 2988) return error.PodFailure;
            } else if (!operand) powers = 0;
            if (operand and (c == '+' or c == '-' or c == '~')) {
                unary += 1;
                if (unary >= 5975) return error.PodFailure;
            } else {
                binary += 1;
                if (binary >= 9992) return error.PodFailure;
                operand = true;
            }
        } else if (std.ascii.isAlphanumeric(c) or c == '_') {
            const start = i;
            while (i < text.len and (std.ascii.isAlphanumeric(text[i]) or text[i] == '_' or text[i] == '.')) i += 1;
            const word = text[start..i];
            if (std.mem.eql(u8, word, "lambda")) {
                lambdas += 1;
                if (lambdas >= 2988) return error.PodFailure;
                operand = true;
            } else if (std.mem.eql(u8, word, "else")) {
                conditionals += 1;
                if (conditionals >= 5975) return error.PodFailure;
                operand = true;
            } else if (std.mem.eql(u8, word, "and") or std.mem.eql(u8, word, "or")) {
                binary = 0;
                operand = true;
            } else if (std.mem.eql(u8, word, "not")) {
                unary += 1;
                if (unary >= 5976) return error.PodFailure;
            } else {
                unary = 0;
                operand = false;
            }
            continue;
        } else {
            unary = 0;
        }
        i += 1;
    }
}
