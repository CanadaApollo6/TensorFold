const std = @import("std");
const cuda = @import("cuda");
const op = @import("operators.zig");
const rotary = @import("rotary.zig");
pub const Rotary = struct {
    data: cuda.DeviceBuffer,
    host: cuda.HostBuffer,
    positions: rotary.Positions = .{},
    theta: f64,
    pub fn init(d: *const cuda.Driver, s: cuda.Stream, theta: f64) !Rotary {
        var data = try cuda.DeviceBuffer.alloc(d, rotary.device_bytes);
        errdefer data.free();
        var host = try cuda.HostBuffer.alloc(d, rotary.capacity * @sizeOf(i32));
        errdefer host.free();
        const inv = rotary.frequencies(theta);
        try s.synchronize();
        try data.upload(0, std.mem.sliceAsBytes(&inv));
        return .{ .data = data, .host = host, .theta = theta };
    }
    pub fn deinit(r: *Rotary) void {
        r.host.free();
        r.data.free();
    }
    pub fn begin(r: *Rotary) void {
        r.positions.reset();
    }
    pub fn rows(r: *Rotary, s: cuda.Stream, qk: *const cuda.qk_rows.QkRows, x: op.View, c: op.Rotary, y: u64, other: u64) !void {
        if (c.dim != 128 or c.theta != r.theta or c.positions.len != x.rows) return error.HeadShape;
        const change = try r.positions.prepare(c.positions);
        if (change != .same) {
            // A changed position set inside one frame must not overwrite an in-flight pinned upload.
            if (change == .replace) try s.synchronize();
            const raw = std.mem.sliceAsBytes(r.positions.values[0..r.positions.len]);
            @memcpy(r.host.bytes[0..raw.len], raw);
            try r.data.uploadAsync(rotary.frequency_bytes, r.host.bytes[0..raw.len], s.handle);
        }
        const src: cuda.qk_rows.Src = .{ .q = x.handle, .k = x.handle, .q_ld = x.stride, .k_ld = x.stride, .q_hs = 128, .k_hs = 128 };
        try qk.rowsFrom(s, src, .{ .h = c.heads, .hk = c.heads, .dim = 128 }, .{ .inv = r.data.ptr, .rot = 128 }, r.data.ptr + rotary.frequency_bytes, null, .{ .q = y, .k = other, .v = 0 }, x.rows);
    }
};
