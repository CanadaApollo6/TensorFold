const std = @import("std");
pub const reference = @import("kv4_ref.zig");
const cuda = @import("root.zig");

pub const Mode = enum {
    bf16,
    q4,

    pub fn parse(bits: []const u8) !Mode {
        if (std.mem.eql(u8, bits, "16")) return .bf16;
        if (std.mem.eql(u8, bits, "4")) return .q4;
        return error.BadKvBits;
    }

    pub fn rowBytes(mode: Mode, dim: usize) usize {
        return if (mode == .q4) dim / 2 + dim / 16 else dim * 2;
    }

    pub fn bytes(mode: Mode, layers: usize, context: usize, heads: usize, dim: usize) !usize {
        const rows = try std.math.mul(usize, try std.math.mul(usize, layers * 2, heads), context);
        return std.math.mul(usize, rows, mode.rowBytes(dim));
    }
};

pub const Attention = struct {
    mod: cuda.Module,
    pack_fn: cuda.Function,
    unpack_fn: cuda.Function,
    partial: [2]cuda.Function,
    merge_fn: cuda.Function,
    prompt_fn: cuda.Function,
    prompt_merge: cuda.Function,

    pub fn load(d: *const cuda.Driver) !Attention {
        var mod = try cuda.Module.load(d, cuda.kernels.kv4);
        errdefer mod.unload();
        return .{ .mod = mod, .pack_fn = try mod.function("tf_kv4_pack"), .unpack_fn = try mod.function("tf_kv4_unpack"), .partial = .{ try mod.function("tf_kv4_partial128"), try mod.function("tf_kv4_partial256") }, .prompt_fn = try mod.function("tf_kv4_prompt128"), .prompt_merge = try mod.function("tf_kv4_prompt_merge"), .merge_fn = try mod.function("tf_kv4_merge") };
    }

    pub fn unload(k: *Attention) void {
        k.mod.unload();
    }

    pub fn groups(keys: usize) usize {
        return (keys + 511) / 512;
    }

    pub fn scratchBytes(heads: usize, dim: usize, keys: usize) usize {
        return heads * groups(keys) * 4 * (dim + 2) * 4;
    }

    fn convert(k: *const Attention, f: cuda.Function, s: cuda.Stream, x: u64, out: u64, rows: usize, dim: usize) !void {
        _ = k;
        if (dim != 128 and dim != 256) return error.HeadShape;
        if (rows == 0) return;
        var a: cuda.Args = .{};
        a.add(x);
        a.add(out);
        a.add(@as(i32, @intCast(rows)));
        a.add(@as(i32, @intCast(dim)));
        try cuda.launch.launch(f, .{ .grid = .{ .x = @intCast((rows * (dim / 32) + 3) / 4) }, .block = .{ .x = 128 } }, s, &a);
    }

    pub fn pack(k: *const Attention, s: cuda.Stream, x: u64, out: u64, rows: usize, dim: usize) !void {
        try k.convert(k.pack_fn, s, x, out, rows, dim);
    }

    pub fn unpack(k: *const Attention, s: cuda.Stream, x: u64, out: u64, rows: usize, dim: usize) !void {
        try k.convert(k.unpack_fn, s, x, out, rows, dim);
    }

    pub const Rows = struct { q: u64, k: u64, v: u64, first_k: u64 = 0, first_v: u64 = 0, last_k: u64 = 0, last_v: u64 = 0 };

    pub fn prompt(k: *const Attention, s: cuda.Stream, r: Rows, scratch: u64, out: u64, heads: usize, kvheads: usize, dim: usize, position: usize, rows: usize) !void {
        if (dim != 128 or rows == 0 or rows > 8) return error.HeadShape;
        const ng = groups(position + rows);
        var a: cuda.Args = .{};
        inline for (.{ "q", "k", "v", "first_k", "first_v", "last_k", "last_v" }) |name| a.add(@field(r, name));
        a.add(scratch);
        for ([_]usize{ heads, kvheads, position + 1, ng }) |n| a.add(@as(i32, @intCast(n)));
        try cuda.launch.launch(k.prompt_fn, .{ .grid = .{ .x = @intCast(ng), .y = @intCast(heads), .z = @intCast(rows) }, .block = .{ .x = 128 } }, s, &a);
        var b: cuda.Args = .{};
        b.add(scratch);
        b.add(out);
        b.add(@as(i32, @intCast(dim)));
        b.add(@as(i32, @intCast(ng)));
        try cuda.launch.launch(k.prompt_merge, .{ .grid = .{ .x = @intCast(heads), .y = @intCast(rows) }, .block = .{ .x = @intCast(dim) } }, s, &b);
    }

    pub fn run(k: *const Attention, s: cuda.Stream, r: Rows, scratch: u64, out: u64, heads: usize, kvheads: usize, dim: usize, keys: usize) !void {
        if ((dim != 128 and dim != 256) or kvheads == 0 or heads % kvheads != 0 or keys == 0) return error.HeadShape;
        const ng = groups(keys);
        var a: cuda.Args = .{};
        inline for (.{ "q", "k", "v", "first_k", "first_v", "last_k", "last_v" }) |name| a.add(@field(r, name));
        a.add(scratch);
        for ([_]usize{ heads, kvheads, keys, ng }) |n| a.add(@as(i32, @intCast(n)));
        try cuda.launch.launch(k.partial[@intFromBool(dim == 256)], .{ .grid = .{ .x = @intCast(ng), .y = @intCast(heads) }, .block = .{ .x = 128 } }, s, &a);
        var b: cuda.Args = .{};
        b.add(scratch);
        b.add(out);
        b.add(@as(i32, @intCast(dim)));
        b.add(@as(i32, @intCast(ng)));
        try cuda.launch.launch(k.merge_fn, .{ .grid = .{ .x = @intCast(heads) }, .block = .{ .x = @intCast(dim) } }, s, &b);
    }
};

test "KV formats account real head widths and reject ambiguous bits" {
    try std.testing.expectEqual(Mode.q4, try Mode.parse("4"));
    try std.testing.expectEqual(Mode.bf16, try Mode.parse("16"));
    try std.testing.expectError(error.BadKvBits, Mode.parse("8"));
    try std.testing.expectEqual(@as(usize, 72), Mode.q4.rowBytes(128));
    try std.testing.expectEqual(@as(usize, 144), Mode.q4.rowBytes(256));
    try std.testing.expectEqual(@as(usize, 59904), try Mode.q4.bytes(52, 1, 8, 128));
}

test {
    _ = reference;
}
