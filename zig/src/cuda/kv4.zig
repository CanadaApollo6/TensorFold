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
    tiles_fn: cuda.Function,
    tile_merge: cuda.Function,

    /// A prompt call's most rows, and the absolute key span each prompt partial covers (tf_kv4_tiles128's SEG).
    pub const wave = 128;
    pub const segment = 2048;

    pub fn load(d: *const cuda.Driver) !Attention {
        var mod = try cuda.Module.load(d, cuda.kernels.kv4);
        errdefer mod.unload();
        const tiles = try mod.function("tf_kv4_tiles128");
        try d.check(d.api.cuFuncSetAttribute(tiles.handle, .max_dynamic_shared_size_bytes, @intCast(tileShared(8))), "KV4 tile shared memory");
        return .{ .mod = mod, .pack_fn = try mod.function("tf_kv4_pack"), .unpack_fn = try mod.function("tf_kv4_unpack"), .partial = .{ try mod.function("tf_kv4_partial128"), try mod.function("tf_kv4_partial256") }, .tiles_fn = tiles, .tile_merge = try mod.function("tf_kv4_tile_merge"), .merge_fn = try mod.function("tf_kv4_merge") };
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

    /// Scratch for decode rows and prompt waves over a context of `keys`.
    pub fn workBytes(heads: usize, dim: usize, keys: usize) usize {
        return @max(scratchBytes(heads, dim, keys), wave * heads * ((keys + segment - 1) / segment) * (dim + 2) * 4);
    }

    fn tileShared(warps: usize) usize {
        return 4 * 64 * 72 + 2 * 64 * 128 * 2 + 2 * 64 * 4 * 4 + warps * 16 * 32 * 4;
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

    /// Prompt rows from `position` on, keys already packed; BOS and each row's own key come from first_* and last_*.
    pub fn prompt(k: *const Attention, s: cuda.Stream, r: Rows, scratch: u64, out: u64, heads: usize, kvheads: usize, dim: usize, position: usize, rows: usize) !void {
        if (dim != 128 or rows == 0 or rows > wave or kvheads == 0 or heads % kvheads != 0 or heads / kvheads > 8) return error.HeadShape;
        const last = position + rows - 1;
        const segs = if (last >= 2) (last - 1) / segment + 1 else 1;
        var a: cuda.Args = .{};
        for ([_]u64{ r.q, r.k, r.v, scratch }) |v| a.add(v);
        for ([_]usize{ heads, kvheads, position, rows, segs }) |n| a.add(@as(i32, @intCast(n)));
        const warps = heads / kvheads;
        try cuda.launch.launch(k.tiles_fn, .{ .grid = .{ .x = @intCast((rows + 15) / 16), .y = @intCast(kvheads), .z = @intCast(segs) }, .block = .{ .x = @intCast(32 * warps) }, .shared = @intCast(tileShared(warps)) }, s, &a);
        var b: cuda.Args = .{};
        inline for (.{ "q", "first_k", "first_v", "last_k", "last_v" }) |name| b.add(@field(r, name));
        b.add(scratch);
        b.add(out);
        for ([_]usize{ heads, kvheads, position, segs }) |n| b.add(@as(i32, @intCast(n)));
        try cuda.launch.launch(k.tile_merge, .{ .grid = .{ .x = @intCast(heads), .y = @intCast(rows) }, .block = .{ .x = @intCast(dim) } }, s, &b);
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
