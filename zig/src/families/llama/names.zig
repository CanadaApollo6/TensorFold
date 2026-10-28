const std = @import("std");
const core = @import("core");
const cuda = @import("cuda");
const Config = @import("config.zig").Config;
const Tensor = core.checkpoint.Tensor;
const Checkpoint = core.checkpoint.Checkpoint;

pub const Linear = struct { trellis: Tensor, suh: Tensor, svh: Tensor, halves: u32, n: usize, k: usize };
pub const Layer = struct { input_norm: Tensor, q: Linear, k: Linear, v: Linear, o: Linear, post_norm: Tensor, gate: Linear, up: Linear, down: Linear };
pub const Map = struct {
    embed: Tensor,
    layers: []Layer,
    norm: Tensor,
    head: Linear,

    pub fn deinit(m: *Map, a: std.mem.Allocator) void {
        a.free(m.layers);
    }
};

pub fn linear(ck: *Checkpoint, base: []const u8, n: usize, k: usize) !Linear {
    if (n == 0 or k == 0 or n % 128 != 0 or k % 128 != 0) return error.UnsupportedLlamaShapes;
    var buf: [200]u8 = undefined;
    const t = try ck.get(try std.fmt.bufPrint(&buf, "{s}.trellis", .{base}));
    if (t.dtype != .i16 or t.rank != 3 or t.dim(0) != k / 16 or t.dim(1) != n / 16 or t.dim(2) % 8 != 0)
        return error.UnexpectedTensor;
    const halves: u32 = @intCast(t.dim(2) / 8);
    if (cuda.exl3.variantIndex(halves, .mul1, 4) == null) return error.UnsupportedLlamaQuantization;
    const marker = try ck.get(try std.fmt.bufPrint(&buf, "{s}.mul1", .{base}));
    if ((marker.dtype != .i32 and marker.dtype != .u32) or marker.numel() != 1 or marker.bytes.len != 4 or
        std.mem.readInt(u32, marker.bytes[0..4], .little) != 0x83DCD12D) return error.UnsupportedLlamaQuantization;
    return .{
        .trellis = t,
        .suh = try ck.expect(try std.fmt.bufPrint(&buf, "{s}.suh", .{base}), .f16, &.{k}),
        .svh = try ck.expect(try std.fmt.bufPrint(&buf, "{s}.svh", .{base}), .f16, &.{n}),
        .halves = halves,
        .n = n,
        .k = k,
    };
}

pub fn map(a: std.mem.Allocator, ck: *Checkpoint, c: Config) !Map {
    const layers = try a.alloc(Layer, c.layers);
    errdefer a.free(layers);
    var buf: [180]u8 = undefined;
    const embed = try ck.expect("model.embed_tokens.weight", .bf16, &.{ c.vocab, c.hidden });
    for (layers, 0..) |*l, i| {
        l.input_norm = try ck.expect(try std.fmt.bufPrint(&buf, "model.layers.{d}.input_layernorm.weight", .{i}), .bf16, &.{c.hidden});
        l.post_norm = try ck.expect(try std.fmt.bufPrint(&buf, "model.layers.{d}.post_attention_layernorm.weight", .{i}), .bf16, &.{c.hidden});
        inline for (.{ "q", "k", "v", "o", "gate", "up", "down" }) |key| {
            const attn = comptime std.mem.eql(u8, key, "q") or std.mem.eql(u8, key, "k") or std.mem.eql(u8, key, "v") or std.mem.eql(u8, key, "o");
            const down = comptime std.mem.eql(u8, key, "down");
            const n = if (attn) (if (std.mem.eql(u8, key, "k") or std.mem.eql(u8, key, "v")) c.kvDim() else c.qDim()) else if (down) c.hidden else c.intermediate;
            const k = if (down) c.intermediate else c.hidden;
            @field(l, key) = try linear(ck, try std.fmt.bufPrint(&buf, "model.layers.{d}.{s}.{s}_proj", .{ i, if (attn) "self_attn" else "mlp", key }), n, k);
        }
    }
    const norm = try ck.expect("model.norm.weight", .bf16, &.{c.hidden});
    const head = try linear(ck, "lm_head", c.vocab, c.hidden);
    if (ck.unused() != 0) return error.UnusedTensors;
    return .{ .embed = embed, .layers = layers, .norm = norm, .head = head };
}

fn fixture(a: std.mem.Allocator, marker: u32, shape: []const u8) ![]u8 {
    const header = try std.fmt.allocPrint(a, "{{\"x.trellis\":{{\"dtype\":\"I16\",\"shape\":{s},\"data_offsets\":[0,7168]}}," ++
        "\"x.suh\":{{\"dtype\":\"F16\",\"shape\":[128],\"data_offsets\":[7168,7424]}}," ++
        "\"x.svh\":{{\"dtype\":\"F16\",\"shape\":[128],\"data_offsets\":[7424,7680]}}," ++
        "\"x.mul1\":{{\"dtype\":\"I32\",\"shape\":[],\"data_offsets\":[7680,7684]}}}}", .{shape});
    defer a.free(header);
    const bytes = try a.alloc(u8, 8 + header.len + 7684);
    @memset(bytes, 0);
    std.mem.writeInt(u64, bytes[0..8], header.len, .little);
    @memcpy(bytes[8..][0..header.len], header);
    std.mem.writeInt(u32, bytes[8 + header.len + 7680 ..][0..4], marker, .little);
    return bytes;
}

test "EXL3 mul1 admission rejects a changed marker and mismatched projection geometry" {
    const a = std.testing.allocator;
    const io = std.testing.io;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const dir = try std.fmt.allocPrint(a, ".zig-cache/tmp/{s}", .{tmp.sub_path});
    defer a.free(dir);
    for ([_]u32{ 0x83DCD12D, 0x83DCD12C }) |marker| {
        const bytes = try fixture(a, marker, "[8,8,56]");
        defer a.free(bytes);
        try tmp.dir.writeFile(io, .{ .sub_path = "model.safetensors", .data = bytes });
        var ck = try Checkpoint.openModel(a, io, dir);
        defer ck.close();
        if (marker == 0x83DCD12D) {
            const l = try linear(&ck, "x", 128, 128);
            try std.testing.expectEqual(@as(u32, 7), l.halves);
            try std.testing.expectError(error.UnexpectedTensor, linear(&ck, "x", 256, 128));
        } else {
            try std.testing.expectError(error.UnsupportedLlamaQuantization, linear(&ck, "x", 128, 128));
        }
    }
}
