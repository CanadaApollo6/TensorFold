const std = @import("std");
const eos_ids = @import("core").tokenizer.eos;

pub const Config = struct {
    layers: usize,
    hidden: usize,
    heads: usize,
    kv_heads: usize,
    head_dim: usize,
    intermediate: usize,
    vocab: usize,
    max_context: usize,
    bos: u32,
    eos: [4]u32 = @splat(0),
    eos_count: usize = 0,
    rope_theta: f32,
    eps: f32,

    pub fn qDim(c: Config) usize {
        return c.heads * c.head_dim;
    }

    pub fn kvDim(c: Config) usize {
        return c.kv_heads * c.head_dim;
    }

    pub fn kvBytes(c: Config, layers: usize, context: usize) !usize {
        return std.math.mul(usize, layers * 4 * c.kvDim(), context);
    }

    pub fn isEos(c: Config, token: u32) bool {
        return std.mem.indexOfScalar(u32, c.eos[0..c.eos_count], token) != null;
    }

    pub fn read(a: std.mem.Allocator, io: std.Io, dir: []const u8) !Config {
        const path = try std.fs.path.join(a, &.{ dir, "config.json" });
        defer a.free(path);
        const bytes = try std.Io.Dir.cwd().readFileAlloc(io, path, a, .limited(1 << 22));
        defer a.free(bytes);
        var c = try parse(a, bytes);
        const gen_path = try std.fs.path.join(a, &.{ dir, "generation_config.json" });
        defer a.free(gen_path);
        const gen = std.Io.Dir.cwd().readFileAlloc(io, gen_path, a, .limited(1 << 20)) catch |e| switch (e) {
            error.FileNotFound => return c,
            else => return e,
        };
        defer a.free(gen);
        const p = try std.json.parseFromSlice(std.json.Value, a, gen, .{});
        defer p.deinit();
        const ids = try eos_ids.merge(a, c.eos[0..c.eos_count], p.value, c.vocab);
        defer a.free(ids);
        try c.setEos(ids);
        return c;
    }

    fn setEos(c: *Config, ids: []const u32) !void {
        if (ids.len > c.eos.len) return error.TooManyEos;
        @memcpy(c.eos[0..ids.len], ids);
        c.eos_count = ids.len;
    }
};

fn int(o: std.json.ObjectMap, key: []const u8) !usize {
    const v = o.get(key) orelse return error.BadLlamaConfig;
    if (v != .integer or v.integer <= 0 or v.integer > 1 << 24) return error.BadLlamaConfig;
    return @intCast(v.integer);
}

fn number(o: std.json.ObjectMap, key: []const u8, default: f32) !f32 {
    const v = o.get(key) orelse return default;
    const f: f32 = switch (v) {
        .integer => |n| @floatFromInt(n),
        .float => |n| @floatCast(n),
        else => return error.BadLlamaConfig,
    };
    if (!std.math.isFinite(f) or f <= 0) return error.BadLlamaConfig;
    return f;
}

fn string(o: std.json.ObjectMap, key: []const u8, expected: []const u8) bool {
    const v = o.get(key) orelse return false;
    return v == .string and std.mem.eql(u8, v.string, expected);
}

pub fn parse(a: std.mem.Allocator, bytes: []const u8) !Config {
    const p = try std.json.parseFromSlice(std.json.Value, a, bytes, .{});
    defer p.deinit();
    if (p.value != .object) return error.BadLlamaConfig;
    const o = p.value.object;
    if (!string(o, "model_type", "llama")) return error.NotLlama;
    if (!string(o, "hidden_act", "silu")) return error.UnsupportedLlamaActivation;
    for ([_][]const u8{ "attention_bias", "mlp_bias", "tie_word_embeddings" }) |key| {
        if (o.get(key)) |v| if (v != .bool or v.bool) return error.UnsupportedLlamaLayout;
    }
    for ([_][]const u8{ "rope_scaling", "rope_parameters", "vision_config" }) |key| {
        if (o.get(key)) |v| if (v != .null) return error.UnsupportedLlamaRopeOrVision;
    }
    if (o.get("partial_rotary_factor")) |_| {
        if (try number(o, "partial_rotary_factor", 1) != 1) return error.UnsupportedLlamaRopeOrVision;
    }
    const q = o.get("quantization_config") orelse return error.UnsupportedLlamaQuantization;
    if (q != .object or !string(q.object, "quant_method", "exl3") or !string(q.object, "codebook", "mul1"))
        return error.UnsupportedLlamaQuantization;
    const hidden = try int(o, "hidden_size");
    const heads = try int(o, "num_attention_heads");
    const hd = if (o.get("head_dim") != null) try int(o, "head_dim") else hidden / heads;
    var c: Config = .{
        .layers = try int(o, "num_hidden_layers"),
        .hidden = hidden,
        .heads = heads,
        .kv_heads = try int(o, "num_key_value_heads"),
        .head_dim = hd,
        .intermediate = try int(o, "intermediate_size"),
        .vocab = try int(o, "vocab_size"),
        .max_context = try int(o, "max_position_embeddings"),
        .bos = @intCast(try int(o, "bos_token_id")),
        .rope_theta = try number(o, "rope_theta", 10000),
        .eps = try number(o, "rms_norm_eps", 1e-5),
    };
    if (c.layers > 128 or hd != 128 or c.heads % c.kv_heads != 0 or c.heads / c.kv_heads > 16 or
        hidden != c.qDim() or hidden % 128 != 0 or c.intermediate % 128 != 0 or c.vocab % 128 != 0 or c.bos >= c.vocab)
        return error.UnsupportedLlamaShapes;
    const ids = try eos_ids.resolve(a, p.value, .null, c.vocab);
    defer a.free(ids);
    try c.setEos(ids);
    return c;
}

const fixture =
    \\{"model_type":"llama","hidden_act":"silu","num_hidden_layers":52,"hidden_size":7168,
    \\"num_attention_heads":56,"num_key_value_heads":8,"head_dim":128,"intermediate_size":22016,
    \\"vocab_size":166912,"max_position_embeddings":524288,"bos_token_id":166100,"eos_token_id":166101,
    \\"rms_norm_eps":0.00001,"rope_theta":120000000,"rope_scaling":null,
    \\"quantization_config":{"quant_method":"exl3","codebook":"mul1"}}
;

test "Nanbeige geometry admits GQA seven and accounts BF16 KV" {
    const c = try parse(std.testing.allocator, fixture);
    try std.testing.expectEqual(@as(usize, 7), c.heads / c.kv_heads);
    try std.testing.expectEqual(@as(usize, 208 * 1024), try c.kvBytes(c.layers, 1));
    try std.testing.expect(c.isEos(166101));
    try std.testing.expectEqual(@as(f32, 120000000), c.rope_theta);
}

test "unsupported rope scaling and quantization refuse before loading" {
    const a = std.testing.allocator;
    const rope = try std.mem.replaceOwned(u8, a, fixture, "\"rope_scaling\":null", "\"rope_scaling\":{\"factor\":8}");
    defer a.free(rope);
    try std.testing.expectError(error.UnsupportedLlamaRopeOrVision, parse(a, rope));
    const quant = try std.mem.replaceOwned(u8, a, fixture, "\"mul1\"", "\"mcg\"");
    defer a.free(quant);
    try std.testing.expectError(error.UnsupportedLlamaQuantization, parse(a, quant));
}
