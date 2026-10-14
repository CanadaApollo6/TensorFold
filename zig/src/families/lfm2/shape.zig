//! LFM2-MoE's shapes (Liquid's lfm2_moe, e.g. LFM2.5-24B-A2B) from config.json, held to what the kernels take.
const std = @import("std");

pub const head_dim = 64;
pub const conv_taps = 3; // conv_L_cache: the short conv's taps, two rows of state
pub const max_eos = 4;

pub const Geometry = struct {
    hidden: usize,
    intermediate: usize, // the dense layers' SwiGLU
    layers: usize,
    dense_layers: usize, // the first layers keep a dense MLP
    heads: usize,
    kv_heads: usize,
    experts: usize,
    top_k: usize,
    moe_inter: usize,
    vocab: usize,
    eps: f32,
    theta: f64,
    attention: u64, // bit i: layer i is attention, else a short conv

    pub fn isAttention(g: Geometry, i: usize) bool {
        return (g.attention >> @intCast(i)) & 1 == 1;
    }
    pub fn qInner(g: Geometry) usize {
        return g.heads * head_dim;
    }
    pub fn kvInner(g: Geometry) usize {
        return g.kv_heads * head_dim;
    }
    pub fn attentionLayers(g: Geometry) usize {
        return @popCount(g.attention);
    }
};

pub const Config = struct {
    g: Geometry,
    eos: [max_eos]u32 = undefined,
    eos_count: usize = 0,

    pub fn read(gpa: std.mem.Allocator, io: std.Io, dir: []const u8) !Config {
        const path = try std.fs.path.join(gpa, &.{ dir, "config.json" });
        defer gpa.free(path);
        const text = try std.Io.Dir.cwd().readFileAlloc(io, path, gpa, .limited(1 << 22));
        defer gpa.free(text);
        return parse(gpa, text);
    }

    pub fn parse(gpa: std.mem.Allocator, text: []const u8) !Config {
        const parsed = try std.json.parseFromSlice(std.json.Value, gpa, text, .{});
        defer parsed.deinit();
        const o = parsed.value.object;
        const model_type = (o.get("model_type") orelse return error.MissingConfig).string;
        if (!std.mem.eql(u8, model_type, "lfm2_moe")) return error.NotLfm2Moe;
        const int = struct {
            fn f(obj: std.json.ObjectMap, key: []const u8) !usize {
                const v = obj.get(key) orelse {
                    std.log.err("config.json has no {s}", .{key});
                    return error.MissingConfig;
                };
                return @intCast(v.integer);
            }
        }.f;
        const num = struct {
            fn f(v: std.json.Value) f64 {
                return switch (v) {
                    .float => |x| x,
                    .integer => |x| @floatFromInt(x),
                    else => std.math.nan(f64),
                };
            }
        }.f;
        var g: Geometry = .{
            .hidden = try int(o, "hidden_size"),
            .intermediate = try int(o, "intermediate_size"),
            .layers = try int(o, "num_hidden_layers"),
            .dense_layers = try int(o, "num_dense_layers"),
            .heads = try int(o, "num_attention_heads"),
            .kv_heads = try int(o, "num_key_value_heads"),
            .experts = try int(o, "num_experts"),
            .top_k = try int(o, "num_experts_per_tok"),
            .moe_inter = try int(o, "moe_intermediate_size"),
            .vocab = try int(o, "vocab_size"),
            .eps = @floatCast(num(o.get("norm_eps") orelse return error.MissingConfig)),
            .theta = 0,
            .attention = 0,
        };
        if (o.get("rope_parameters")) |rp| {
            if (rp.object.get("rope_theta")) |t| g.theta = num(t);
        }
        if (g.theta == 0) if (o.get("rope_theta")) |t| {
            g.theta = num(t);
        };
        const kinds = (o.get("layer_types") orelse return error.MissingConfig).array.items;
        if (kinds.len != g.layers or g.layers > 64) return error.UnsupportedLfm2Geometry;
        for (kinds, 0..) |k, i| {
            if (std.mem.eql(u8, k.string, "full_attention")) g.attention |= @as(u64, 1) << @intCast(i) else if (!std.mem.eql(u8, k.string, "conv")) return error.UnsupportedLfm2Geometry;
        }
        if ((o.get("conv_L_cache") orelse return error.MissingConfig).integer != conv_taps) return error.UnsupportedLfm2Geometry;
        if (o.get("conv_bias")) |b| if (b == .bool and b.bool) return error.UnsupportedLfm2Geometry;
        if (o.get("tie_word_embeddings")) |t| if (t == .bool and !t.bool) return error.UnsupportedLfm2Geometry;
        if (g.hidden / g.heads != head_dim or g.hidden % 64 != 0 or g.hidden > 2048 or g.heads != 4 * g.kv_heads) return error.UnsupportedLfm2Geometry;
        // the router kernel: 8 warps of 8 experts, top k at most 8; the experts' kernels: widths of 64-column groups
        if (g.experts > 64 or g.top_k > 8 or g.moe_inter % 64 != 0 or g.intermediate % 64 != 0 or g.theta <= 0) return error.UnsupportedLfm2Geometry;
        var c: Config = .{ .g = g };
        if (o.get("eos_token_id")) |e| switch (e) {
            .integer => |x| {
                c.eos[0] = @intCast(x);
                c.eos_count = 1;
            },
            .array => |a| for (a.items) |x| {
                if (c.eos_count == max_eos) break;
                c.eos[c.eos_count] = @intCast(x.integer);
                c.eos_count += 1;
            },
            else => {},
        };
        return c;
    }
};

test "LFM2.5-24B-A2B's config" {
    const text =
        \\{"model_type": "lfm2_moe", "hidden_size": 2048, "intermediate_size": 11776, "num_hidden_layers": 4,
        \\ "num_dense_layers": 2, "num_attention_heads": 32, "num_key_value_heads": 8, "num_experts": 64,
        \\ "num_experts_per_tok": 4, "moe_intermediate_size": 1536, "vocab_size": 128000, "norm_eps": 1e-05,
        \\ "rope_parameters": {"rope_theta": 10000000.0}, "conv_L_cache": 3, "conv_bias": false,
        \\ "layer_types": ["conv", "conv", "full_attention", "conv"], "eos_token_id": [124900]}
    ;
    const c = try Config.parse(std.testing.allocator, text);
    try std.testing.expect(c.g.isAttention(2) and !c.g.isAttention(3));
    try std.testing.expectEqual(@as(usize, 1), c.g.attentionLayers());
    try std.testing.expectEqual(@as(u32, 124900), c.eos[0]);
    try std.testing.expectEqual(@as(f64, 1e7), c.g.theta);
}
