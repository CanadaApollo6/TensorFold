const std = @import("std");
const cuda = @import("cuda");
const lanes = @import("lanes");
const core = @import("core");
const weights = @import("weights.zig");
const names = @import("names.zig");
const Config = @import("config.zig").Config;
const Kernels = @import("kernels.zig").Kernels;
const Model = @import("forward.zig").Model;
const Llama = @import("backend.zig").Llama;
const memory = @import("memory.zig");

pub const model_type = "llama";
pub const formats: []const []const u8 = &.{"exl3-mul1"};
pub const default_context: i64 = 16384;
pub const max_segments: u32 = 1;
pub const prompt_rows: u32 = 32;
pub const takes_drafter = false;
pub const drafter_refusal = "--drafter: Llama P1 supports plain serial decode only";
pub const explains_open = true;
pub const Options = struct { context: usize, drafts: bool, segments: usize = 1, slide: bool = false, lanes: u32 = 1, lanes_fixed: bool = false, room: ?memory.Room = null };
pub const Loaded = struct { backend: lanes.backend.Backend, facts: lanes.Model, rows: u32, stream_bytes: usize, free_streams: u32, ctx: *anyopaque, deinit: *const fn (*anyopaque) void };
const Owned = struct { gpa: std.mem.Allocator, w: weights.Weights, k: Kernels, s: cuda.Stream, m: Model, b: Llama };

pub fn tensorWeightBytes(a: std.mem.Allocator, io: std.Io, dir: []const u8, _: bool) !u64 {
    const c = try Config.read(a, io, dir);
    var ck = try core.checkpoint.Checkpoint.openModel(a, io, dir);
    defer ck.close();
    var map = try names.map(a, &ck, c);
    defer map.deinit(a);
    return weights.deviceBytes(map, c.layers);
}

pub fn open(a: std.mem.Allocator, io: std.Io, ctx: *const cuda.Context, dir: []const u8, captured: ?[]const u8, o: Options) !Loaded {
    if (captured != null) return error.CapturedKernels;
    if (o.segments != 1 or o.slide or (o.lanes_fixed and o.lanes != 1)) return error.LlamaSerialOnly;
    const mode = try kvMode();
    const own = try a.create(Owned);
    errdefer a.destroy(own);
    own.gpa = a;
    own.w = try weights.loadMode(a, io, ctx, dir, null, o.context, mode);
    errdefer own.w.deinit();
    own.k = try Kernels.load(ctx);
    errdefer own.k.deinit();
    own.s = try cuda.Stream.init(ctx.d, false);
    errdefer own.s.deinit();
    try memory.admitMode(own.w.config, own.w.layers.len, o.context, mode, o.room);
    own.m = try Model.initMode(a, ctx.d, &own.w, &own.k, own.s, o.context, mode);
    own.b = .{ .m = &own.m };
    std.log.info("Llama KV {s}, {d} bytes/token; BOS bf16", .{ @tagName(mode), try mode.bytes(own.w.layers.len, 1, own.w.config.kv_heads, own.w.config.head_dim) });
    return .{ .backend = own.b.backend(), .facts = own.b.facts(), .rows = 1, .stream_bytes = try mode.bytes(own.w.layers.len, o.context, own.w.config.kv_heads, own.w.config.head_dim), .free_streams = 1, .ctx = own, .deinit = release };
}

pub fn kvMode() !cuda.kv4.Mode {
    return cuda.kv4.Mode.parse(if (std.c.getenv("TF_LLAMA_KV_BITS")) |p| std.mem.span(p) else "16");
}

fn release(p: *anyopaque) void {
    const x: *Owned = @ptrCast(@alignCast(p));
    x.m.deinit();
    x.s.deinit();
    x.k.deinit();
    x.w.deinit();
    x.gpa.destroy(x);
}

pub fn explain(_: ?*anyopaque, err: anyerror) ?[]const u8 {
    return switch (err) {
        error.BadKvBits => "TF_LLAMA_KV_BITS takes 4 or 16",
        error.LlamaGpuBusy => "Llama weights and KV do not fit while preserving 2 GiB for the desktop",
        error.LlamaUnqualifiedGpu => "Llama dense pointwise kernels are qualified for sm_86 only",
        error.LlamaMemoryBudget => "Llama KV and scratch exceed the CUDA memory budget after weights",
        error.MissingBos => "Llama requires its bos_token_id at position zero",
        error.LlamaSerialOnly => "Llama P1 supports one serial stream without drafts",
        error.UnsupportedLlamaRopeOrVision => "Llama P1 supports full unscaled RoPE and text input only",
        error.UnsupportedLlamaQuantization => "Llama P1 requires EXL3 mul1 projections with a compiled bit width",
        error.PastContext => "the prompt and reply exceed the Llama context window",
        error.CapturedKernels => "Llama uses native shared operators; unset TENSORFOLD_CUDA_KERNELS",
        else => null,
    };
}
