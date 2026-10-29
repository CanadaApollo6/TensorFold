const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const Model = @import("forward.zig").Model;
pub const config = @import("../dflash2/config.zig");
pub const weights = @import("../dflash2/weights.zig");
pub const checkpoint = @import("../qwen27/checkpoint.zig");
pub const plan = @import("../dflash2/cuda_plan.zig");
pub const operators = @import("../dflash2/operators.zig");
pub const runtime = @import("../dflash2/cuda_runtime.zig");
pub const storage = @import("../dflash2/cuda_weights.zig");
pub const Attachment = struct {
    m: *Model,
    r: *runtime.Runtime,
    captures: cuda.DeviceBuffer,
    held: [7]u32 = undefined,
    count: usize = 0,

    pub fn init(a: std.mem.Allocator, io: std.Io, m: *Model, dir: []const u8, format: storage.Format) !Attachment {
        const c = m.w.config;
        const embed: core.checkpoint.Tensor = .{ .dtype = .bf16, .rank = 2, .shape = .{ c.vocab, c.hidden, 1, 1, 1 }, .bytes = m.w.embed.?.bytes };
        const r = try runtime.Runtime.load(a, io, m.d, m.s, dir, .{ .ptr = m, .embed = embed, .layers = @intCast(c.layers), .head = head }, format);
        errdefer r.deinit();
        const captures = try cuda.DeviceBuffer.alloc(m.d, 8 * r.graph.config.tapWidth() * 2);
        m.tap_ids = r.graph.config.taps;
        return .{ .m = m, .r = r, .captures = captures };
    }
    pub fn deinit(d: *Attachment) void {
        d.m.tap_buffer = 0;
        d.r.deinit();
        d.captures.free();
    }
    fn head(p: *anyopaque, x: u64, y: u64, rows: usize) !void {
        const m: *Model = @ptrCast(@alignCast(p));
        try m.k.exl.decode(m.s, m.w.head, x, .bf16, rows, y, .f32);
    }
    pub fn capture(d: *Attachment, row: usize) void {
        d.m.tap_buffer = d.captures.ptr + row * d.r.graph.config.tapWidth() * 2;
    }
    pub fn ingest(d: *Attachment, row: usize, position: usize) !void {
        try d.r.ingest(d.captures.ptr + row * d.r.graph.config.tapWidth() * 2, position);
    }
    pub fn propose(d: *Attachment, pending: u32, depth: u32) !void {
        d.count = 0;
        if (depth == 0) return;
        var t = try d.r.propose(pending, @min(depth, 7));
        defer t.deinit();
        @memcpy(d.held[0..t.tokens.len], t.tokens);
        d.count = t.tokens.len;
    }
};
