const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const lanes = @import("lanes");
const Model = @import("forward.zig").Model;
pub const config = @import("../dflash2/config.zig");
pub const weights = @import("../dflash2/weights.zig");
pub const checkpoint = @import("../qwen27/checkpoint.zig");
pub const plan = @import("../dflash2/cuda_plan.zig");
pub const operators = @import("../dflash2/operators.zig");
pub const runtime = @import("../dflash2/cuda_runtime.zig");
pub const storage = @import("../dflash2/cuda_weights.zig");
pub const Attachment = Binding(Model, runtime.Runtime);

pub fn Binding(comptime Target: type, comptime Draft: type) type {
    return struct {
        const Self = @This();
        m: *Target,
        r: *Draft,
        captures: cuda.DeviceBuffer,
        history: cuda.DeviceBuffer,
        retained: @import("tap_history.zig").History = .{},
        held: [7]u32 = undefined,
        count: usize = 0,
        windows: [8]lanes.config.Cost, // the target's verify of 1 to 8 rows, timed at load: what prices a draft depth
        timed: usize = 8, // widths that fit the context

        pub fn init(a: std.mem.Allocator, io: std.Io, m: *Target, dir: []const u8, format: storage.Format, room: ?@import("memory.zig").Room) !Self {
            const c = m.w.config;
            const embed: core.checkpoint.Tensor = .{ .dtype = .bf16, .rank = 2, .shape = .{ c.vocab, c.hidden, 1, 1, 1 }, .bytes = m.w.embed.?.bytes };
            const r = try runtime.Runtime.load(a, io, m.d, m.s, dir, .{ .ptr = m, .embed = embed, .layers = @intCast(c.layers), .head = head }, format, room);
            errdefer r.deinit();
            var windows: [8]lanes.config.Cost = undefined;
            const timed = try timeWindows(m, io, &windows);
            const capture_bytes = @max(8, m.prompt_rows) * r.graph.config.tapWidth() * 2;
            const history_bytes = r.graph.config.window * r.graph.config.tapWidth() * 2;
            try @import("../dflash2/admission.zig").check(capture_bytes + history_bytes, room);
            var free: usize = 0;
            var total: usize = 0;
            try m.d.check(m.d.api.cuMemGetInfo_v2(&free, &total), "prefill taps budget");
            if (capture_bytes + history_bytes + (2 << 30) > free) return error.LlamaMemoryBudget;
            var captures = try cuda.DeviceBuffer.alloc(m.d, capture_bytes);
            errdefer captures.free();
            const history = try cuda.DeviceBuffer.alloc(m.d, history_bytes);
            m.tap_ids = r.graph.config.taps;
            return .{ .m = m, .r = r, .captures = captures, .history = history, .windows = windows, .timed = timed };
        }
        // Row 0 is the BOS row every prompt writes; the first prefill rewrites the rest (nothing is cached yet).
        fn timeWindows(m: *Target, io: std.Io, out: *[8]lanes.config.Cost) !usize {
            const tokens: [8]u32 = @splat(m.w.config.bos);
            const timed = @min(out.len, m.context, m.capacity_rows);
            for (out[0..timed], 1..) |*c, n| {
                var best = std.math.inf(f64);
                for (0..4) |i| {
                    const t0 = std.Io.Clock.awake.now(io);
                    try m.batch(tokens[0..n], 0);
                    try m.s.synchronize();
                    const ms = @as(f64, @floatFromInt(t0.durationTo(std.Io.Clock.awake.now(io)).toNanoseconds())) / 1e6;
                    if (i > 0) best = @min(best, ms);
                }
                c.* = .{ .width = @intCast(n), .ms = best };
            }
            return timed;
        }
        pub fn deinit(d: *Self) void {
            d.m.tap_buffer = 0;
            d.r.deinit();
            d.captures.free();
            d.history.free();
        }
        fn head(p: *anyopaque, x: u64, y: u64, rows: usize) !void {
            const m: *Model = @ptrCast(@alignCast(p));
            try m.k.exl.decode(m.s, m.w.head, x, .bf16, rows, y, .f32);
        }
        pub fn capture(d: *Self, row: usize) void {
            d.m.tap_buffer = d.captures.ptr + row * d.r.graph.config.tapWidth() * 2;
        }
        pub fn ingest(d: *Self, row: usize, position: usize) !void {
            const width = d.r.graph.config.tapWidth() * 2;
            const source = d.captures.ptr + row * width;
            try d.r.ingest(source, position);
            try d.m.d.check(d.m.d.api.cuMemcpyDtoDAsync_v2(d.history.ptr + (position % d.r.graph.config.window) * width, source, width, d.m.s.handle), "retain prompt taps");
            d.retained.append(position, d.r.graph.config.window);
        }
        pub fn reuse(d: *Self, common: usize, length: usize) !usize {
            const start = length -| (d.r.graph.config.window - 1);
            return if (d.retained.covers(start, common)) common else @min(common, start);
        }
        pub fn restore(d: *Self, start: usize, common: usize) !void {
            if (!d.retained.covers(start, common)) return error.MissingTapHistory;
            d.retained.truncate(common);
            const width = d.r.graph.config.tapWidth() * 2;
            if (common > start) for (start..common) |position| {
                try d.r.ingest(d.history.ptr + (position % d.r.graph.config.window) * width, position);
            };
        }
        pub fn propose(d: *Self, pending: u32, depth: u32) !void {
            d.count = 0;
            if (depth == 0) return;
            var t = try d.r.propose(pending, @min(depth, 7));
            defer t.deinit();
            @memcpy(d.held[0..t.tokens.len], t.tokens);
            d.count = t.tokens.len;
        }
    };
}

pub const lattice = @import("../dflash2/lattice.zig");
