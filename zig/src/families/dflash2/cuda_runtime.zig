const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const cfg = @import("config.zig");
const weights = @import("weights.zig");
const ck = @import("../qwen27/checkpoint.zig");
const op = @import("operators.zig");
const store = @import("cuda_weights.zig");
const callbacks = @import("cuda_callbacks.zig");
pub const Target = struct {
    ptr: *anyopaque,
    embed: ck.Tensor,
    layers: u32,
    head: *const fn (*anyopaque, u64, u64, usize) anyerror!void,
};
pub const Runtime = struct {
    a: std.mem.Allocator,
    d: *const cuda.Driver,
    s: cuda.Stream,
    target: Target,
    checkpoint: core.Checkpoint,
    graph: weights.Graph,
    w: store.Store,
    modules: [6]cuda.Module,
    affine: cuda.qlinear.Affine4Kernels,
    dense: cuda.dense_rows.DenseRows,
    norms: cuda.norms.Norms,
    qk: cuda.qk_rows.QkRows,
    act: cuda.act_rows.ActRows,
    conv_fn: cuda.Function,
    attn_fn: cuda.Function,
    narrow_fn: cuda.Function,
    ring: cuda.DeviceBuffer,
    context: op.Context = .{ .cache = 1, .begin = 0, .end = 0 },
    transaction: bool = false,
    active: bool = false,
    pool: std.ArrayList(cuda.DeviceBuffer) = .empty,
    cursor: usize = 0,
    profile: ?@import("cuda_profile.zig").Profile = null,

    pub fn load(a: std.mem.Allocator, io: std.Io, d: *const cuda.Driver, s: cuda.Stream, dir: []const u8, target: Target, format: store.Format, room: ?@import("admission.zig").Room) !*Runtime {
        const path = try std.fs.path.join(a, &.{ dir, "config.json" });
        defer a.free(path);
        const raw = try std.Io.Dir.cwd().readFileAlloc(io, path, a, .limited(1 << 20));
        defer a.free(raw);
        const c = try cfg.parse(a, raw);
        if (c.target_layers != target.layers or target.embed.rank != 2 or target.embed.dtype != .bf16 or
            target.embed.shape[0] != c.vocab or target.embed.shape[1] != c.hidden or c.head_dim != 128) return error.TargetBinding;
        const r = try a.create(Runtime);
        errdefer a.destroy(r);
        r.* = .{ .a = a, .d = d, .s = s, .target = target, .checkpoint = try core.Checkpoint.openModel(a, io, dir), .graph = undefined, .w = undefined, .modules = undefined, .affine = undefined, .dense = undefined, .norms = undefined, .qk = undefined, .act = undefined, .conv_fn = undefined, .attn_fn = undefined, .narrow_fn = undefined, .ring = undefined };
        errdefer r.checkpoint.close();
        errdefer {
            for (r.pool.items) |*b| b.free();
            r.pool.deinit(a);
        }
        const names = try weights.names(a, c);
        defer weights.freeNames(a, names);
        r.graph = try weights.load(a, .{ .names = names, .source = ck.Source.from(&r.checkpoint) }, c, .{ .dense = target.embed });
        errdefer r.graph.deinit();
        if (r.checkpoint.unused() != 0) return error.DraftTensorCoverage;
        const plan = @import("cuda_plan.zig");
        var footprint: store.Footprint = .{};
        for (names) |name| footprint.add(name, try r.checkpoint.get(name), format);
        const sizes = try plan.slots(a, &r.graph, format);
        defer a.free(sizes);
        var scratch_bytes: usize = 0;
        for (sizes) |n| scratch_bytes += n;
        const live = plan.fixed(c, footprint, scratch_bytes);
        const peak = @max(live, footprint.load_peak + plan.split_bytes);
        try @import("admission.zig").check(peak, room);
        var free: usize = 0;
        var total: usize = 0;
        try d.check(d.api.cuMemGetInfo_v2(&free, &total), "draft memory budget");
        if (peak + (2 << 30) > free) return error.LlamaMemoryBudget;
        var made: usize = 0;
        errdefer for (r.modules[0..made]) |*m| m.unload();
        for (&r.modules, [_][]const u8{ cuda.kernels.qmm_group, cuda.kernels.qmm_prefill, cuda.kernels.lane_gemv, cuda.kernels.affine4_pack, cuda.kernels.qwen35_draft, cuda.kernels.torch_pointwise }) |*m, image| {
            m.* = try cuda.Module.load(d, image);
            made += 1;
        }
        r.affine = try cuda.qlinear.Affine4Kernels.resolve(d, r.modules[0], r.modules[1], r.modules[2], r.modules[3], 1, false);
        errdefer r.affine.deinit();
        r.dense = try cuda.dense_rows.DenseRows.load(d);
        errdefer r.dense.unload();
        r.norms = try cuda.norms.Norms.load(d);
        errdefer r.norms.unload();
        r.qk = try cuda.qk_rows.QkRows.load(d);
        errdefer r.qk.unload();
        r.act = try cuda.act_rows.ActRows.load(d);
        errdefer r.act.unload();
        r.conv_fn = try r.modules[4].function("tf_q35d_dyn_conv");
        r.attn_fn = try r.modules[4].function("tf_q35d_attn");
        r.w = try store.Store.load(a, d, s, &r.checkpoint, c, format, &r.affine);
        errdefer r.w.deinit();
        r.ring = try cuda.DeviceBuffer.alloc(d, @as(usize, c.layers) * c.window * c.kvWidth() * 4);
        errdefer r.ring.free();
        for (sizes) |n| {
            var buffer = try cuda.DeviceBuffer.alloc(d, n);
            errdefer buffer.free();
            try r.pool.append(a, buffer);
        }
        r.narrow_fn = try r.modules[5].function("tf_f32_to_bf16_kernel");
        if (std.c.getenv("TF_LLAMA_DRAFT_PROFILE") != null)
            r.profile = try @import("cuda_profile.zig").Profile.init(d, io);
        return r;
    }
    pub fn deinit(r: *Runtime) void {
        r.s.synchronize() catch {};
        if (r.profile) |*p| p.deinit();
        for (r.pool.items) |*b| b.free();
        r.pool.deinit(r.a);
        r.ring.free();
        r.w.deinit();
        r.act.unload();
        r.qk.unload();
        r.norms.unload();
        r.dense.unload();
        r.affine.deinit();
        for (&r.modules) |*m| m.unload();
        r.graph.deinit();
        r.checkpoint.close();
        r.a.destroy(r);
    }
    pub fn bytes(r: *const Runtime) usize {
        var n = r.ring.len + r.w.footprint.device;
        for (r.pool.items) |b| n += b.len;
        return n;
    }
    pub fn view(ptr: u64, rows: u32, width: u32, dtype: ck.DType) op.View {
        return .{ .handle = ptr, .rows = rows, .width = width, .stride = width, .dtype = dtype };
    }
    pub fn alloc(r: *Runtime, rows: u32, width: u32, dtype: ck.DType) !op.View {
        const n = @as(usize, rows) * width * dtype.size();
        if (r.cursor == r.pool.items.len or r.pool.items[r.cursor].len < n) return error.DraftPoolExceeded;
        const p = r.pool.items[r.cursor].ptr;
        r.cursor += 1;
        return view(p, rows, width, dtype);
    }
    pub fn upload(r: *Runtime, ptr: u64, raw: []const u8) !void {
        try r.s.synchronize();
        try r.d.check(r.d.api.cuMemcpyHtoD_v2(ptr, raw.ptr, raw.len), "draft upload");
    }
    pub fn download(r: *Runtime, ptr: u64, raw: []u8) !void {
        try r.s.synchronize();
        try r.d.check(r.d.api.cuMemcpyDtoH_v2(raw.ptr, ptr, raw.len), "draft download");
    }
    pub fn copy(r: *Runtime, dst: u64, src: u64, n: usize) !void {
        try r.d.check(r.d.api.cuMemcpyDtoDAsync_v2(dst, src, n, r.s.handle), "draft copy");
    }
    pub fn run(r: *Runtime, f: cuda.Function, grid: cuda.Dim3, threads: u32, values: anytype) !void {
        var args: cuda.Args = .{};
        inline for (values) |v| args.add(v);
        try cuda.launch.launch(f, .{ .grid = grid, .block = .{ .x = threads } }, r.s, &args);
    }
    pub fn ops(r: *Runtime) op.Ops {
        return .{ .ptr = r, .vtable = &callbacks.vtable };
    }
    pub fn preparation(r: *const Runtime) op.Preparation {
        return .{ .mode = if (r.w.format == .bf16) .bf16_reference else .prepared_q4_reference, .quantization_verified = r.w.format == .affine4, .source_sha256 = core.affine4.signature() };
    }
    pub fn reset(r: *Runtime) void {
        r.context = .{ .cache = 1, .begin = 0, .end = 0 };
        r.transaction = false;
    }
    pub fn ingest(r: *Runtime, taps: u64, position: u64) !void {
        if (r.profile) |*p| p.begin(.ingest);
        errdefer if (r.profile) |*p| {
            p.active = false;
        };
        const c = r.graph.config;
        if (position != r.context.end) return error.BadCommittedTaps;
        const next: op.Context = .{ .cache = 1, .begin = @max(r.context.begin, (position + 1) -| (c.window - 1)), .end = position + 1 };
        try @import("execution.zig").append(r.ops(), &r.graph, r.preparation(), r.context, view(taps, 1, c.tapWidth(), .bf16), &.{position}, next);
        if (r.profile) |*p| try p.finish(r.s, r.context.end, 0);
    }
    pub fn propose(r: *Runtime, anchor: u32, nodes: u32) !@import("selector.zig").Tree {
        if (r.profile) |*p| p.begin(.proposal);
        errdefer if (r.profile) |*p| {
            p.active = false;
        };
        var lattice = try @import("execution.zig").forward(r.ops(), &r.graph, r.preparation(), r.context, anchor);
        defer lattice.deinit();
        const ticket = if (r.profile) |*p| try p.enter(r.s, .selector, false) else null;
        const tree = try @import("selector.zig").propose(r.a, &r.graph, lattice, anchor, r.context.end + 1, .{ .nodes = nodes, .children = 1 });
        errdefer {
            var owned = tree;
            owned.deinit();
        }
        if (r.profile) |*p| {
            p.leave(r.s, ticket);
            try p.finish(r.s, r.context.end, nodes);
        }
        return tree;
    }
};
