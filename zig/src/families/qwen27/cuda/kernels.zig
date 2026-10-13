//! The 27B's CUDA kernels: our .cu fatbins with the Python wrappers' launch logic, and the captured Triton set.

const std = @import("std");
const cuda = @import("cuda");
pub const torch_ops = @import("nemotron").torch_ops;

/// Mangled names of the instantiations the copies in zig/kernels/cuda export (cuobjdump -symbols of each fatbin).
const sym = struct {
    const group = "_ZN12tf_qmm_group12group_kernelILi64ELi16ELi64ELi1ELi4ELi8ELb0ELb0ELb0ELb0EEEvPK13__nv_bfloat16PKfNS_5PartsEiiiii";
    const wide = [_][:0]const u8{
        "_ZN12tf_qmm_group12group_kernelILi64ELi32ELi64ELi1ELi4ELi4ELb0ELb0ELb0ELb0EEEvPK13__nv_bfloat16PKfNS_5PartsEiiiii",
        "_ZN12tf_qmm_group12group_kernelILi64ELi64ELi64ELi1ELi4ELi4ELb0ELb0ELb0ELb0EEEvPK13__nv_bfloat16PKfNS_5PartsEiiiii",
        "_ZN12tf_qmm_group12group_kernelILi64ELi64ELi128ELi2ELi4ELi3ELb0ELb0ELb0ELb0EEEvPK13__nv_bfloat16PKfNS_5PartsEiiiii",
    };
    const gemv = "_ZN12tf_lane_gemv11gemv_kernelILi64ELi64ELi8EEEvPK13__nv_bfloat16PKfNS_4PartEiii";
    const prefill_mm = "_ZN14tf_qmm_prefill14prefill_kernelILi64ELi128ELi128ELi2ELi2ELi2ELb0EEEvPK13__nv_bfloat16PKjS3_S3_Pviiiiii";
    const pattn = "_ZN20tf_prefill_attention12pattn_kernelILi256ELi8ELi2ELi2EEEvPK13__nv_bfloat16S3_S3_PS1_iiiiif";
    const chain128 = "_ZN14tf_gdn_prefill12chain_kernelI13__nv_bfloat16Li128ELi32EEEvPKT_S4_PKS1_PKfS8_S8_PfPS1_iii";
    const chain64 = "_ZN14tf_gdn_prefill12chain_kernelI13__nv_bfloat16Li64ELi32EEEvPKT_S4_PKS1_PKfS8_S8_PfPS1_iii";
};

/// qmm_group.cu's Part and Parts, passed by value: four projections at most, one used here.
pub const Part = extern struct { w: u64, scales: u64, biases: u64, out: u64, n: c_int, npad: c_int, sk: c_int, tiles: c_int, first: c_int };
pub const Parts = extern struct { p: [4]Part, count: c_int };

/// lane_gemv.cu's Part: one projection whose column tiles the CTAs share out.
pub const GemvPart = extern struct { w: u64, scales: u64, biases: u64, out: u64, n: c_int, npad: c_int, sk: c_int, tiles: c_int };

/// gdn.cu's Pending (the tree kernel's rows still to fold): all null here, commits replay instead.
pub const Pending = extern struct { k: u64 = 0, v: u64 = 0, g: u64 = 0, beta: u64 = 0, rows: u64 = 0, row_stride: c_int = 0, counts: u64 = 0, count_stride: c_int = 0 };

/// qmm_lane.cu's Seg and Segs (the 3090 stack's segmented launch), passed by value: four weights at most, one used here.
pub const Seg = extern struct { w: u64, scales: u64, biases: u64, out: u64, part: u64, n: c_int, npad: c_int, sk: c_int, blocks: c_int, sync: c_int };
pub const Segs = extern struct { s: [4]Seg, count: c_int };

comptime {
    std.debug.assert(@sizeOf(Seg) == 64 and @sizeOf(Segs) == 264);
    std.debug.assert(@sizeOf(Part) == 56 and @sizeOf(Parts) == 232 and @sizeOf(GemvPart) == 48 and @sizeOf(Pending) == 64);
}

pub const group_smem: u32 = 35328; // LaneTile<64, 16, 64, 1, 4, 8>::SMEM
/// qmm_group's GB10 tiles 3-5 for wider rounds: rows a tile, columns a tile, threads, LaneTile::SMEM.
pub const Wide = struct { bm: usize, bn: usize, threads: u32, smem: u32 };
pub const wide = [_]Wide{ .{ .bm = 32, .bn = 64, .threads = 128, .smem = 26112 }, .{ .bm = 64, .bn = 64, .threads = 128, .smem = 43008 }, .{ .bm = 64, .bn = 128, .threads = 256, .smem = 39168 } };
/// qmm_lane.cu's Tile<64, BM, 64, 1, 4, STAGES>: a stage's bytes at 16 rows, and SMEM at 4 stages for 16, 32 and 64 rows.
pub const seg_stage: u32 = 16 * 128 + 64 * 32 + 2 * 64 * 2 + 16 * 4;
pub const lane_smem = [_]u32{ 17664, 26112, 43008 };
/// Rows a wide lane launch takes at most (rows are independent, so wider rounds run in turns): K-slice partials fit `part`.
pub const lane_rows: usize = 128;
/// split_k stops once tiles x slices reach 192, so a weight's slices hold fewer than 384 column tiles of partials.
pub const part_bytes: usize = lane_rows * 384 * 64 * 4;
pub const sync_counters: usize = 1 << 14;
pub const prefill_mm_smem: u32 = 41984; // Tile<64, 128, 128, 2, 2, 2>::SMEM: prompt_tile 9, two stages
pub const pattn_smem: u32 = 98304; // two 32-key slots of 256 dims beside eight warps' staged queries
/// kv4.cu's attn4 blocks: (MT tiles of) staged query fragments, row maxima and sums, probability fragments.
pub fn attn4Smem(mt: usize) u32 {
    return @intCast((mt * 16 * 32 * 4 + 2 * 4 * 16 * mt + mt * 4 * 32 * 2) * 4);
}

/// One item of a batched copy: 16-byte aligned pointers, a multiple of 16 bytes.
pub const Copy = extern struct { dst: u64, src: u64, bytes: u64 };

/// A tiled 4-bit projection: packed words, (kg, npad) scales and biases, n outputs from k inputs.
pub const QLinear = struct { w: u64, s: u64, b: u64, n: usize, k: usize, npad: usize };

pub const Kernels = struct {
    d: *const cuda.Driver,
    mods: [17]cuda.Module,
    glue: @import("triton.zig").Glue,
    group: cuda.Function,
    wide: [wide.len]cuda.Function, // tiles 3-5: rows <= 32, <= 64, more
    gemv: cuda.Function,
    seg_probe: [2][3]cuda.Function, // timing probes by stages and P (TF_QWEN27_TUNE probe=1..3): wrong outputs
    seg: [2][2][3]cuda.Function, // qmm_lane's segmented launch by stages (3, 4), 8-token tiles (1, 2), slices a block (1, 2, 4)
    lane: [3]cuda.Function, // qmm_lane's row tiles of 16, 32 and 64 for wider rounds
    reduce: cuda.Function,
    part: cuda.DeviceBuffer, // K-slice partials of the pre-sm_90 launches
    sync: cuda.DeviceBuffer, // the segmented launch's finished-slice counters, zero between launches
    sms: usize,
    prefill_mm: cuda.Function,
    pattn: cuda.Function,
    kv4_rotate: cuda.Function, // kv4.cu: H32 rotation, 4-bit packing, attn4's shared chunks, tail and merge, 4-bit prefill
    kv4_quant: cuda.Function,
    kv4_merge: cuda.Function,
    attn4_shared: [3]cuda.Function,
    attn4_tail: cuda.Function,
    pattn4: cuda.Function,
    trees: [cuda.kernels.gdn_variants.len]cuda.Function, // tree_kernel by gdn_variants: [0] the chain one
    chip: [3]c_int, // major, minor, SMs: what dispatch_tree reads
    replay: cuda.Function,
    prompt_chain: cuda.Function,
    prompt_rows: usize, // the prompt chain's value rows a block: 64 when heads < SMs
    pack_dense: cuda.Function,
    transpose16: cuda.Function,
    argmax: [2]cuda.Function, // torch_ops/argmax.cu's split pick: slices, then their merge
    torch: torch_ops.Functions,
    copy_batch: cuda.Function,
    tap: cuda.Function,
    rms_norm: cuda.Function,
    inv_freq: cuda.Function,
    torch_rms: cuda.Function, // F.rms_norm's bits on bf16 rows of 5120 (torch_ops/norm_rotary.cu)
    rotary: cuda.Function,
    gemv_blocks: usize, // resident lane_gemv CTAs: per SM times SMs
    gemv_rows: usize, // windows of at most this many rows take lane_gemv (none; TF_QWEN27_GEMV_ROWS)
    gb10: bool,
    clusters: bool, // sm_90 and later: K slices meet in a cluster; older chips take lane_gemv 16 rows at a time
    discrete: bool, // the card has its own memory: checkpoint bytes reach it through page-locked slots
    moe: ?@import("moe.zig").Kernels = null, // qwen3_5_moe's experts (Engine.init loads them for an MoE checkpoint)

    /// Loads every module; `triton_dir` holds the captured aot.json and cubins for this GPU.
    pub fn load(gpa: std.mem.Allocator, io: std.Io, ctx: *const cuda.Context, triton_dir: []const u8, heads: usize) !Kernels {
        const d = ctx.d;
        if (!cuda.kernels.available) return error.BuiltWithoutKernels;
        var k: Kernels = undefined;
        k.d = d;
        const kk = cuda.kernels;
        const images = [_][]const u8{ kk.qmm_group, kk.qmm_prefill, kk.lane_gemv, kk.prefill_attention, kk.gdn, kk.gdn_prefill, kk.affine4_pack, kk.torch_argmax, kk.torch_topk, kk.torch_pointwise, kk.torch_indexing, kk.torch_movement, kk.torch_nemotron_constants, kk.qwen_ops, kk.torch_norm_rotary, kk.qmm_lane, kk.kv4 };
        var loaded: usize = 0;
        errdefer for (k.mods[0..loaded]) |*m| m.unload();
        for (images, 0..) |img, i| {
            k.mods[i] = try cuda.Module.load(d, img);
            loaded += 1;
        }
        k.group = try k.mods[0].function(sym.group);
        for (&k.wide, sym.wide) |*f, name| f.* = try k.mods[0].function(name);
        k.prefill_mm = try k.mods[1].function(sym.prefill_mm);
        k.gemv = try k.mods[2].function(sym.gemv);
        k.pattn = try k.mods[3].function(sym.pattn);
        for (&k.trees, cuda.kernels.gdn_variants) |*f, v| f.* = try k.mods[4].function(v.symbol);
        k.replay = try k.mods[4].function(cuda.kernels.gdn_symbols.replay_bf16);
        k.pack_dense = try k.mods[6].function(cuda.qlinear.symbols.pack_words);
        k.transpose16 = try k.mods[6].function(cuda.qlinear.symbols.pack_scales);
        k.argmax = .{ try k.mods[7].function("tf_argmax_part_bf16"), try k.mods[7].function("tf_argmax_merge_i32") };
        k.torch = try torch_ops.Functions.resolve(k.mods[7..13]);
        k.copy_batch = try k.mods[13].function("tf_copy_batch");
        k.tap = try k.mods[13].function("tf_tap");
        k.rms_norm = try k.mods[13].function("tf_rms_norm");
        k.inv_freq = try k.mods[13].function("tf_inv_freq");
        k.rotary = try k.mods[13].function("tf_rotary");
        k.torch_rms = try k.mods[14].function("tf_dflash_rms5120_bf16_fma_kernel");
        inline for (0..2) |si| inline for (0..2) |ti| inline for (0..3) |gi| {
            const st: u32 = 3 + si;
            const g: u32 = 1 << gi;
            const f = &k.seg[si][ti][gi];
            f.* = try k.mods[15].function(std.fmt.comptimePrint("tf_qmm_seg_s{d}_t{d}_g{d}", .{ st, ti + 1, g }));
            try f.allowDynamicShared(if (g > 1) g * st * seg_stage else @max(st * seg_stage, 4096));
        };
        inline for (0..2) |si| inline for (0..3) |pi| {
            const st: u32 = 3 + si;
            const f = &k.seg_probe[si][pi];
            f.* = try k.mods[15].function(std.fmt.comptimePrint("tf_qmm_seg_probe{d}_s{d}", .{ pi + 1, st }));
            try f.allowDynamicShared(@max(st * seg_stage, 4096));
        };
        for (&k.lane, [_][:0]const u8{ "tf_qmm_lane_m16", "tf_qmm_lane_m32", "tf_qmm_lane_m64" }, lane_smem) |*f, name, smem| {
            f.* = try k.mods[15].function(name);
            try f.allowDynamicShared(smem);
        }
        k.reduce = try k.mods[15].function("tf_qmm_reduce");
        k.kv4_rotate = try k.mods[16].function("tf_kv4_rotate");
        k.kv4_quant = try k.mods[16].function("tf_kv4_quant");
        k.kv4_merge = try k.mods[16].function("tf_kv4_merge");
        for (&k.attn4_shared, [_][:0]const u8{ "tf_attn4_shared_m1", "tf_attn4_shared_m2", "tf_attn4_shared_m3" }, 1..) |*f, name, mt| {
            f.* = try k.mods[16].function(name);
            try f.allowDynamicShared(attn4Smem(mt));
        }
        k.attn4_tail = try k.mods[16].function("tf_attn4_tail");
        try k.attn4_tail.allowDynamicShared(attn4Smem(1));
        k.pattn4 = try k.mods[16].function("tf_pattn4");
        try k.pattn4.allowDynamicShared(pattn_smem);
        const sms: usize = @intCast(try ctx.attribute(.multiprocessor_count));
        // gdn_prefill_cuda: a block takes 128 value rows, or 64 when there are fewer heads than SMs
        k.prompt_rows = if (heads >= sms) 128 else 64;
        k.prompt_chain = try k.mods[5].function(if (k.prompt_rows == 128) sym.chain128 else sym.chain64);
        k.glue = try @import("triton.zig").Glue.load(gpa, io, d, ctx.device, triton_dir);
        errdefer k.glue.unload();
        k.glue.tree.sms = if (std.c.getenv("TF_TREE_SMS")) |v| std.fmt.parseInt(usize, std.mem.span(v), 10) catch 0 else @intCast(try ctx.attribute(.multiprocessor_count));
        k.sms = sms;
        k.clusters = try ctx.attribute(.compute_capability_major) >= 9;
        k.part = try cuda.DeviceBuffer.alloc(d, if (k.clusters) 0 else part_bytes);
        errdefer k.part.free();
        k.sync = try cuda.DeviceBuffer.alloc(d, if (k.clusters) 0 else sync_counters * 4);
        errdefer k.sync.free();
        if (!k.clusters) try k.sync.fill32(0, null);
        try k.group.allowDynamicShared(group_smem);
        for (k.wide, wide) |f, t| try f.allowDynamicShared(t.smem);
        try k.gemv.allowDynamicShared(group_smem);
        try k.prefill_mm.allowDynamicShared(prefill_mm_smem);
        try k.pattn.allowDynamicShared(pattn_smem);
        k.gemv_blocks = @max(1, try k.gemv.occupancy(128, group_smem)) * sms;
        const major = try ctx.attribute(.compute_capability_major);
        const minor = try ctx.attribute(.compute_capability_minor);
        k.gb10 = major == 12 and minor == 1;
        k.discrete = try ctx.attribute(.integrated) == 0;
        k.gemv_rows = 0; // a GB10 runs the 27B's projections faster in Python's clusters, 1 to 16 rows alike
        if (std.c.getenv("TF_QWEN27_GEMV_ROWS")) |v| k.gemv_rows = std.fmt.parseInt(usize, std.mem.span(v), 10) catch k.gemv_rows;
        k.chip = .{ major, minor, @intCast(sms) };
        k.moe = null;
        return k;
    }

    pub fn deinit(k: *Kernels) void {
        if (k.moe) |*m| m.unload();
        k.glue.unload();
        k.part.free();
        k.sync.free();
        for (&k.mods) |*m| m.unload();
    }
};

/// qmm.split_k: K slices fixed by the weight's shape, never by the row count.
/// TF_QWEN27_TUNE (timing experiments, e.g. "st=3,g=1,sk=2,tiles=384,sk@10240x5120=1,st@5120x6144=4"): the segmented
/// launch's stages, slices side by side, a fixed K split (changes the sums' order, so the bits) and splitK's tiles x
/// slices target, for all weights or (key@NxK) one shape's; 0 keeps the default.
pub const Shape = struct { n: usize = 0, k: usize = 0, sk: usize = 0, st: u32 = 0 };
pub const Tune = struct { st: u32 = 0, g: u32 = 0, sk: usize = 0, tiles: usize = 192, probe: u32 = 0, shapes: [8]Shape = @splat(.{}) };

fn shapeOf(t: *const Tune, n: usize, k: usize) Shape {
    for (t.shapes) |sh| if (sh.n == n and sh.k == k) return sh;
    return .{};
}
pub var tune: Tune = .{};
var tune_read = false;

fn readTune() void {
    tune_read = true;
    tune = parseTune(std.mem.span(std.c.getenv("TF_QWEN27_TUNE") orelse return));
    std.debug.print("TF_QWEN27_TUNE: {any}\n", .{tune});
}

pub fn parseTune(v: []const u8) Tune {
    tune_read = true;
    var t: Tune = .{};
    var it = std.mem.tokenizeScalar(u8, v, ',');
    while (it.next()) |kv| {
        const eq = std.mem.indexOfScalar(u8, kv, '=') orelse continue;
        const n = std.fmt.parseInt(usize, kv[eq + 1 ..], 10) catch continue;
        var key = kv[0..eq];
        if (std.mem.indexOfScalar(u8, key, '@')) |at| {
            const x = std.mem.indexOfScalar(u8, key, 'x') orelse continue;
            const sn = std.fmt.parseInt(usize, key[at + 1 .. x], 10) catch continue;
            const sk_ = std.fmt.parseInt(usize, key[x + 1 ..], 10) catch continue;
            key = key[0..at];
            for (&t.shapes) |*sh| if (sh.n == 0 or (sh.n == sn and sh.k == sk_)) {
                sh.n = sn;
                sh.k = sk_;
                if (std.mem.eql(u8, key, "sk")) sh.sk = n else if (std.mem.eql(u8, key, "st")) sh.st = @intCast(n);
                break;
            };
            continue;
        }
        if (std.mem.eql(u8, key, "st")) t.st = @intCast(n) else if (std.mem.eql(u8, key, "g")) t.g = @intCast(n) else if (std.mem.eql(u8, key, "sk")) t.sk = n else if (std.mem.eql(u8, key, "tiles")) t.tiles = n else if (std.mem.eql(u8, key, "probe")) t.probe = @intCast(n);
    }
    return t;
}

/// A weight's split in a launch its input's other weights already fill: one, unless TF_QWEN27_TUNE names its shape.
fn jointSplit(q: QLinear) usize {
    const sk = shapeOf(&tune, q.n, q.k).sk;
    return if (sk > 0) sk else 1;
}

pub fn splitK(n: usize, k: usize) usize {
    if (!tune_read) readTune();
    if (shapeOf(&tune, n, k).sk > 0) return shapeOf(&tune, n, k).sk;
    if (tune.sk > 0) return tune.sk;
    const tiles = (n + 63) / 64;
    const groups = k / 64;
    var sk: usize = 1;
    while (sk < 8 and tiles * sk < tune.tiles and groups % (sk * 2) == 0 and groups / (sk * 2) >= 8) sk *= 2;
    return sk;
}

fn int(x: usize) c_int {
    return @intCast(x);
}

fn u(x: usize) u32 {
    return @intCast(x);
}

/// The GDN shapes every launch takes: key heads, value heads, value rows a head.
pub const Heads = struct { hk: usize, hv: usize, dv: usize };

/// The most slices a row's greedy pick splits into.
pub const argmax_slices = 64;

/// Launch helpers on one stream; each mirrors the Python wrapper it replaces.
pub const Ops = struct {
    k: *const Kernels,
    s: cuda.Stream,

    fn go(o: Ops, f: cuda.Function, grid: [3]usize, block: u32, shared: u32, args: *cuda.Args) !void {
        try cuda.launch.launch(f, .{ .grid = .{ .x = u(grid[0]), .y = u(grid[1]), .z = u(grid[2]) }, .block = .{ .x = block }, .shared = shared }, o.s, args);
    }

    /// qmm.matmul on sm_12x: x (rows, k) bf16 with group sums xs -> out (rows, n) bf16, qmm_group's bits.
    pub fn dense(o: Ops, x: u64, xs: u64, q: QLinear, out: u64, rows: usize) !void {
        const sk = splitK(q.n, q.k);
        if (!o.k.clusters and !(sk > 1 and rows <= o.k.gemv_rows)) return o.lane3090(x, xs, q, out, rows, sk);
        if (rows > 16) return o.wideTile(x, xs, q, out, rows, sk);
        // Python's clusters, a CTA a K slice; lane_gemv (byte-equal) when TF_QWEN27_GEMV_ROWS asks for it
        return if (sk > 1 and rows <= o.k.gemv_rows) o.gemv(x, xs, q, out, rows, sk) else o.cluster(x, xs, q, out, rows, sk);
    }

    /// Several weights over the same input: up to 16 rows on the 3090 path, one segmented launch for each K split they share;
    /// else each alone. When their column tiles together fill splitK's 192 (GDN's qkv, z, b, a; attention's q, k, v), every
    /// weight takes the whole K in one launch: a K split only fills the card, and at 16 rows it cost ~0.7 ms a round.
    pub fn denseMany(o: Ops, x: u64, xs: u64, qs: []const QLinear, outs: []const u64, rows: usize) !void {
        if (o.k.clusters or rows > 16 or (o.k.gemv_rows > 0 and rows <= o.k.gemv_rows)) {
            for (qs, outs) |q, out| try o.dense(x, xs, q, out, rows);
            return;
        }
        if (!tune_read) readTune();
        var all_tiles: usize = 0;
        for (qs) |q| all_tiles += (q.n + 63) / 64;
        const joint = qs.len > 1 and all_tiles >= tune.tiles;
        var done: u8 = 0;
        for (qs, 0..) |q0, i| {
            if (done & (@as(u8, 1) << @intCast(i)) != 0) continue;
            const sk = if (joint) jointSplit(q0) else splitK(q0.n, q0.k);
            var segs: Segs = std.mem.zeroes(Segs);
            var blocks: usize = 0;
            var sync: usize = 0;
            var part: u64 = o.k.part.ptr;
            for (qs[i..], outs[i..], i..) |q, out, j| {
                if ((if (joint) jointSplit(q) else splitK(q.n, q.k)) != sk or q.k != q0.k or segs.count == 4) continue;
                done |= @as(u8, 1) << @intCast(j);
                const tiles = (q.n + 63) / 64;
                segs.s[@intCast(segs.count)] = .{ .w = q.w, .scales = q.s, .biases = q.b, .out = out, .part = if (sk > 1) part else 0, .n = int(q.n), .npad = int(q.npad), .sk = int(sk), .blocks = int(tiles * sk), .sync = int(sync) };
                segs.count += 1;
                blocks += tiles * sk;
                sync += tiles;
                part += sk * rows * q.n * 4;
            }
            if (sk > 1 and (sync > sync_counters or part - o.k.part.ptr > part_bytes)) return error.TooManyTiles;
            try o.segLaunch(x, xs, segs, blocks, sk, rows, q0.k);
        }
    }

    /// qmm.matmul at a K split the caller names (matmul_rows takes the stacked weight's for each part).
    pub fn denseSplit(o: Ops, x: u64, xs: u64, q: QLinear, out: u64, rows: usize, sk: usize) !void {
        if (!o.k.clusters and !(sk > 1 and rows <= o.k.gemv_rows)) return o.lane3090(x, xs, q, out, rows, sk);
        if (rows > 16) return o.wideTile(x, xs, q, out, rows, sk);
        return if (sk > 1 and rows <= o.k.gemv_rows) o.gemv(x, xs, q, out, rows, sk) else o.cluster(x, xs, q, out, rows, sk);
    }

    /// tree_forward's taps: bf16(x + pending) of `rows` rows into columns [col, col + d) of rows `stride` wide.
    pub fn tap(o: Ops, x: u64, pending: u64, out: u64, rows: usize, d: usize, stride: usize, col: usize) !void {
        var a: cuda.Args = .{};
        for ([_]u64{ x, pending, out }) |v| a.add(v);
        for ([_]usize{ d, stride, col }) |v| a.add(int(v));
        try o.go(o.k.tap, .{ rows, 4, 1 }, 256, 0, &a);
    }

    /// lane_gemv: every K slice of a column tile in one CTA, summed in slice order, CTAs looping over the tiles.
    fn gemv(o: Ops, x: u64, xs: u64, q: QLinear, out: u64, rows: usize, sk: usize) !void {
        const tiles = (q.n + 63) / 64;
        var a: cuda.Args = .{};
        a.add(x);
        a.add(xs);
        a.add(GemvPart{ .w = q.w, .scales = q.s, .biases = q.b, .out = out, .n = int(q.n), .npad = int(q.npad), .sk = int(sk), .tiles = int(tiles) });
        for ([_]usize{ rows, q.k, q.k }) |v| a.add(int(v));
        const cfg: cuda.Config = .{ .grid = .{ .x = u(@min(tiles, o.k.gemv_blocks)) }, .block = .{ .x = 128 }, .shared = group_smem, .pdl = o.k.gb10 };
        try cuda.launch.launch(o.k.gemv, cfg, o.s, &a);
    }

    /// Before sm_90 (no clusters): the 3090 stack's qmm.py, its K slices added in slice order as 0.6.6's qmm.cu does there.
    /// Up to 16 rows, the segmented launch with the weights as the mma's A (2 or 4 slices side by side in a block, else
    /// parked and added by the tile's last block); wider rounds, row tiles with slices parked and reduced, 128 rows a turn.
    fn lane3090(o: Ops, x: u64, xs: u64, q: QLinear, out: u64, rows: usize, sk: usize) !void {
        if (rows <= 16) return o.seg(x, xs, q, out, rows, sk);
        var r: usize = 0;
        while (r < rows) : (r += lane_rows) {
            try o.lane(x + r * q.k * 2, xs + r * (q.k / 64) * 4, q, out + r * q.n * 2, @min(lane_rows, rows - r), sk);
        }
    }

    fn seg(o: Ops, x: u64, xs: u64, q: QLinear, out: u64, rows: usize, sk: usize) !void {
        const tiles = (q.n + 63) / 64;
        const blocks = tiles * sk;
        if (sk > 1 and tiles > sync_counters) return error.TooManyTiles;
        var segs: Segs = std.mem.zeroes(Segs);
        segs.count = 1;
        segs.s[0] = .{ .w = q.w, .scales = q.s, .biases = q.b, .out = out, .part = if (sk > 1) o.k.part.ptr else 0, .n = int(q.n), .npad = int(q.npad), .sk = int(sk), .blocks = int(blocks), .sync = 0 };
        return o.segLaunch(x, xs, segs, blocks, sk, rows, q.k);
    }

    fn segLaunch(o: Ops, x: u64, xs: u64, segs: Segs, blocks: usize, sk: usize, rows: usize, k: usize) !void {
        // qmm_seg_cuda: past 4 blocks an SM, 3 stages; 2 or 4 slices side by side in a block
        const sh = shapeOf(&tune, @intCast(segs.s[0].n), k);
        const st: u32 = if (sh.st > 0) sh.st else if (tune.st > 0) tune.st else if (blocks > 4 * o.k.sms) 3 else 4;
        const g: u32 = if (tune.g > 0 and (tune.g == 1 or tune.g == sk)) tune.g else if (sk == 2 or sk == 4) @intCast(sk) else 1;
        const f = if (tune.probe > 0 and rows > 8 and g == 1) o.k.seg_probe[st - 3][tune.probe - 1] else o.k.seg[st - 3][if (rows <= 8) 0 else 1][if (g == 1) 0 else if (g == 2) 1 else 2];
        var a: cuda.Args = .{};
        a.add(x);
        a.add(xs);
        a.add(segs);
        a.add(o.k.sync.ptr);
        for ([_]usize{ rows, k, k }) |v| a.add(int(v));
        const smem = if (g > 1) g * st * seg_stage else @max(st * seg_stage, 4096);
        try cuda.launch.launch(f, .{ .grid = .{ .x = u(blocks / g) }, .block = .{ .x = 128 * g }, .shared = smem }, o.s, &a);
    }

    fn lane(o: Ops, x: u64, xs: u64, q: QLinear, out: u64, rows: usize, sk: usize) !void {
        const i: usize = if (rows <= 16) 0 else if (rows <= 32) 1 else 2;
        const bm: usize = @as(usize, 16) << @intCast(i);
        const rows_t = (rows + bm - 1) / bm;
        // a group's inputs stay near 12 MB of L2 while its blocks sweep the column tiles
        const group = @max(1, @min(rows_t, (12 << 20) / (bm * q.k * 2)));
        var a: cuda.Args = .{};
        for ([_]u64{ x, xs, q.w, q.s, q.b, out, if (sk > 1) o.k.part.ptr else 0 }) |v| a.add(v);
        for ([_]usize{ rows, q.n, q.k, sk, q.npad, q.k, group }) |v| a.add(int(v));
        try cuda.launch.launch(o.k.lane[i], .{ .grid = .{ .x = u(rows_t * ((q.n + 63) / 64)), .z = u(sk) }, .block = .{ .x = 128 }, .shared = lane_smem[i] }, o.s, &a);
        if (sk == 1) return;
        const total = rows * q.n;
        var r: cuda.Args = .{};
        r.add(o.k.part.ptr);
        r.add(out);
        r.add(@as(i64, @intCast(total)));
        r.add(int(sk));
        try cuda.launch.launch(o.k.reduce, .{ .grid = .{ .x = u((total + 255) / 256) }, .block = .{ .x = 256 } }, o.s, &r);
    }

    /// qmm_group's tile 2 with one K slice: a CTA a column tile.
    fn cluster(o: Ops, x: u64, xs: u64, q: QLinear, out: u64, rows: usize, sk: usize) !void {
        const tiles = (q.n + 63) / 64;
        var parts: Parts = std.mem.zeroes(Parts);
        parts.count = 1;
        parts.p[0] = .{ .w = q.w, .scales = q.s, .biases = q.b, .out = out, .n = int(q.n), .npad = int(q.npad), .sk = int(sk), .tiles = int(tiles), .first = 0 };
        const rows_t = (rows + 15) / 16;
        var a: cuda.Args = .{};
        a.add(x);
        a.add(xs);
        a.add(parts);
        for ([_]usize{ rows, q.k, q.k, rows_t, sk }) |v| a.add(int(v));
        const cfg: cuda.Config = .{
            .grid = .{ .x = u(rows_t * tiles * sk) },
            .block = .{ .x = 128 },
            .shared = group_smem,
            .cluster = if (sk > 1) .{ .x = u(sk) } else null,
            .pdl = o.k.gb10,
        };
        try cuda.launch.launch(o.k.group, cfg, o.s, &a);
    }

    /// qmm_group's dispatch on a GB10 past 16 rows: tile 3, 4 or 5, a cluster of `sk` CTAs a column tile.
    fn wideTile(o: Ops, x: u64, xs: u64, q: QLinear, out: u64, rows: usize, sk: usize) !void {
        const i: usize = if (rows <= 32) 0 else if (rows <= 64) 1 else 2;
        const t = wide[i];
        const tiles = (q.n + t.bn - 1) / t.bn;
        const rows_t = (rows + t.bm - 1) / t.bm;
        var parts: Parts = std.mem.zeroes(Parts);
        parts.count = 1;
        parts.p[0] = .{ .w = q.w, .scales = q.s, .biases = q.b, .out = out, .n = int(q.n), .npad = int(q.npad), .sk = int(sk), .tiles = int(tiles), .first = 0 };
        var a: cuda.Args = .{};
        a.add(x);
        a.add(xs);
        a.add(parts);
        for ([_]usize{ rows, q.k, q.k, rows_t, sk }) |v| a.add(int(v));
        const cfg: cuda.Config = .{
            .grid = .{ .x = u(rows_t * tiles * sk) },
            .block = .{ .x = t.threads },
            .shared = t.smem,
            .cluster = if (sk > 1) .{ .x = u(sk) } else null,
            .pdl = o.k.gb10,
        };
        try cuda.launch.launch(o.k.wide[i], cfg, o.s, &a);
    }

    /// qmm.prefill_matmul at prompt_tile 9: weights rounded once to bf16, one fp32 chain over K; bf16 out.
    pub fn prefillDense(o: Ops, x: u64, q: QLinear, out: u64, rows: usize) !void {
        const rows_t = (rows + 127) / 128;
        const band = (12 << 20) / (128 * q.k * 2);
        const group = @max(1, @min(rows_t, band));
        var a: cuda.Args = .{};
        for ([_]u64{ x, q.w, q.s, q.b, out }) |v| a.add(v);
        for ([_]usize{ rows, q.n, q.k, q.npad, q.k, group }) |v| a.add(int(v));
        try o.go(o.k.prefill_mm, .{ rows_t * ((q.n + 127) / 128), 1, 1 }, 128, prefill_mm_smem, &a);
    }

    /// gdn.tree (dispatch_tree): y from one `state`, or each stream's own by `table` and `starts`; `slots` 0: chains.
    pub fn gdnTree(o: Ops, q: u64, k: u64, v: u64, g: u64, beta: u64, state: u64, table: u64, starts: u64, plan: u64, y: u64, rows: usize, streams: usize, slots: usize, most: usize, h: Heads) !void {
        const variant = cuda.kernels.treeVariant(@intCast(slots), @intCast(streams), o.k.chip[0], o.k.chip[1], o.k.chip[2]);
        const index = for (cuda.kernels.gdn_variants, 0..) |x, n| {
            if (std.mem.eql(u8, x.symbol, variant.symbol)) break n;
        } else unreachable;
        var a: cuda.Args = .{};
        for ([_]u64{ q, k, v, g, beta, state, table, starts, plan }) |x| a.add(x);
        a.add(int(rows));
        a.add(y);
        for ([_]usize{ h.hk, h.hv, h.dv }) |x| a.add(int(x));
        a.add(Pending{});
        a.add(@as(u64, 0));
        a.add(@as(u64, 0));
        a.add(h.dv % variant.r == 0 and v % (2 * @as(u64, @min(variant.r, 8))) == 0);
        const shared = 16 * @as(usize, variant.warps) * variant.slots * variant.r * 32 + (if (variant.chain) 0 else 4 * 3 * most);
        try o.go(o.k.trees[index], .{ (h.dv + variant.r * variant.warps - 1) / (variant.r * variant.warps), h.hv, streams }, 32 * variant.warps, @intCast(shared), &a);
    }

    /// gdn.replay in place: each stream's accepted rows stepped into the states the table names.
    pub fn gdnReplay(o: Ops, table: u64, layers: usize, streams: usize, rows: u64, row_stride: usize, counts: u64, count_stride: usize, h: Heads) !void {
        var a: cuda.Args = .{};
        a.add(table);
        a.add(int(layers));
        a.add(rows);
        a.add(int(row_stride));
        a.add(counts);
        a.add(int(count_stride));
        a.add(@as(u64, 0));
        for ([_]usize{ h.hk, h.hv, h.dv }) |x| a.add(int(x));
        try o.go(o.k.replay, .{ (h.dv + 31) / 32, h.hv, layers * streams }, 128, 0, &a);
    }

    /// gdn.chain: a prompt chunk's rows from `state` (read only) to `last`, chunk-invariant bits.
    pub fn gdnPrompt(o: Ops, q: u64, k: u64, v: u64, g: u64, beta: u64, state: u64, last: u64, y: u64, rows: usize, h: Heads) !void {
        var a: cuda.Args = .{};
        for ([_]u64{ q, k, v, g, beta, state, last, y }) |x| a.add(x);
        for ([_]usize{ rows, h.hk, h.hv }) |x| a.add(int(x));
        const r = o.k.prompt_rows;
        try o.go(o.k.prompt_chain, .{ h.hv, 128 / r, 1 }, u(2 * r), 0, &a);
    }

    /// prefill_attention (head dim 256, two query heads a block): q (rows, heads, 256) at positions p0..
    pub fn prefillAttention(o: Ops, q: u64, kc: u64, vc: u64, out: u64, p0: usize, rows: usize, heads: usize, kv_heads: usize, scale: f32) !void {
        const g = heads / kv_heads;
        const hpc = 2;
        var a: cuda.Args = .{};
        for ([_]u64{ q, kc, vc, out }) |v| a.add(v);
        for ([_]usize{ p0, rows, heads, kv_heads, g }) |v| a.add(int(v));
        a.add(scale);
        const block_rows = 16 * (8 / hpc);
        try o.go(o.k.pattn, .{ (rows + block_rows - 1) / block_rows, kv_heads * (g / hpc), 1 }, 256, pattn_smem, &a);
    }

    /// prefillAttention over a 4-bit cache (kv4.cu): q rotated, the output in rotated space.
    pub fn prefillAttention4(o: Ops, q: u64, kc: u64, vc: u64, out: u64, p0: usize, rows: usize, heads: usize, kv_heads: usize, scale: f32) !void {
        const g = heads / kv_heads;
        const hpc = 2;
        var a: cuda.Args = .{};
        for ([_]u64{ q, kc, vc, out }) |v| a.add(v);
        for ([_]usize{ p0, rows, heads, kv_heads, g }) |v| a.add(int(v));
        a.add(scale);
        const block_rows = 16 * (8 / hpc);
        try o.go(o.k.pattn4, .{ (rows + block_rows - 1) / block_rows, kv_heads * (g / hpc), 1 }, 256, pattn_smem, &a);
    }

    /// kvq.rotate: H32 over every 32 bf16 values of x into y (y may be x).
    pub fn kv4Rotate(o: Ops, x: u64, y: u64, values: usize) !void {
        const groups = values / 32;
        if (groups == 0) return;
        var a: cuda.Args = .{};
        a.add(x);
        a.add(y);
        a.add(@as(c_longlong, @intCast(groups)));
        try o.go(o.k.kv4_rotate, .{ (groups + 127) / 128, 1, 1 }, 128, 0, &a);
    }

    /// kvq.store: (row-heads, 256) bf16 rows packed into 144-byte 4-bit rows.
    pub fn kv4Quant(o: Ops, x: u64, out: u64, row_heads: usize) !void {
        const groups = row_heads * 8;
        if (groups == 0) return;
        var a: cuda.Args = .{};
        a.add(x);
        a.add(out);
        a.add(@as(c_longlong, @intCast(groups)));
        try o.go(o.k.kv4_quant, .{ (groups + 127) / 128, 1, 1 }, 128, 0, &a);
    }

    /// attention._attention4 over rotated queries: shared 512-key chunks, each row's tail, the chunks merged in order
    /// (still rotated). `kn`, `vn`: the window's packed keys and values; `base`: the origin the byte offsets count from.
    pub fn attention4(o: Ops, q: u64, base: u64, offs: u64, streams: u64, items: u64, n_items: usize, rows_of: u64, paths: u64, depths: u64, kn: u64, vn: u64, po: u64, pm: u64, pl: u64, out: u64, w: usize, heads: usize, kv_heads: usize, max_nodes: usize, chunk: usize) !void {
        const g = heads / kv_heads;
        const scale: f32 = @floatCast(std.math.pow(f64, 256.0, -0.5));
        var a: cuda.Args = .{};
        for ([_]u64{ q, base, offs, streams, items }) |v| a.add(v);
        a.add(int(n_items));
        for ([_]u64{ rows_of, paths, depths, kn, vn, po, pm, pl }) |v| a.add(v);
        for ([_]usize{ w, heads, kv_heads, g }) |v| a.add(int(v));
        a.add(scale);
        if (n_items > 0) {
            // 16-row tiles a block: all of them if they fit three, else two (the fork's choice; bits the same)
            const tiles = (w * g + 15) / 16;
            const mt: usize = if (tiles <= 3) tiles else 2;
            try o.go(o.k.attn4_shared[mt - 1], .{ kv_heads, (n_items + mt - 1) / mt * mt, 1 }, 128, attn4Smem(mt), &a);
        }
        try o.go(o.k.attn4_tail, .{ w, kv_heads, 1 + (max_nodes + chunk - 1) / chunk }, 128, attn4Smem(1), &a);
        var m: cuda.Args = .{};
        for ([_]u64{ po, pm, pl, out, streams, rows_of }) |v| m.add(v);
        m.add(int(w));
        m.add(int(heads));
        try o.go(o.k.kv4_merge, .{ w, heads, 1 }, 256, 0, &m);
    }

    /// torch.argmax(rows of bf16 logits) as int32: the first maximum, a NaN first; `part`: (rows, 64) 16-byte slots.
    pub fn argmax(o: Ops, logits: u64, vocab: usize, ld: usize, out: u64, rows: usize, part: u64) !void {
        const slices = @min(argmax_slices, (vocab + 4095) / 4096);
        const per = ((vocab + slices - 1) / slices + 7) / 8 * 8;
        var a: cuda.Args = .{};
        for ([_]u64{ logits, part }) |v| a.add(v);
        for ([_]usize{ vocab, ld, per }) |v| a.add(@as(i64, @intCast(v)));
        try o.go(o.k.argmax[0], .{ rows, (vocab + per - 1) / per, 1 }, 256, 0, &a);
        var b: cuda.Args = .{};
        for ([_]u64{ part, out }) |v| b.add(v);
        b.add(@as(i32, @intCast((vocab + per - 1) / per)));
        try o.go(o.k.argmax[1], .{ rows, 1, 1 }, 32, 0, &b);
    }

    /// The torch-op replacements on this stream (topk and casts for sampled draws).
    pub fn torch(o: Ops) torch_ops.Torch {
        return .{ .f = &o.k.torch, .s = o.s };
    }

    /// qwen_ops.cu's tf_copy_batch: `count` (dst, src, bytes) items from the device table `items`, one launch.
    pub fn copies(o: Ops, items: u64, count: usize) !void {
        if (count == 0) return;
        var a: cuda.Args = .{};
        a.add(items);
        try o.go(o.k.copy_batch, .{ count, 4, 1 }, 256, 0, &a);
    }

    pub fn copy(o: Ops, dst: u64, src: u64, bytes: usize) !void {
        if (bytes == 0) return;
        try o.k.d.check(o.k.d.api.cuMemcpyDtoDAsync_v2(dst, src, bytes, o.s.handle), "cuMemcpyDtoDAsync");
    }

    pub fn fill32(o: Ops, dst: u64, value: u32, words: usize) !void {
        try o.k.d.check(o.k.d.api.cuMemsetD32Async(dst, value, words, o.s.handle), "cuMemsetD32Async");
    }

    pub fn upload(o: Ops, dst: u64, bytes: []const u8) !void {
        if (bytes.len == 0) return;
        try o.k.d.check(o.k.d.api.cuMemcpyHtoDAsync_v2(dst, bytes.ptr, bytes.len, o.s.handle), "cuMemcpyHtoDAsync");
    }

    pub fn download(o: Ops, dst: []u8, src: u64) !void {
        if (dst.len == 0) return;
        try o.k.d.check(o.k.d.api.cuMemcpyDtoHAsync_v2(dst.ptr, src, dst.len, o.s.handle), "cuMemcpyDtoHAsync");
    }
};

test "split_k follows the 27B's shapes" {
    try std.testing.expectEqual(@as(usize, 2), splitK(10240, 5120)); // qkv
    try std.testing.expectEqual(@as(usize, 2), splitK(6144, 5120)); // z
    try std.testing.expectEqual(@as(usize, 8), splitK(48, 5120)); // b, a
    try std.testing.expectEqual(@as(usize, 1), splitK(17408, 5120)); // gate, up
    try std.testing.expectEqual(@as(usize, 4), splitK(5120, 17408)); // down
    try std.testing.expectEqual(@as(usize, 4), splitK(5120, 6144)); // out_proj, o_proj
    try std.testing.expectEqual(@as(usize, 1), splitK(12288, 5120)); // q with its gate
    try std.testing.expectEqual(@as(usize, 8), splitK(1024, 5120)); // k, v
    try std.testing.expectEqual(@as(usize, 1), splitK(248320, 5120)); // head
}
