//! The 27B's forward on CUDA: verify rounds (tree_forward, multi_tree_forward), their commit, and prompt chunks.

const std = @import("std");
const c = @import("shape.zig");
const kern = @import("kernels.zig");
const tri = @import("triton.zig");
const st = @import("state.zig");
const wts = @import("weights.zig");
const rnd = @import("round.zig");

const bf = 2;
pub const Part = rnd.Part;
/// The target layers DFlash2 reads (dflash_config.target_layer_ids).
pub const tap_layers = [_]usize{ 5, 19, 33, 47, 61 };

/// TF_QWEN27_PROFILE's breakdown of verify rounds: synced ms by operation (a sync after each, so a few us extra each).
pub const Op = enum { stage, embed, norm, gdn_proj, gdn_pre, gdn_tree, gated_norm, gdn_out, attn_proj, attn_prep, attention, gate_mul, attn_out, gate_up, swiglu, down, tap, head, empty };
pub var prof_on: bool = false;
pub var prompt_dense_rows: ?usize = null;
pub var prof_ms: [std.enums.values(Op).len]f64 = @splat(0);
var prof_mark: f64 = 0;

pub fn nowMs() f64 {
    var ts: std.c.timespec = undefined;
    _ = std.c.clock_gettime(.MONOTONIC, &ts);
    return @as(f64, @floatFromInt(ts.sec)) * 1e3 + @as(f64, @floatFromInt(ts.nsec)) / 1e6;
}

pub const Forward = struct {
    gpa: std.mem.Allocator,
    w: *const wts.Weights,
    ops: kern.Ops,
    t: tri.Tri,
    s: *st.Scratch,
    dump: ?Dump = null, // a debugging copy of every norm's (h, y) rows, as capture.py's hook writes them
    taps: ?u64 = null, // rows of the drafter's five layer taps (5 x hidden bf16), written by round and chunk

    pub const Dump = struct { io: std.Io, dir: []const u8, n: usize = 0, rows: usize = 0 };

    fn g(f: *const Forward) c.Geometry {
        return f.w.g;
    }

    fn gdnShape(f: *const Forward) tri.Gdn {
        const g_ = f.g();
        return .{ .c = g_.convDim(), .kh = g_.linear_k_heads, .vh = g_.linear_v_heads, .dk = c.linear_dim, .keep = c.conv_taps - 1 };
    }

    fn heads(f: *const Forward) kern.Heads {
        return .{ .hk = f.g().linear_k_heads, .hv = f.g().linear_v_heads, .dv = c.linear_dim };
    }

    fn attnShape(f: *const Forward) tri.Attn {
        return .{ .heads = f.g().query_heads, .kv_heads = f.g().kv_heads, .dim = c.head_dim, .half = c.rotary_dim / 2 };
    }

    fn upload32(f: *Forward, dst: u64, values: []const i32) !void {
        try f.ops.upload(dst, std.mem.sliceAsBytes(values));
        try f.ops.s.synchronize(); // the host copy must outlive the async upload
    }

/// TF_QWEN27_PROMPT_DENSE_ROWS: prompt chunks up to this many rows take the decode matmuls (16-row slices read the
    /// weights once each) instead of prefill_mm, whose 128-row tiles leave a short chunk ~10x slower. 0: off.
    fn promptDense(f: *Forward, x: u64, xs: u64, q: kern.QLinear, out: u64, rows: usize) !void {
        if (prompt_dense_rows == null) prompt_dense_rows = if (std.c.getenv("TF_QWEN27_PROMPT_DENSE_ROWS")) |v| std.fmt.parseInt(usize, std.mem.span(v), 10) catch 0 else 0;
        if (rows <= prompt_dense_rows.?) return f.ops.dense(x, xs, q, out, rows);
        return f.ops.prefillDense(x, q, out, rows);
    }

    fn tick(f: *Forward, op: Op) !void {
        if (!prof_on) return;
        try f.ops.s.synchronize();
        const t = nowMs();
        prof_ms[@intFromEnum(op)] += t - prof_mark;
        prof_mark = t;
    }

    /// add_rmsnorm into the scratch's other residual buffer; returns the residual stream's new home.
    fn norm(f: *Forward, x: u64, r: ?u64, weight: u64, rows: usize) !u64 {
        const out = if (r == null) x else if (x == f.s.x) f.s.h else f.s.x;
        try f.t.addRmsnorm(x, r, weight, out, f.s.y, f.s.xs, rows, f.g().hidden, c.eps);
        if (f.dump) |*d| try f.saveRows(d, out, f.s.y, rows);
        return out;
    }

    /// Layer `i`'s tap, when the drafter reads it: (x + pending) of the call's rows into the taps buffer.
    fn tapLayer(f: *Forward, i: usize, x: u64, rows: usize) !void {
        const out = f.taps orelse return;
        const j = std.mem.indexOfScalar(usize, &tap_layers, i) orelse return;
        const h = f.g().hidden;
        try f.ops.tap(x, f.s.pending, out, rows, h, tap_layers.len * h, j * h);
    }

    fn saveRows(f: *Forward, d: *Dump, h: u64, y: u64, rows: usize) !void {
        const bytes = try f.gpa.alloc(u8, rows * f.g().hidden * bf);
        defer f.gpa.free(bytes);
        for ([_]u64{ h, y }, [_][]const u8{ "h", "y" }) |src, name| {
            try f.ops.download(bytes, src);
            try f.ops.s.synchronize();
            var buf: [256]u8 = undefined;
            const path = try std.fmt.bufPrint(&buf, "{s}/{d:0>3}_{s}.bin", .{ d.dir, d.n, name });
            try std.Io.Dir.cwd().writeFile(d.io, .{ .sub_path = path, .data = bytes });
        }
        d.n += 1;
    }

    /// The window of one stream at its position (tree_forward): logits (rows, vocab) bf16 in the scratch, records kept.
    pub fn window(f: *Forward, tokens: []const u32, seq: *st.Seq) !void {
        return f.round(&.{.{ .seq = seq, .tokens = tokens }});
    }

    /// Several streams' chain windows in one forward (multi_tree_forward; one stream: tree_forward), rows in order.
    pub fn round(f: *Forward, parts: []const Part) !void {
        const s = f.s;
        const g_ = f.g();
        const w = f.w;
        if (prof_on) {
            try f.ops.s.synchronize();
            prof_mark = nowMs();
        }
        const r = try rnd.stage(f.gpa, f.ops, f.t, s, g_, parts);
        const W = r.rows;
        try f.tick(.stage);
        try f.t.embed(r.ids, w.embed.w, w.embed.s, w.embed.b, s.x, W, g_.hidden);
        try f.tick(.embed);
        var x = s.x;
        var pending: ?u64 = null;
        var lin: usize = 0;
        var att: usize = 0;
        for (w.layers, 0..) |layer, i| {
            x = try f.norm(x, pending, layer.input_norm, W);
            try f.tick(.norm);
            switch (layer.mixer) {
                .delta => |d| {
                    const rec = s.records[i];
                    try f.ops.denseMany(s.y, s.xs, &.{ d.qkv, d.z, d.b, d.a }, &.{ rec.qkv, s.z, s.b, s.a }, W);
                    try f.tick(.gdn_proj);
                    const sid: ?u64 = if (r.multi) r.sids else null;
                    try f.t.gdnPre(rec.qkv, r.conv(parts, s, g_, i, lin), d.conv, r.windows, sid, s.a, s.b, d.a_log, d.dt_bias, rec.q, rec.k, rec.v, rec.g, rec.beta, W, f.gdnShape());
                    try f.tick(.gdn_pre);
                    const state = if (r.multi) 0 else parts[0].seq.rec[i];
                    const table = if (r.multi) r.table(lin) else 0;
                    try f.ops.gdnTree(rec.q, rec.k, rec.v, rec.g, rec.beta, state, table, if (r.multi) r.starts else 0, r.plan, s.yr, W, r.streams, r.slots, r.most, f.heads());
                    try f.tick(.gdn_tree);
                    try f.t.gatedNorm(s.yr, s.z, d.norm, s.gated, s.gated_xs, W, g_.linear_v_heads, c.linear_dim, c.eps);
                    try f.tick(.gated_norm);
                    try f.ops.dense(s.gated, s.gated_xs, d.out, s.r, W);
                    try f.tick(.gdn_out);
                    lin += 1;
                },
                .attention => |a| {
                    const rec = s.attn_records[i];
                    try f.ops.denseMany(s.y, s.xs, &.{ a.q, a.k, a.v }, &.{ s.qg, s.key, if (c.kv4()) s.prompt_values else rec.v }, W);
                    try f.tick(.attn_proj);
                    if (c.kv4()) {
                        // the window's keys and values packed as the cache keeps them, attended in rotated space
                        try f.t.attnPrep(s.qg, s.key, a.q_norm, a.k_norm, r.pos, w.inv_freq, s.q_rot, s.prompt_keys, W, f.attnShape(), c.eps);
                        try f.ops.kv4Quant(s.prompt_keys, rec.k, W * g_.kv_heads);
                        try f.ops.kv4Quant(s.prompt_values, rec.v, W * g_.kv_heads);
                        try f.ops.kv4Rotate(s.q_rot, s.q_rot, W * g_.qInner());
                        const p = r.attn;
                        try f.ops.attention4(s.q_rot, s.origin, r.offsets(i, att), p.streams, p.items, p.n_items, p.rows, p.paths, p.depths, rec.k, rec.v, s.part_o, s.part_m, s.part_l, s.attn, W, g_.query_heads, g_.kv_heads, tri.max_nodes, tri.chunk);
                        try f.ops.kv4Rotate(s.attn, s.attn, W * g_.qInner());
                    } else {
                        try f.t.attnPrep(s.qg, s.key, a.q_norm, a.k_norm, r.pos, w.inv_freq, s.q_rot, rec.k, W, f.attnShape(), c.eps);
                        try f.t.attention(s.q_rot, rec.k, rec.v, s.origin, r.offsets(i, att), r.attn, .{ .o = s.part_o, .m = s.part_m, .l = s.part_l }, s.attn, f.attnShape());
                    }
                    try f.tick(.attention);
                    try f.t.gateMul(s.attn, s.qg, s.gated, s.gated_xs, W, g_.query_heads, c.head_dim);
                    try f.tick(.gate_mul);
                    try f.ops.dense(s.gated, s.gated_xs, a.o, s.r, W);
                    try f.tick(.attn_out);
                    att += 1;
                },
            }
            x = try f.norm(x, s.r, layer.post_norm, W);
            try f.tick(.norm);
            try f.ops.denseMany(s.y, s.xs, &.{ layer.gate, layer.up }, &.{ s.gate, s.up }, W);
            try f.tick(.gate_up);
            try f.t.swiglu(s.gate, s.up, s.act, s.act_xs, W, g_.intermediate);
            try f.tick(.swiglu);
            try f.ops.dense(s.act, s.act_xs, layer.down, s.pending, W);
            try f.tick(.down);
            pending = s.pending;
            try f.tapLayer(i, x, W);
            try f.tick(.tap);
            try f.tick(.empty); // the sync and clock alone: subtract it from each op above
        }
        const spare = if (x == s.x) s.h else s.x;
        try f.t.addRmsnorm(x, pending, w.norm, spare, s.last_h, s.last_xs, W, g_.hidden, c.eps);
        if (f.dump) |*d| try f.saveRows(d, spare, s.last_h, W);
        try f.ops.dense(s.last_h, s.last_xs, w.head, s.logits, W);
        try f.tick(.head);
    }

    /// commit_streams after a round over `parts`: each stream keeps the window rows of its accepted path.
    pub fn commit(f: *Forward, parts: []const Part, paths: []const []const u32) !void {
        const s = f.s;
        const g_ = f.g();
        const S = parts.len;
        var width: usize = 0;
        for (parts, paths) |p, path| {
            if (path.len < 1 or path.len > p.tokens.len) return error.InvalidQwenCommit;
            for (path) |row| if (row >= p.tokens.len) return error.InvalidQwenCommit;
            width += p.tokens.len;
        }
        // gdn.replay_table: each DeltaNet layer's k, v, g, beta, then each stream's state of each layer
        var table: std.ArrayList(u64) = .empty;
        defer table.deinit(f.gpa);
        var layers: usize = 0;
        for (0..g_.layers) |i| if (c.linear(i)) {
            const rc = s.records[i];
            try table.appendSlice(f.gpa, &.{ rc.k, rc.v, rc.g, rc.beta });
            layers += 1;
        };
        for (parts) |p| for (0..g_.layers) |i| if (c.linear(i)) try table.append(f.gpa, p.seq.rec[i]);
        try f.ops.upload(s.replay_table, std.mem.sliceAsBytes(table.items));
        // path_indices: each stream's accepted rows (round rows) padded to the round's width, then its count
        var rows: std.ArrayList(i32) = .empty;
        defer rows.deinit(f.gpa);
        var base: usize = 0;
        for (parts, paths) |p, path| {
            for (0..width) |j| try rows.append(f.gpa, if (j < path.len) @intCast(base + path[j]) else 0);
            try rows.append(f.gpa, @intCast(path.len));
            base += p.tokens.len;
        }
        try f.ops.upload(s.replay_rows, std.mem.sliceAsBytes(rows.items));
        try f.ops.s.synchronize();
        try f.ops.gdnReplay(s.replay_table, layers, S, s.replay_rows, width + 1, s.replay_rows + width * 4, width + 1, f.heads());
        const row = g_.convDim() * bf;
        const keep = c.conv_taps - 1;
        const kv = g_.kvRow();
        // conv rows: the last three of [committed | accepted rows]; old ones saved first, then written with the rest
        var save: std.ArrayList(kern.Copy) = .empty;
        defer save.deinit(f.gpa);
        var write: std.ArrayList(kern.Copy) = .empty;
        defer write.deinit(f.gpa);
        base = 0;
        for (parts, paths, 0..) |p, path, k| {
            const n = path.len;
            var lin: usize = 0;
            for (0..g_.layers) |i| {
                if (c.linear(i)) {
                    const old = s.conv_old + (k * g_.layers + lin) * keep * row;
                    const stay = keep -| n; // committed rows that stay: old[n..3) go to new[0..3-n)
                    if (stay > 0) {
                        try save.append(f.gpa, .{ .dst = old, .src = p.seq.conv[i] + n * row, .bytes = stay * row });
                        try write.append(f.gpa, .{ .dst = p.seq.conv[i], .src = old, .bytes = stay * row });
                    }
                    for (path[n - (keep - stay) ..], stay..) |acc, j| try write.append(f.gpa, .{ .dst = p.seq.conv[i] + j * row, .src = s.records[i].qkv + (base + acc) * row, .bytes = row });
                    lin += 1;
                } else {
                    for (path, 0..) |acc, j| {
                        const at = (p.seq.pos + j) * kv;
                        try write.append(f.gpa, .{ .dst = p.seq.keys[i] + at, .src = s.attn_records[i].k + (base + acc) * kv, .bytes = kv });
                        try write.append(f.gpa, .{ .dst = p.seq.values[i] + at, .src = s.attn_records[i].v + (base + acc) * kv, .bytes = kv });
                    }
                }
            }
            base += p.tokens.len;
            p.seq.pos += n;
        }
        const second = s.copy_items + st.copy_items * @sizeOf(kern.Copy);
        try f.ops.upload(s.copy_items, std.mem.sliceAsBytes(save.items));
        try f.ops.upload(second, std.mem.sliceAsBytes(write.items));
        try f.ops.copies(s.copy_items, save.items.len);
        try f.ops.copies(second, write.items.len);
        try f.ops.s.synchronize(); // the tables' host copies go when this returns
    }

    /// prefill_chunk: `ids` committed at the stream's position; `last` leaves the last row normed, with group_sums.
    pub fn chunk(f: *Forward, ids: []const u32, seq: *st.Seq, last: bool) !void {
        const W = ids.len;
        const s = f.s;
        const g_ = f.g();
        const w = f.w;
        if (W < 1 or W > s.rows) return error.ChunkTooLong;
        if (seq.pos + W > seq.capacity) return error.PromptTooLong;
        const p0 = seq.pos;
        if (prof_on) {
            try f.ops.s.synchronize();
            prof_mark = nowMs();
        }
        {
            const host = try f.gpa.alloc(i32, W * (2 + c.conv_taps));
            defer f.gpa.free(host);
            for (ids, 0..) |tok, r| {
                host[r] = @intCast(tok);
                host[W + r] = @intCast(p0 + r);
                for (0..c.conv_taps) |j| host[2 * W + r * c.conv_taps + j] = @intCast(r + j);
            }
            try f.upload32(s.ids, host[0..W]);
            try f.upload32(s.pos, host[W .. 2 * W]);
            try f.upload32(s.windows, host[2 * W ..]);
        }
        try f.tick(.stage);
        try f.t.embed(s.ids, w.embed.w, w.embed.s, w.embed.b, s.x, W, g_.hidden);
        try f.tick(.embed);
        var x = s.x;
        var pending: ?u64 = null;
        const kv = g_.kvRow();
        const row = g_.convDim() * bf;
        const keep = c.conv_taps - 1;
        for (w.layers, 0..) |layer, i| {
            x = try f.norm(x, pending, layer.input_norm, W);
            try f.tick(.norm);
            switch (layer.mixer) {
                .delta => |d| {
                    try f.promptDense(s.y, s.xs, d.qkv, s.qkv, W);
                    try f.promptDense(s.y, s.xs, d.z, s.z, W);
                    try f.promptDense(s.y, s.xs, d.b, s.b, W);
                    try f.promptDense(s.y, s.xs, d.a, s.a, W);
                    try f.tick(.gdn_proj);
                    try f.t.gdnPre(s.qkv, seq.conv[i], d.conv, s.windows, null, s.a, s.b, d.a_log, d.dt_bias, s.gq, s.gk, s.gv, s.gg, s.gbeta, W, f.gdnShape());
                    try f.tick(.gdn_pre);
                    // each thread reads its state rows before it writes them: the chain may end in place
                    try f.ops.gdnPrompt(s.gq, s.gk, s.gv, s.gg, s.gbeta, seq.rec[i], seq.rec[i], s.yr, W, f.heads());
                    try f.tick(.gdn_tree);
                    if (W >= keep) {
                        try f.ops.copy(seq.conv[i], s.qkv + (W - keep) * row, keep * row);
                    } else {
                        try f.ops.copy(s.conv_tmp, seq.conv[i], keep * row);
                        try f.ops.copy(s.conv_tmp + keep * row, s.qkv, W * row);
                        try f.ops.copy(seq.conv[i], s.conv_tmp + W * row, keep * row);
                    }
                    try f.t.gatedNorm(s.yr, s.z, d.norm, s.gated, s.gated_xs, W, g_.linear_v_heads, c.linear_dim, c.eps);
                    try f.tick(.gated_norm);
                    try f.promptDense(s.gated, s.gated_xs, d.out, s.r, W);
                    try f.tick(.gdn_out);
                },
                .attention => |a| {
                    try f.promptDense(s.y, s.xs, a.q, s.qg, W);
                    try f.promptDense(s.y, s.xs, a.k, s.key, W);
                    try f.tick(.attn_proj);
                    const scale: f32 = @floatCast(std.math.pow(f64, c.head_dim, -0.5));
                    if (c.kv4()) {
                        // the chunk's rows packed into the cache first: its rows read them back as later ones will
                        try f.promptDense(s.y, s.xs, a.v, s.prompt_values, W);
                        try f.t.attnPrep(s.qg, s.key, a.q_norm, a.k_norm, s.pos, w.inv_freq, s.q_rot, s.prompt_keys, W, f.attnShape(), c.eps);
                        try f.ops.kv4Quant(s.prompt_keys, seq.keys[i] + p0 * kv, W * g_.kv_heads);
                        try f.ops.kv4Quant(s.prompt_values, seq.values[i] + p0 * kv, W * g_.kv_heads);
                        try f.ops.kv4Rotate(s.q_rot, s.q_rot, W * g_.qInner());
                        try f.ops.prefillAttention4(s.q_rot, seq.keys[i], seq.values[i], s.attn, p0, W, g_.query_heads, g_.kv_heads, scale);
                        try f.ops.kv4Rotate(s.attn, s.attn, W * g_.qInner());
                    } else {
                        try f.promptDense(s.y, s.xs, a.v, seq.values[i] + p0 * kv, W);
                        try f.t.attnPrep(s.qg, s.key, a.q_norm, a.k_norm, s.pos, w.inv_freq, s.q_rot, seq.keys[i] + p0 * kv, W, f.attnShape(), c.eps);
                        try f.ops.prefillAttention(s.q_rot, seq.keys[i], seq.values[i], s.attn, p0, W, g_.query_heads, g_.kv_heads, scale);
                    }
                    try f.tick(.attention);
                    try f.t.gateMul(s.attn, s.qg, s.gated, s.gated_xs, W, g_.query_heads, c.head_dim);
                    try f.tick(.gate_mul);
                    try f.promptDense(s.gated, s.gated_xs, a.o, s.r, W);
                    try f.tick(.attn_out);
                },
            }
            x = try f.norm(x, s.r, layer.post_norm, W);
            try f.tick(.norm);
            try f.promptDense(s.y, s.xs, layer.gate, s.gate, W);
            try f.promptDense(s.y, s.xs, layer.up, s.up, W);
            try f.tick(.gate_up);
            try f.t.swiglu(s.gate, s.up, s.act, s.act_xs, W, g_.intermediate);
            try f.tick(.swiglu);
            try f.promptDense(s.act, s.act_xs, layer.down, s.pending, W);
            try f.tick(.down);
            pending = s.pending;
            try f.tapLayer(i, x, W);
            try f.tick(.tap);
        }
        seq.pos = p0 + W;
        if (!last) return;
        const at = (W - 1) * g_.hidden * bf;
        const spare = if (x == s.x) s.h else s.x;
        try f.t.addRmsnorm(x + at, pending.? + at, w.norm, spare, seq.head_in, s.last_xs, 1, g_.hidden, c.eps);
        try f.t.groupSums(seq.head_in, seq.head_xs, g_.hidden, 1, g_.hidden);
    }

    /// One stream's share of a batched prompt pass: its ids, committed at its position; `last` normalizes its last row.
    pub const Piece = struct { ids: []const u32, seq: *st.Seq, last: bool };

    /// prefill_rows: several streams' prompt chunks in one pass (each stream's conv, DeltaNet and attention its own).
    pub fn chunks(f: *Forward, pieces: []const Piece) !void {
        const s = f.s;
        const g_ = f.g();
        const w = f.w;
        const S = pieces.len;
        if (S == 0 or S > s.streams()) return error.InvalidQwenWindows;
        var starts: [st.max_streams + 1]usize = undefined;
        starts[0] = 0;
        for (pieces, 0..) |p, k| {
            if (p.ids.len == 0) return error.ChunkTooLong;
            if (p.seq.pos + p.ids.len > p.seq.capacity) return error.PromptTooLong;
            starts[k + 1] = starts[k] + p.ids.len;
        }
        const W = starts[S];
        if (W > s.rows) return error.ChunkTooLong;
        const keep = c.conv_taps - 1;
        const kv = g_.kvRow();
        const row = g_.convDim() * bf;
        // a 4-bit cache takes the prompt's rows packed: keys into the spent projection buffer, values beside them
        const packed_keys = if (c.kv4()) s.key else s.prompt_keys;
        const packed_values = if (c.kv4()) s.value else s.prompt_values;
        {
            // ids, positions, each row's stream and its conv window: [the streams' kept rows; every prompt row]
            const host = try f.gpa.alloc(i32, W * (3 + c.conv_taps));
            defer f.gpa.free(host);
            for (pieces, 0..) |p, k| for (p.ids, 0..) |tok, r| {
                const at = starts[k] + r;
                host[at] = @intCast(tok);
                host[W + at] = @intCast(p.seq.pos + r);
                host[2 * W + at] = @intCast(k);
                for (0..c.conv_taps) |j| host[3 * W + at * c.conv_taps + j] = @intCast(if (r + j < keep) r + j else r + j + starts[k]);
            };
            try f.upload32(s.ids, host[0..W]);
            try f.upload32(s.pos, host[W .. 2 * W]);
            try f.upload32(s.sids, host[2 * W .. 3 * W]);
            try f.upload32(s.windows, host[3 * W ..]);
        }
        // copies the pass needs: every stream's conv rows stacked by DeltaNet layer, then each layer's writes
        var items: std.ArrayList(kern.Copy) = .empty;
        defer items.deinit(f.gpa);
        var lin_n: usize = 0;
        for (0..g_.layers) |i| if (c.linear(i)) {
            const cat = s.conv_cat + lin_n * s.streams() * keep * row;
            for (pieces, 0..) |p, k| try items.append(f.gpa, .{ .dst = cat + k * keep * row, .src = p.seq.conv[i], .bytes = keep * row });
            lin_n += 1;
        };
        const staged = items.items.len;
        var at_layer: [256]usize = undefined; // each layer's first write, then the end
        lin_n = 0;
        for (0..g_.layers) |i| {
            at_layer[i] = items.items.len;
            if (c.linear(i)) {
                // the last `keep` rows of [kept rows; the stream's prompt rows]
                const cat = s.conv_cat + lin_n * s.streams() * keep * row;
                for (pieces, 0..) |p, k| {
                    const n = p.ids.len;
                    const o = starts[k];
                    if (n >= keep) {
                        try items.append(f.gpa, .{ .dst = p.seq.conv[i], .src = s.qkv + (o + n - keep) * row, .bytes = keep * row });
                    } else {
                        try items.append(f.gpa, .{ .dst = p.seq.conv[i], .src = cat + (k * keep + n) * row, .bytes = (keep - n) * row });
                        try items.append(f.gpa, .{ .dst = p.seq.conv[i] + (keep - n) * row, .src = s.qkv + o * row, .bytes = n * row });
                    }
                }
                lin_n += 1;
            } else {
                for (pieces, 0..) |p, k| {
                    const o = starts[k];
                    const n = p.ids.len;
                    try items.append(f.gpa, .{ .dst = p.seq.keys[i] + p.seq.pos * kv, .src = packed_keys + o * kv, .bytes = n * kv });
                    try items.append(f.gpa, .{ .dst = p.seq.values[i] + p.seq.pos * kv, .src = packed_values + o * kv, .bytes = n * kv });
                }
            }
        }
        at_layer[g_.layers] = items.items.len;
        if (items.items.len > 2 * st.copy_items) return error.InvalidQwenWindows;
        try f.ops.upload(s.copy_items, std.mem.sliceAsBytes(items.items));
        try f.ops.copies(s.copy_items, staged);
        const item = @sizeOf(kern.Copy);
        try f.t.embed(s.ids, w.embed.w, w.embed.s, w.embed.b, s.x, W, g_.hidden);
        var x = s.x;
        var pending: ?u64 = null;
        lin_n = 0;
        for (w.layers, 0..) |layer, i| {
            x = try f.norm(x, pending, layer.input_norm, W);
            switch (layer.mixer) {
                .delta => |d| {
                    try f.ops.prefillDense(s.y, d.qkv, s.qkv, W);
                    try f.ops.prefillDense(s.y, d.z, s.z, W);
                    try f.ops.prefillDense(s.y, d.b, s.b, W);
                    try f.ops.prefillDense(s.y, d.a, s.a, W);
                    const cat = s.conv_cat + lin_n * s.streams() * keep * row;
                    try f.t.gdnPre(s.qkv, cat, d.conv, s.windows, s.sids, s.a, s.b, d.a_log, d.dt_bias, s.gq, s.gk, s.gv, s.gg, s.gbeta, W, f.gdnShape());
                    for (pieces, 0..) |p, k| {
                        const o = starts[k];
                        // each thread reads its state rows before it writes them: the chain may end in place
                        try f.ops.gdnPrompt(s.gq + o * g_.kInner() * bf, s.gk + o * g_.kInner() * bf, s.gv + o * g_.vInner() * bf, s.gg + o * g_.linear_v_heads * 4, s.gbeta + o * g_.linear_v_heads * 4, p.seq.rec[i], p.seq.rec[i], s.yr + o * g_.vInner() * bf, p.ids.len, f.heads());
                    }
                    try f.ops.copies(s.copy_items + at_layer[i] * item, at_layer[i + 1] - at_layer[i]);
                    try f.t.gatedNorm(s.yr, s.z, d.norm, s.gated, s.gated_xs, W, g_.linear_v_heads, c.linear_dim, c.eps);
                    try f.ops.prefillDense(s.gated, d.out, s.r, W);
                    lin_n += 1;
                },
                .attention => |a| {
                    try f.ops.prefillDense(s.y, a.q, s.qg, W);
                    try f.ops.prefillDense(s.y, a.k, s.key, W);
                    try f.ops.prefillDense(s.y, a.v, s.prompt_values, W);
                    try f.t.attnPrep(s.qg, s.key, a.q_norm, a.k_norm, s.pos, w.inv_freq, s.q_rot, s.prompt_keys, W, f.attnShape(), c.eps);
                    if (c.kv4()) {
                        try f.ops.kv4Quant(s.prompt_keys, packed_keys, W * g_.kv_heads);
                        try f.ops.kv4Quant(s.prompt_values, packed_values, W * g_.kv_heads);
                        try f.ops.kv4Rotate(s.q_rot, s.q_rot, W * g_.qInner());
                    }
                    try f.ops.copies(s.copy_items + at_layer[i] * item, at_layer[i + 1] - at_layer[i]);
                    const scale: f32 = @floatCast(std.math.pow(f64, c.head_dim, -0.5));
                    const qrow = g_.query_heads * c.head_dim * bf;
                    const attend = if (c.kv4()) &kern.Ops.prefillAttention4 else &kern.Ops.prefillAttention;
                    for (pieces, 0..) |p, k| try attend(f.ops, s.q_rot + starts[k] * qrow, p.seq.keys[i], p.seq.values[i], s.attn + starts[k] * qrow, p.seq.pos, p.ids.len, g_.query_heads, g_.kv_heads, scale);
                    if (c.kv4()) try f.ops.kv4Rotate(s.attn, s.attn, W * g_.qInner());
                    try f.t.gateMul(s.attn, s.qg, s.gated, s.gated_xs, W, g_.query_heads, c.head_dim);
                    try f.ops.prefillDense(s.gated, a.o, s.r, W);
                },
            }
            x = try f.norm(x, s.r, layer.post_norm, W);
            try f.ops.prefillDense(s.y, layer.gate, s.gate, W);
            try f.ops.prefillDense(s.y, layer.up, s.up, W);
            try f.t.swiglu(s.gate, s.up, s.act, s.act_xs, W, g_.intermediate);
            try f.ops.prefillDense(s.act, layer.down, s.pending, W);
            pending = s.pending;
            try f.tapLayer(i, x, W);
        }
        const spare = if (x == s.x) s.h else s.x;
        for (pieces, 0..) |p, k| {
            p.seq.pos += p.ids.len;
            if (!p.last) continue;
            const at = (starts[k + 1] - 1) * g_.hidden * bf;
            try f.t.addRmsnorm(x + at, pending.? + at, w.norm, spare, p.seq.head_in, s.last_xs, 1, g_.hidden, c.eps);
            try f.t.groupSums(p.seq.head_in, p.seq.head_xs, g_.hidden, 1, g_.hidden);
        }
        try f.ops.s.synchronize(); // the copy table's host list goes when this returns
    }

    /// The head over the stream's last prompt row: logits (1, vocab) in the scratch.
    pub fn head(f: *Forward, seq: *const st.Seq) !void {
        try f.ops.dense(seq.head_in, seq.head_xs, f.w.head, f.s.logits, 1);
    }

    /// Greedy picks of the scratch's first `rows` logits rows (torch.argmax, the first maximum).
    pub fn picks(f: *Forward, rows: usize, out: []u32) !void {
        try f.ops.argmax(f.s.logits, c.vocab, c.vocab, f.s.picks, rows);
        try f.ops.download(std.mem.sliceAsBytes(out[0..rows]), f.s.picks);
        try f.ops.s.synchronize();
    }
};
