//! LFM2-MoE's forward on CUDA: rows of one stream at its position, committed (a decode window or a prompt chunk).
//! A layer: h += mixer(rms(h)) (short conv or GQA attention), then h += mlp(rms(h)) (dense SwiGLU or routed experts).

const std = @import("std");
const cuda = @import("cuda");
const c = @import("shape.zig");
const kern = @import("kernels.zig");
const st = @import("state.zig");
const wts = @import("weights.zig");
const moe = @import("moe.zig");

pub const Forward = struct {
    gpa: std.mem.Allocator,
    w: *const wts.Weights,
    ops: kern.Ops,
    s: *st.Scratch,
    dump: ?*std.ArrayList(u16) = null, // debugging: each norm's row 0 (bf16), as ref.py --dump takes them

    fn upload32(f: *Forward, dst: u64, values: []const i32) !void {
        try f.ops.b().upload(dst, std.mem.sliceAsBytes(values));
        try f.ops.s.synchronize(); // the host copy must outlive the async upload
    }

    /// add_rmsnorm into the residual's other home; returns the residual's new home.
    fn norm(f: *Forward, x: u64, r: ?u64, weight: u64, rows: usize) !u64 {
        const out = if (r == null) x else if (x == f.s.x) f.s.h else f.s.x; // r null: h stays x (not copied)
        try f.ops.addRms(x, r, weight, out, f.s.y, f.s.xs, rows, f.w.g.hidden, f.w.g.eps);
        try f.save(f.s.y);
        return out;
    }

    fn save(f: *Forward, row: u64) !void {
        const d = f.dump orelse return;
        const n = d.items.len;
        try d.resize(f.gpa, n + f.w.g.hidden);
        try f.ops.b().download(std.mem.sliceAsBytes(d.items[n..]), row);
        try f.ops.s.synchronize();
    }

    /// `ids` at the stream's position, committed: its conv rows and its keys and values stay. `head`: the rows'
    /// logits (a decode window, at most max_rows); else only the last row's normed input stays (a prompt chunk's end).
    pub fn step(f: *Forward, ids: []const u32, seq: *st.Seq, head: bool) !void {
        const W = ids.len;
        if (W < 1 or W > f.s.rows or (head and W > st.max_rows)) return error.ChunkTooLong;
        if (seq.pos + W > seq.capacity) return error.PromptTooLong;
        try f.stage(ids, seq.pos);
        try f.body(W, seq, head);
        seq.pos += W;
    }

    /// The rows' ids and positions from `pos` on, to the device.
    pub fn stage(f: *Forward, ids: []const u32, pos: usize) !void {
        const W = ids.len;
        var host: [2 * 4096]i32 = undefined;
        if (W > 4096) return error.ChunkTooLong;
        for (ids, 0..) |tok, r| {
            host[r] = @intCast(tok);
            host[W + r] = @intCast(pos + r);
        }
        try f.upload32(f.s.ids, host[0..W]);
        try f.upload32(f.s.pos, host[W .. 2 * W]);
    }

    /// The layers over W staged rows, launches only (no host reads): a decode step's graph captures it.
    pub fn body(f: *Forward, W: usize, seq: *st.Seq, head: bool) !void {
        const s = f.s;
        const g = f.w.g;
        const w = f.w;
        const H = g.hidden;
        const o = f.ops;
        const prompt = W > st.max_rows;
        try o.k.base.glue.embed.affine4(o.s, s.ids, w.embed.w, w.embed.s, w.embed.b, s.x, W, H);
        var x = s.x;
        var pending: ?u64 = null;
        for (w.layers, 0..) |layer, i| {
            x = try f.norm(x, pending, layer.op_norm, W);
            switch (layer.mixer) {
                .conv => |cv| {
                    try o.mm(s.y, s.xs, cv.in, s.bcx, W);
                    try o.convRows(s.bcx, cv.w, seq.conv[i], null, s.co, W, H, W);
                    try o.groupSums(s.co, H, s.co_xs, W, H);
                    try o.mm(s.co, s.co_xs, cv.out, s.r, W);
                },
                .attention => |a| {
                    if (prompt) {
                        for ([_]kern.QLinear{ a.q, a.k, a.v }, [_]u64{ s.q, s.k, s.v }) |q, out| try o.mm(s.y, s.xs, q, out, W);
                    } else try o.b().denseMany(s.y, s.xs, &.{ a.q, a.k, a.v }, &.{ s.q, s.k, s.v }, W);
                    const qd = g.qInner();
                    const kd = g.kvInner();
                    try o.k.base.glue.qk.rowsFrom(o.s, .{ .q = s.q, .q_ld = @intCast(qd), .q_hs = c.head_dim, .k = s.k, .k_ld = @intCast(kd), .k_hs = c.head_dim, .v = s.v, .v_ld = @intCast(kd), .v_hs = c.head_dim }, cuda.qk_rows.Heads{ .h = g.heads, .hk = g.kv_heads, .dim = c.head_dim }, .{ .norms = a.q_norm, .norm_k = a.k_norm, .norms_bf16 = true, .inv = w.inv_freq, .rot = c.head_dim, .eps = g.eps }, s.pos, null, .{ .q = s.qr, .k = s.kr, .v = s.vr, .kc = seq.keys[i], .vc = seq.values[i], .len = seq.capacity }, W);
                    try o.attn(s.qr, seq.keys[i], seq.values[i], s.pos, g.kv_heads, s.attn_part, s.attn_sync, s.attn, W);
                    try o.groupSums(s.attn, qd, s.attn_xs, W, qd);
                    try o.mm(s.attn, s.attn_xs, a.o, s.r, W);
                },
            }
            x = try f.norm(x, s.r, layer.ffn_norm, W);
            switch (layer.ffn) {
                .dense => |dn| {
                    if (prompt) {
                        try o.mm(s.y, s.xs, dn.gate, s.gate, W);
                        try o.mm(s.y, s.xs, dn.up, s.up, W);
                    } else try o.b().denseMany(s.y, s.xs, &.{ dn.gate, dn.up }, &.{ s.gate, s.up }, W);
                    try o.k.base.glue.act.swiglu(o.s, s.gate, g.intermediate, s.up, g.intermediate, s.act, s.act_xs, W, g.intermediate);
                    try o.mm(s.act, s.act_xs, dn.down, s.pending, W);
                },
                .moe => |l| {
                    const m: moe.Moe = .{ .k = &o.k.moe, .splitk = &o.k.splitk, .route_fn = o.k.route, .s = o.s, .sh = .{ .hidden = H, .width = g.moe_inter, .experts = g.experts, .top_k = g.top_k }, .sc = s.moe };
                    try m.run(l, s.y, s.pending, W, prompt);
                },
            }
            pending = s.pending;
        }
        const spare = if (x == s.x) s.h else s.x;
        if (head) {
            try o.addRms(x, pending, w.norm, spare, s.last_h, s.last_xs, W, H, g.eps);
            try f.save(s.last_h);
            return o.b().dense(s.last_h, s.last_xs, w.head, s.logits, W);
        }
        const at = (W - 1) * H * 2;
        try o.addRms(x + at, pending.? + at, w.norm, spare, seq.head_in, seq.head_xs, 1, H, g.eps);
    }

    /// The head over the stream's last prompt row: logits (1, vocab) in the scratch.
    pub fn headLast(f: *Forward, seq: *const st.Seq) !void {
        try f.ops.b().dense(seq.head_in, seq.head_xs, f.w.head, f.s.logits, 1);
    }

    /// A graphed decode step: the staged row through the head, its pick staged as the next row (Ops.next).
    pub fn decodeStep(f: *Forward, seq: *st.Seq) !void {
        try f.body(1, seq, true);
        const v = f.w.g.vocab;
        try f.ops.b().argmax(f.s.logits, v, v, f.s.picks, 1, f.s.pick_part);
        try f.ops.next(f.s.picks, f.s.ids, f.s.pos, f.s.log);
    }

    /// Greedy picks of the scratch's first `n` logits rows (the first maximum).
    pub fn picks(f: *Forward, n: usize, out: []u32) !void {
        const v = f.w.g.vocab;
        try f.ops.b().argmax(f.s.logits, v, v, f.s.picks, n, f.s.pick_part);
        try f.ops.b().download(std.mem.sliceAsBytes(out[0..n]), f.s.picks);
        try f.ops.s.synchronize();
    }
};
