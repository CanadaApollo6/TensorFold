//! qwen3_5_moe's MTP drafts (mtp.Head, decode.mtp_decode): chains from the target's final normed rows, verified exactly.

const std = @import("std");
const cuda = @import("cuda");
const c = @import("shape.zig");
const st = @import("state.zig");
const wts = @import("weights.zig");
const moe = @import("moe.zig");
const fw = @import("forward.zig");
const lone = @import("lone.zig");
const Engine = @import("engine.zig").Engine;
const Forward = fw.Forward;
const Part = fw.Part;

const bf = 2;
/// mtp.DRAFT_WINDOW: the head attends to the last 4096 keys before a call's rows (drafts only; the verify decides).
pub const window = 4096;
/// The most drafts a round (qwen3_5_moe.cuda.DEPTH), each verified (decode.FOLLOW: no confidence cut).
pub const depth = 3;

/// The head's attention keys and values by position, and its rows: [held | a prompt chunk's normed rows], drafts.
pub const Cache = struct {
    buf: cuda.DeviceBuffer,
    keys: u64,
    values: u64,
    capacity: usize,
    pos: usize = 0, // slots below it absorbed; the rest scratch
    rows: u64, // (1 + prompt_rows, hidden): the held row, then the target's normed rows
    out: u64, // (prompt_rows, hidden): the head's normed outputs
    out_xs: u64,
    carry: u64, // (max_rows, hidden): the kept rows' states, absorbed with their next tokens next round
    carry_tokens: [st.max_rows]u32 = undefined,
    carry_n: usize = 0,

    pub fn init(e: *Engine, capacity: usize) !Cache {
        const g = e.w.g;
        const kv = g.kvRow();
        const row = g.hidden * bf;
        const R = e.prompt_rows;
        const sizes = [_]usize{ capacity * kv, capacity * kv, (1 + R) * row, R * row, R * g.hidden / 64 * 4, st.max_rows * row };
        var total: usize = 0;
        for (sizes) |n| total += std.mem.alignForward(usize, n, 256);
        var x: Cache = undefined;
        x.buf = try cuda.DeviceBuffer.alloc(e.ctx.d, total);
        var at = x.buf.ptr;
        var ptrs: [sizes.len]u64 = undefined;
        for (sizes, 0..) |n, i| {
            ptrs[i] = at;
            at += std.mem.alignForward(usize, n, 256);
        }
        x = .{ .buf = x.buf, .keys = ptrs[0], .values = ptrs[1], .capacity = capacity, .rows = ptrs[2], .out = ptrs[3], .out_xs = ptrs[4], .carry = ptrs[5] };
        return x;
    }

    pub fn deinit(x: *Cache) void {
        x.buf.free();
    }
};

/// Head.forward: rows at positions [p0, p0 + n), each (state at t, token t + 1), attending to the window before p0
/// and to each other; their normed outputs to cache.out and their keys at slots [p0, p0 + n).
pub fn forward(f: *Forward, x: *Cache, states: u64, tokens: []const u32, p0: usize) !void {
    const m = f.w.mtp orelse return error.NoMtpHead;
    const s = f.s;
    const g = f.w.g;
    const W = tokens.len;
    const d = g.hidden;
    if (W < 1 or W > s.rows) return error.ChunkTooLong;
    if (p0 + W > x.capacity) return error.PromptTooLong;
    {
        const host = try f.gpa.alloc(i32, 2 * W);
        defer f.gpa.free(host);
        for (tokens, 0..) |t, r| {
            host[r] = @intCast(t);
            host[W + r] = @intCast(p0 + r);
        }
        try f.ops.upload(s.ids, std.mem.sliceAsBytes(host[0..W]));
        try f.ops.upload(s.pos, std.mem.sliceAsBytes(host[W..]));
        try f.ops.s.synchronize(); // the host copy must outlive the async upload
    }
    try f.t.embed(s.ids, f.w.embed.w, f.w.embed.s, f.w.embed.b, s.x, W, d);
    try f.t.addRmsnorm(s.x, null, m.norm_e, s.x, s.y, s.xs, W, d, c.eps);
    try f.t.addRmsnorm(states, null, m.norm_h, states, s.gated, s.gated_xs, W, d, c.eps);
    try f.ops.dense(s.y, s.xs, m.fc_e, s.r, W);
    try f.ops.dense(s.gated, s.gated_xs, m.fc_h, s.pending, W);
    try f.t.addRmsnorm(s.r, s.pending, m.input_norm, s.h, s.y, s.xs, W, d, c.eps);
    const a = m.attn;
    const kv = g.kvRow();
    try f.ops.denseMany(s.y, s.xs, &.{ a.q, a.k, a.v }, &.{ s.qg, s.key, x.values + p0 * kv }, W);
    const shape: @import("triton.zig").Attn = .{ .heads = g.query_heads, .kv_heads = g.kv_heads, .dim = c.head_dim, .half = c.rotary_dim / 2 };
    try f.t.attnPrep(s.qg, s.key, a.q_norm, a.k_norm, s.pos, f.w.inv_freq, s.q_rot, x.keys + p0 * kv, W, shape, c.eps);
    // keys written first: each row attends to [p0 - window, p0 + its row], prefill_attention's causal rows
    const base = p0 -| window;
    const scale: f32 = @floatCast(std.math.pow(f64, c.head_dim, -0.5));
    try f.ops.prefillAttention(s.q_rot, x.keys + base * kv, x.values + base * kv, s.attn, p0 - base, W, g.query_heads, g.kv_heads, scale);
    try f.t.gateMul(s.attn, s.qg, s.gated, s.gated_xs, W, g.query_heads, c.head_dim);
    try f.ops.dense(s.gated, s.gated_xs, a.o, s.r, W);
    try f.t.addRmsnorm(s.h, s.r, m.post_norm, s.x, s.y, s.xs, W, d, c.eps);
    const sh: moe.Shape = .{ .hidden = d, .width = g.moe_inter, .experts = g.experts, .top_k = g.top_k };
    const mo: moe.Moe = .{ .k = &f.ops.k.moe.?, .s = f.ops.s, .sh = sh, .sc = s.moe };
    try mo.run(m.ffn, s.y, s.pending, W, W > s.window_rows);
    try f.t.addRmsnorm(s.x, s.pending, m.norm, s.h, x.out, x.out_xs, W, d, c.eps);
}

/// The draft head's greedy pick from cache.out's row `row` (Head.logits, argmax).
fn draft(f: *Forward, x: *const Cache, row: usize) !u32 {
    const d = f.w.g.hidden;
    try f.ops.dense(x.out + row * d * bf, x.out_xs + row * d / 64 * 4, f.w.head, f.s.logits, 1);
    var pick: [1]u32 = undefined;
    try f.picks(1, &pick);
    return pick[0];
}

/// decode.prefill with the head: the prompt in even chunks, every row but the last absorbed, the last held.
pub fn prefill(e: *Engine, seq: *st.Seq, x: *Cache, prompt: []const u32) !void {
    if (prompt.len == 0) return error.EmptyPrompt;
    if (prompt.len >= seq.capacity or prompt.len >= x.capacity) return error.PromptTooLong;
    try seq.reset(e.ops(), e.w.g);
    const row = e.w.g.hidden * bf;
    var f = e.forward();
    f.normed = x.rows + row;
    const n = (prompt.len + e.prompt_rows - 1) / e.prompt_rows;
    for (0..n) |j| {
        const b = @import("engine.zig").chunkBounds(0, prompt.len, e.prompt_rows, j);
        try f.chunk(prompt[b[0]..b[1]], seq, j == n - 1);
        const held: usize = if (j > 0) 1 else 0;
        const rows = held + b[1] - b[0] - 1; // [held | chunk rows] but the last, each with its next prompt token
        if (rows > 0) try forward(&f, x, x.rows + (1 - held) * row, prompt[b[0] - held + 1 ..][0..rows], b[0] - held);
        try f.ops.copy(x.rows, x.rows + (b[1] - b[0]) * row, row);
    }
    x.pos = prompt.len - 1;
    x.carry_n = 0;
}

/// mtp_decode, greedy: each round absorbs the carry (its last row drafts first), proposes a copy from the context or
/// a chain of `depth` drafts, verifies them in one window and keeps the target's path; `pending` is the first token.
pub fn decode(e: *Engine, seq: *st.Seq, x: *Cache, prompt: []const u32, pending: u32, count: usize, out: *std.ArrayList(u32), stats: *lone.Stats) !void {
    const gpa = e.gpa;
    const row = e.w.g.hidden * bf;
    var context: std.ArrayList(u32) = .empty;
    defer context.deinit(gpa);
    try context.appendSlice(gpa, prompt);
    try out.append(gpa, pending);
    try context.append(gpa, pending);
    var copies: lone.CopyIndex = .{ .gpa = gpa };
    defer copies.deinit();
    const copy_rows = 16; // decode.COPY_ROWS
    var f = e.forward();
    // the held prompt row and the first token are the first carry
    try f.ops.copy(x.carry, x.rows, row);
    x.carry_tokens[0] = pending;
    x.carry_n = 1;
    var tokens: [st.max_rows]u32 = undefined;
    var sampled: [st.max_rows]u32 = undefined;
    var path: [st.max_rows]u32 = undefined;
    while (out.items.len < count and !e.w.isEos(out.items[out.items.len - 1])) {
        tokens[0] = out.items[out.items.len - 1];
        var rows: usize = 1;
        try forward(&f, x, x.carry, x.carry_tokens[0..x.carry_n], x.pos);
        x.pos += x.carry_n;
        const copied = try copies.propose(context.items, copy_rows - 1);
        if (copied.len > 0) {
            @memcpy(tokens[1..][0..copied.len], copied);
            rows += copied.len;
        } else {
            tokens[1] = try draft(&f, x, x.carry_n - 1);
            rows = 2;
            const last = x.out + (x.carry_n - 1) * row;
            if (x.carry_n > 1) try f.ops.copy(x.out, last, row);
            while (rows < depth + 1) : (rows += 1) {
                try forward(&f, x, x.out, tokens[rows - 1 .. rows], x.pos + rows - 2);
                tokens[rows] = try draft(&f, x, 0);
            }
        }
        const part = [_]Part{.{ .seq = seq, .tokens = tokens[0..rows] }};
        try f.round(&part);
        try f.picks(rows, &sampled);
        // a chain: row r + 1 stays while its token is the one the target drew after row r
        var n: usize = 1;
        path[0] = 0;
        var terminal = sampled[0];
        while (out.items.len + n < count and !e.w.isEos(terminal) and n < rows and tokens[n] == terminal) {
            path[n] = @intCast(n);
            terminal = sampled[n];
            n += 1;
        }
        try f.commit(&part, &.{path[0..n]});
        try f.ops.copy(x.carry, f.s.last_h, n * row);
        for (1..n) |r| x.carry_tokens[r - 1] = tokens[r];
        x.carry_tokens[n - 1] = terminal;
        x.carry_n = n;
        try out.appendSlice(gpa, x.carry_tokens[0..n]);
        try context.appendSlice(gpa, x.carry_tokens[0..n]);
        stats.rounds += 1;
        stats.drafted += rows - 1;
        stats.accepted += n - 1;
    }
}
