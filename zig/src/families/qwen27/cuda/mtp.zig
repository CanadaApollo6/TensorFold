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
/// The head's acceptance at depth 1, 2, 3 given the ones before it, until a stream has its own (R1s's check prompts).
pub const draft_prior = [_]f64{ 0.85, 0.71, 0.65 };

/// A head call of at most this many rows attends with qwen_ops' split-key kernels, longer ones with prefill_attention.
const few_rows = 16;
/// tf_window_part's keys a split: 16 or more, about 2 x 82 blocks over the key heads (a 3090's SMs, twice).
fn splitKeys(keys: usize, kv_heads: usize) usize {
    return @max(16, (keys * kv_heads + 163) / 164 + 15) / 16 * 16;
}
const max_splits = 164 + 1;

/// The rows a head call reads and writes, shared by the streams: [held | a prompt chunk's normed rows], outputs.
pub const Rows = struct {
    buf: cuda.DeviceBuffer,
    rows: u64, // (1 + prompt_rows, hidden): the held row, then the target's normed rows
    out: u64, // (prompt_rows, hidden): the head's normed outputs
    out_xs: u64,
    part_o: u64, // (splits, few_rows, heads, 256) fp32: tf_window_part's outputs
    part_ml: u64, // (splits, few_rows, heads, 2) fp32: their max and sum
    win_part: cuda.Function,
    win_merge: cuda.Function,
    pick: cuda.Function, // tf_draft_pick: a draft's token into the next step's id and the chain's drafts
    drafts: u64, // (max_rows) u32: the chain's drafts, read once a chain
    map: u64, // (vocab.len) u32: the head's column j is token map[j]; 0 when it scores every token

    /// `ops`: the qwen_ops module; `vocab`: the head's tokens (Mtp.ids), whose device copy the caller uploads to map.
    pub fn init(d: *const cuda.Driver, g: c.Geometry, prompt_rows: usize, ops: cuda.Module, vocab: []const u32) !Rows {
        const row = g.hidden * bf;
        const parts = max_splits * few_rows * g.query_heads;
        const at = [_]usize{ 0, (1 + prompt_rows) * row, (1 + 2 * prompt_rows) * row, (1 + 2 * prompt_rows) * row + prompt_rows * g.hidden / 64 * 4 };
        const tail = at[3] + parts * (c.head_dim + 2) * 4;
        var buf = try cuda.DeviceBuffer.alloc(d, tail + (st.max_rows + vocab.len) * 4);
        errdefer buf.free();
        return .{
            .pick = try ops.function("tf_draft_pick"),
            .drafts = buf.ptr + tail,
            .map = if (vocab.len > 0) buf.ptr + tail + st.max_rows * 4 else 0,
            .buf = buf,
            .rows = buf.ptr + at[0],
            .out = buf.ptr + at[1],
            .out_xs = buf.ptr + at[2],
            .part_o = buf.ptr + at[3],
            .part_ml = buf.ptr + at[3] + parts * c.head_dim * 4,
            .win_part = try ops.function("tf_window_part"),
            .win_merge = try ops.function("tf_window_merge"),
        };
    }

    pub fn deinit(r: *Rows) void {
        r.buf.free();
    }
};

/// A stream's head: its attention keys and values from position `origin` on, the prompt's held row and its drafts.
pub const Cache = struct {
    buf: cuda.DeviceBuffer,
    keys: u64,
    values: u64,
    capacity: usize,
    origin: usize = 0, // the position in slot 0: a full cache slides its last window of keys to the front
    pos: usize = 0, // slots below it absorbed; the rest scratch
    carry: u64, // (1, hidden): the prompt's last row, absorbed with the first token
    drafts: [st.max_rows]u32 = undefined,
    held: usize = 0,

    /// Device bytes a stream's head takes for `capacity` positions.
    pub fn bytes(g: c.Geometry, capacity: usize) usize {
        return 2 * capacity * g.kvRow() + g.hidden * bf;
    }

    pub fn init(d: *const cuda.Driver, g: c.Geometry, capacity: usize) !Cache {
        const kv = capacity * g.kvRow();
        const buf = try cuda.DeviceBuffer.alloc(d, bytes(g, capacity));
        return .{ .buf = buf, .keys = buf.ptr, .values = buf.ptr + kv, .capacity = capacity, .carry = buf.ptr + 2 * kv };
    }

    /// A head's positions for a `context`-token stream: two windows and a call's rows, so slides never overlap.
    pub fn slots(context: usize, call_rows: usize) usize {
        return @min(context, 2 * window + call_rows);
    }

    pub fn deinit(x: *Cache) void {
        x.buf.free();
    }

    /// The slot of position p0 with `n` rows to write there, first sliding the window before p0 to slot 0 if full.
    fn slot(x: *Cache, f: *Forward, p0: usize, n: usize) !usize {
        if (p0 < x.origin + @min(p0, window)) return error.InvalidQwenWindows;
        if (p0 + n - x.origin <= x.capacity) return p0 - x.origin;
        const keep = @min(p0, window);
        const from = p0 - keep - x.origin;
        if (from < keep or keep + n > x.capacity) return error.PromptTooLong;
        const kv = f.w.g.kvRow();
        try f.ops.copy(x.keys, x.keys + from * kv, keep * kv);
        try f.ops.copy(x.values, x.values + from * kv, keep * kv);
        x.origin = p0 - keep;
        return keep;
    }
};

/// Head.forward: rows at positions [p0, p0 + n), each (state at t, token t + 1), attending to the window before p0
/// and to each other; their normed outputs to cache.out and their keys at slots [p0, p0 + n).
pub fn forward(f: *Forward, x: *Cache, r: *const Rows, states: u64, tokens: []const u32, p0: usize) !void {
    return step(f, x, r, states, tokens, tokens.len, p0);
}

/// forward with `tokens` null: the W ids already in the scratch's ids (a chained draft's, from tf_draft_pick).
fn step(f: *Forward, x: *Cache, r: *const Rows, states: u64, tokens: ?[]const u32, W: usize, p0: usize) !void {
    const m = f.w.mtp orelse return error.NoMtpHead;
    const s = f.s;
    const g = f.w.g;
    const d = g.hidden;
    if (W < 1 or W > s.rows) return error.ChunkTooLong;
    const at = try x.slot(f, p0, W);
    {
        // pageable host memory: the async upload stages it before returning, so no wait for the device
        const host = try f.gpa.alloc(i32, 2 * W);
        defer f.gpa.free(host);
        for (0..W) |i| host[W + i] = @intCast(p0 + i);
        if (tokens) |ts| {
            for (ts, 0..) |t, i| host[i] = @intCast(t);
            try f.ops.upload(s.ids, std.mem.sliceAsBytes(host[0..W]));
        }
        try f.ops.upload(s.pos, std.mem.sliceAsBytes(host[W..]));
    }
    try f.t.embed(s.ids, f.w.embed.w, f.w.embed.s, f.w.embed.b, s.x, W, d);
    try f.t.addRmsnorm(s.x, null, m.norm_e, s.x, s.y, s.xs, W, d, c.eps);
    try f.t.addRmsnorm(states, null, m.norm_h, states, s.gated, s.gated_xs, W, d, c.eps);
    try f.ops.dense(s.y, s.xs, m.fc_e, s.r, W);
    try f.ops.dense(s.gated, s.gated_xs, m.fc_h, s.pending, W);
    try f.t.addRmsnorm(s.r, s.pending, m.input_norm, s.h, s.y, s.xs, W, d, c.eps);
    const a = m.attn;
    const kv = g.kvRow();
    try f.ops.denseMany(s.y, s.xs, &.{ a.q, a.k, a.v }, &.{ s.qg, s.key, x.values + at * kv }, W);
    const shape: @import("triton.zig").Attn = .{ .heads = g.query_heads, .kv_heads = g.kv_heads, .dim = c.head_dim, .half = c.rotary_dim / 2 };
    try f.t.attnPrep(s.qg, s.key, a.q_norm, a.k_norm, s.pos, f.w.inv_freq, s.q_rot, x.keys + at * kv, W, shape, c.eps);
    // keys written first: each row attends to [p0 - window, p0 + its row], prefill_attention's causal rows
    const base = at - @min(p0, window);
    const scale: f32 = @floatCast(std.math.pow(f64, c.head_dim, -0.5));
    if (W <= few_rows and c.head_dim == 256) {
        const per = splitKeys(at + W - base, g.kv_heads);
        const splits = (at + W - base + per - 1) / per;
        if (splits > max_splits) return error.InvalidQwenWindows;
        var pa: cuda.Args = .{};
        for ([_]u64{ s.q_rot, x.keys, x.values, r.part_o, r.part_ml }) |v| pa.add(v);
        for ([_]usize{ at, base, W, g.query_heads, g.kv_heads, per }) |v| pa.add(@as(c_int, @intCast(v)));
        pa.add(scale);
        try cuda.launch.launch(r.win_part, .{ .grid = .{ .x = @intCast(g.kv_heads), .y = @intCast(splits) }, .block = .{ .x = 256 } }, f.ops.s, &pa);
        var b: cuda.Args = .{};
        for ([_]u64{ r.part_o, r.part_ml, s.attn }) |v| b.add(v);
        for ([_]usize{ splits, W * g.query_heads }) |v| b.add(@as(c_int, @intCast(v)));
        try cuda.launch.launch(r.win_merge, .{ .grid = .{ .x = @intCast(W * g.query_heads) }, .block = .{ .x = 256 } }, f.ops.s, &b);
    } else try f.ops.prefillAttention(s.q_rot, x.keys + base * kv, x.values + base * kv, s.attn, at - base, W, g.query_heads, g.kv_heads, scale);
    try f.t.gateMul(s.attn, s.qg, s.gated, s.gated_xs, W, g.query_heads, c.head_dim);
    try f.ops.dense(s.gated, s.gated_xs, a.o, s.r, W);
    try f.t.addRmsnorm(s.h, s.r, m.post_norm, s.x, s.y, s.xs, W, d, c.eps);
    const sh: moe.Shape = .{ .hidden = d, .width = g.moe_inter, .experts = g.experts, .top_k = g.top_k };
    const mo: moe.Moe = .{ .k = &f.ops.k.moe.?, .s = f.ops.s, .sh = sh, .sc = s.moe };
    try mo.run(m.ffn, s.y, s.pending, W, W > s.window_rows);
    try f.t.addRmsnorm(s.x, s.pending, m.norm, s.h, r.out, r.out_xs, W, d, c.eps);
}

/// The draft head's greedy pick from the outputs' row `row` (Head.logits, argmax; column j is token map[j]), left on
/// the device as the next chained step's id and the chain's draft `slot`.
fn draft(f: *Forward, r: *const Rows, row: usize, slot: usize) !void {
    const m = f.w.mtp.?;
    const d = f.w.g.hidden;
    try f.ops.dense(r.out + row * d * bf, r.out_xs + row * d / 64 * 4, m.head, f.s.logits, 1);
    try f.ops.argmax(f.s.logits, m.head.n, m.head.n, f.s.picks, 1);
    var a: cuda.Args = .{};
    for ([_]u64{ f.s.picks, r.map, f.s.ids, r.drafts }) |v| a.add(v);
    a.add(@as(c_int, @intCast(slot)));
    try cuda.launch.launch(r.pick, .{ .grid = .{ .x = 1 }, .block = .{ .x = 32 } }, f.ops.s, &a);
}

/// Absorbs kept rows (their states at `states`, each with the token after it in `follow`) at the head's position,
/// then holds `count` chained drafts from the last one (decode.mtp_round's chain).
pub fn chain(f: *Forward, x: *Cache, r: *const Rows, states: u64, follow: []const u32, count: usize) !void {
    if (count > st.max_rows - 1) return error.InvalidQwenWindows;
    const row = f.w.g.hidden * bf;
    try forward(f, x, r, states, follow, x.pos);
    x.pos += follow.len;
    x.held = 0;
    if (count == 0) return;
    try draft(f, r, follow.len - 1, 0);
    if (follow.len > 1) try f.ops.copy(r.out, r.out + (follow.len - 1) * row, row);
    x.held = 1;
    while (x.held < count) : (x.held += 1) {
        try step(f, x, r, r.out, null, 1, x.pos + x.held - 1);
        try draft(f, r, 0, x.held);
    }
    try f.ops.download(std.mem.sliceAsBytes(x.drafts[0..count]), r.drafts);
    try f.ops.s.synchronize();
}

/// decode.prefill with the head: the prompt in even chunks, every row but the last absorbed, the last held.
pub fn prefill(e: *Engine, seq: *st.Seq, x: *Cache, prompt: []const u32) !void {
    if (prompt.len == 0) return error.EmptyPrompt;
    if (prompt.len >= seq.capacity) return error.PromptTooLong;
    const r = &(e.mtp_rows orelse return error.NoMtpHead);
    try seq.reset(e.ops(), e.w.g);
    x.origin = 0;
    const row = e.w.g.hidden * bf;
    var f = e.forward();
    f.normed = r.rows + row;
    const n = (prompt.len + e.prompt_rows - 1) / e.prompt_rows;
    for (0..n) |j| {
        const b = @import("engine.zig").chunkBounds(0, prompt.len, e.prompt_rows, j);
        try f.chunk(prompt[b[0]..b[1]], seq, j == n - 1);
        const held: usize = if (j > 0) 1 else 0;
        const rows = held + b[1] - b[0] - 1; // [held | chunk rows] but the last, each with its next prompt token
        if (rows > 0) try forward(&f, x, r, r.rows + (1 - held) * row, prompt[b[0] - held + 1 ..][0..rows], b[0] - held);
        try f.ops.copy(r.rows, r.rows + (b[1] - b[0]) * row, row);
    }
    try f.ops.copy(x.carry, r.rows, row);
    x.pos = prompt.len - 1;
    x.held = 0;
}

/// mtp_decode, greedy: each round absorbs the last round's kept rows (the prompt's held row first), proposes a copy
/// from the context or a chain of `depth` drafts, verifies them in one window and keeps the target's path.
pub fn decode(e: *Engine, seq: *st.Seq, x: *Cache, prompt: []const u32, pending: u32, count: usize, out: *std.ArrayList(u32), stats: *lone.Stats) !void {
    const gpa = e.gpa;
    const r = &(e.mtp_rows orelse return error.NoMtpHead);
    var context: std.ArrayList(u32) = .empty;
    defer context.deinit(gpa);
    try context.appendSlice(gpa, prompt);
    try out.append(gpa, pending);
    try context.append(gpa, pending);
    var copies: lone.CopyIndex = .{ .gpa = gpa };
    defer copies.deinit();
    const copy_rows = 16; // decode.COPY_ROWS
    var f = e.forward();
    var tokens: [st.max_rows]u32 = undefined;
    var sampled: [st.max_rows]u32 = undefined;
    var path: [st.max_rows]u32 = undefined;
    var states = x.carry;
    var follow: []const u32 = &.{pending};
    while (out.items.len < count and !e.w.isEos(out.items[out.items.len - 1])) {
        tokens[0] = out.items[out.items.len - 1];
        const copied = try copies.propose(context.items, copy_rows - 1);
        try chain(&f, x, r, states, follow, if (copied.len > 0) 0 else depth);
        const drafted = if (copied.len > 0) copied else x.drafts[0..x.held];
        @memcpy(tokens[1..][0..drafted.len], drafted);
        const rows = 1 + drafted.len;
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
        const before = out.items.len;
        try out.appendSlice(gpa, tokens[1..n]);
        try out.append(gpa, terminal);
        try context.appendSlice(gpa, out.items[before..]);
        states = f.s.last_h;
        follow = out.items[before..];
        stats.rounds += 1;
        stats.drafted += rows - 1;
        stats.accepted += n - 1;
        // a draft is tried when every draft before it landed
        if (copied.len == 0) for (0..@min(n, rows - 1)) |dd| {
            stats.reached[dd] += 1;
            if (dd + 1 < n) stats.landed[dd] += 1;
        };
    }
}
