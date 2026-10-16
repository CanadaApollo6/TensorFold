//! LFM2-MoE (lfm2_moe: LiquidAI's LFM2.5-24B-A2B) behind the shared serial host: one stream, greedy replies in
//! prompt-lookup drafted rounds, sampled ones a token at a time. A prompt that extends what the stream already holds
//! (the last turn's prompt and reply) resumes there instead of starting over.
const std = @import("std");
const cuda = @import("cuda");
const api = @import("engine_api");
const lfm2 = @import("lfm2");
const Serial = api.serial_host;
const Allocator = std.mem.Allocator;
const max_rows = lfm2.state.max_rows;
/// Prompt positions between conv-state marks: a prompt that leaves the stream's tokens reads at most this many again.
const mark_every = 512;
comptime {
    if (mark_every % lfm2.engine.prompt_rows != 0) @compileError("marks fall on the prompt grid");
}

/// The context a callback needs current: the serial host calls the driver on its own thread.
threadlocal var bound: ?*const cuda.Context = null;

pub const Gpu = struct { driver: cuda.Driver, ctx: cuda.Context };

pub const Host = struct {
    gpa: Allocator,
    gpu: *Gpu,
    e: *lfm2.Engine,
    seq: lfm2.state.Seq,
    serial: *Serial.Host = undefined,
    /// The tokens the stream's state holds, in order.
    fed: std.ArrayList(u32) = .empty,
    /// Where each pair of tokens in `fed` ends (the later one's position), for the drafts' lookup; `indexed` of
    /// `fed`'s tokens are in it.
    index: std.AutoHashMapUnmanaged(u64, std.ArrayList(u32)) = .empty,
    indexed: usize = 0,
    /// The conv states at every `mark_every`th position a prompt passed: mark j holds position j * mark_every and
    /// `marked[j]` says it is `fed`'s (position 0 is the empty state, always there).
    marks: lfm2.state.Marks,
    marked: []bool,
    /// The request's prompt tokens taken so far: matched against `fed`, fed, or queued.
    cursor: usize = 0,
    /// Prompt tokens after `fed` not read yet: prompts are read in pieces on a fixed grid of positions (multiples of
    /// prompt_rows), so where a request's chunks are cut, or where it resumed, never changes the arithmetic.
    queue: std.ArrayList(u32) = .empty,
    /// Drafted rounds: the last pick, emitted but not yet fed.
    pending: ?u32 = null,
    /// Drafts a round at most (0: plain steps) and what the last draft earned.
    k: usize,
    earned: usize = 2,
    /// The request's repetition penalty and its window (1: off; the picks are the GPU's).
    penalty: f32 = 1,
    penalty_n: u32 = 64,
    startup: []u8 = &.{},
    io: std.Io,
    /// TF_LFM2_TRACE=1: ns inside drafted rounds and between them (the server's share), printed at each request's start.
    trace: bool = false,
    inside: i96 = 0,
    outside: i96 = 0,
    rounds: u64 = 0,
    gpu_ns: i96 = 0,
    left_at: i96 = 0,

    fn bind(p: *anyopaque) *Host {
        const h: *Host = @ptrCast(@alignCast(p));
        if (bound != &h.gpu.ctx) {
            h.gpu.ctx.makeCurrent() catch |e| std.log.err("cuCtxSetCurrent on the serial thread: {s}", .{@errorName(e)});
            bound = &h.gpu.ctx;
        }
        return h;
    }

    /// A new request: its prompt is matched against what the stream holds as its chunks arrive.
    fn reset(p: *anyopaque) !void {
        const h = bind(p);
        h.cursor = 0;
        h.pending = null;
        h.queue.clearRetainingCapacity();
    }

    /// The prompt's tokens [cursor, cursor + ids.len): those the stream already holds are skipped. A prompt that
    /// leaves them (or ends inside them: its last row's logits are needed) resumes at the latest mark before that.
    fn chunk(p: *anyopaque, ids: []const u32, last: bool) !void {
        const h = bind(p);
        errdefer h.lost();
        var rest = ids;
        if (h.cursor < h.fed.items.len) {
            const have = h.fed.items[h.cursor..];
            const n = @min(have.len, ids.len);
            const same = std.mem.indexOfDiff(u32, have[0..n], ids[0..n]) orelse n;
            if (same == n and !(last and n == ids.len)) {
                h.cursor += n;
                rest = ids[n..];
                if (rest.len == 0) return;
            } else {
                const limit = h.cursor + same - @intFromBool(last and same == ids.len);
                try h.rewind(limit);
                rest = ids[limit - h.cursor ..];
                h.cursor = limit;
            }
        }
        try h.queue.appendSlice(h.gpa, rest);
        h.cursor += rest.len;
        // whole grid pieces now; the last chunk reads the rest
        const grid = h.e.prompt_rows;
        const upto = if (last) h.queue.items.len else (h.fed.items.len + h.queue.items.len) / grid * grid -| h.fed.items.len;
        if (upto > 0) {
            try h.feed(h.queue.items[0..upto]);
            h.queue.replaceRangeAssumeCapacity(0, upto, &.{});
        }
        if (last) {
            var f = h.e.forward();
            try f.headLast(&h.seq);
            try h.e.s.synchronize(); // the prompt's time is its own, not the first round's
        }
    }

    /// The stream back to holding `fed[0..q]` for the latest mark q at or before `limit`, `fed[q..limit]` queued.
    fn rewind(h: *Host, limit: usize) !void {
        var j = @min(limit / mark_every, h.marked.len - 1);
        while (j > 0 and !h.marked[j]) j -= 1;
        const q = j * mark_every;
        if (j == 0) try h.seq.reset(h.e.ops(), h.e.w.g) else try h.marks.restore(h.e.ops(), &h.seq, j, q);
        for (h.marked[j + 1 ..]) |*m| m.* = false;
        h.forget();
        try h.queue.appendSlice(h.gpa, h.fed.items[q..limit]);
        h.fed.shrinkRetainingCapacity(q);
    }

    fn forget(h: *Host) void {
        var it = h.index.valueIterator();
        while (it.next()) |list| list.deinit(h.gpa);
        h.index.clearRetainingCapacity();
        h.indexed = 0;
    }

    fn pair(a: u32, b: u32) u64 {
        return @as(u64, a) << 32 | b;
    }

    /// Up to `k` tokens that followed the latest earlier match of the longest suffix (2 to 64 tokens) of `fed` and
    /// `next`, as lfm2.Engine.lookup finds them, from the 256 latest places their last pair ends.
    fn lookup(h: *Host, next: u32, k: usize) ![]const u32 {
        const fed = h.fed.items;
        for (@max(h.indexed, 1)..fed.len) |p| {
            const got = try h.index.getOrPut(h.gpa, pair(fed[p - 1], fed[p]));
            if (!got.found_existing) got.value_ptr.* = .empty;
            try got.value_ptr.append(h.gpa, @intCast(p));
        }
        h.indexed = fed.len;
        if (k == 0 or fed.len == 0) return &.{};
        const places = (h.index.get(pair(fed[fed.len - 1], next)) orelse return &.{}).items;
        const n = fed.len + 1; // the context: fed, then next
        var best: usize = 0;
        var at: usize = 0;
        var i = places.len;
        while (i > 0 and places.len - i < 256) {
            i -= 1;
            const s = places[i];
            var m: usize = 2;
            while (m < 64 and m <= s and fed[s - m] == fed[n - 1 - m]) m += 1;
            if (m > best) {
                best = m;
                at = s + 1;
                if (m == 64) break;
            }
        }
        if (best == 0) return &.{};
        try h.fed.append(h.gpa, next); // a draft may run up to `next` itself
        defer h.fed.items.len -= 1;
        return h.fed.items[at..@min(n, at + k)];
    }

    /// Prompt rows in pieces cut on the grid (multiples of prompt_rows from position 0), each mark taken as it is passed.
    fn feed(h: *Host, ids: []const u32) !void {
        var f = h.e.forward();
        var at: usize = 0;
        while (at < ids.len) {
            const pos = h.fed.items.len;
            const end = @min(ids.len, at + (h.e.prompt_rows - pos % h.e.prompt_rows)); // marks fall on the grid
            try f.step(ids[at..end], &h.seq, false);
            try h.fed.appendSlice(h.gpa, ids[at..end]);
            at = end;
            const j = h.fed.items.len / mark_every;
            if (h.fed.items.len % mark_every == 0 and j < h.marked.len) {
                try h.marks.save(h.e.ops(), &h.seq, j);
                h.marked[j] = true;
            }
        }
    }

    /// After a failed call the stream's state is unknown: start the next prompt from empty.
    fn lost(h: *Host) void {
        h.forget();
        h.queue.clearRetainingCapacity();
        @memset(h.marked, false);
        h.fed.clearRetainingCapacity();
        h.cursor = 0;
        h.pending = null;
        h.seq.reset(h.e.ops(), h.e.w.g) catch {};
    }

    fn fedCount(p: *anyopaque) u64 {
        const h: *Host = @ptrCast(@alignCast(p));
        return h.fed.items.len;
    }

    /// The next token from the last row's logits: greedy (the first maximum) or the request's sampling.
    fn draw(p: *anyopaque, sampling: ?api.Sampling) !u32 {
        const h = bind(p);
        var f = h.e.forward();
        if (sampling == null and h.penalty == 1) {
            var pick: [1]u32 = undefined;
            try f.picks(1, &pick);
            return pick[0];
        }
        const v = h.e.w.g.vocab;
        const row = try h.gpa.alloc(u16, v);
        defer h.gpa.free(row);
        try h.e.ops().b().download(std.mem.sliceAsBytes(row), h.e.scratch.logits);
        try h.e.s.synchronize();
        if (h.penalty != 1) try penalize(h.gpa, row, h.fed.items[h.fed.items.len - @min(h.fed.items.len, h.penalty_n)..], h.penalty);
        const s = sampling orelse return topPick(row);
        if (s.top_k == 0 or s.top_k >= v or s.temperature <= 0) return Serial.choose(h.gpa, .{ .bf16 = row }, h.fed.items.len, s);
        return drawTopK(h.gpa, row, h.fed.items.len, s);
    }

    fn setPenalty(p: *anyopaque, factor: f32, last_n: u32) !void {
        const h = bind(p);
        if (!(factor > 0) or !std.math.isFinite(factor)) return error.BadPenalty;
        h.penalty = factor;
        h.penalty_n = last_n;
    }

    fn advance(p: *anyopaque, token: u32) !void {
        const h = bind(p);
        errdefer h.lost();
        try h.e.advance(&h.seq, token);
        try h.fed.append(h.gpa, token);
    }

    fn begin(p: *anyopaque, _: []const u32, _: u32, _: []const u32) !void {
        const h = bind(p);
        h.pending = null;
        h.earned = 2;
        if (h.trace and h.rounds > 0) std.log.info("lfm2 rounds {d}: {d:.3} ms inside ({d:.3} in verify), {d:.3} ms between", .{ h.rounds, @as(f64, @floatFromInt(h.inside)) / 1e6 / @as(f64, @floatFromInt(h.rounds)), @as(f64, @floatFromInt(h.gpu_ns)) / 1e6 / @as(f64, @floatFromInt(h.rounds)), @as(f64, @floatFromInt(h.outside)) / 1e6 / @as(f64, @floatFromInt(h.rounds)) });
        h.inside = 0;
        h.outside = 0;
        h.rounds = 0;
        h.gpu_ns = 0;
        h.left_at = 0;
    }

    /// A drafted round: the prompt's first pick alone, then each round one window of the last pick and the tokens
    /// that followed the latest earlier match of the context's longest suffix, as far as the picks confirm them.
    fn batch(p: *anyopaque, budget: u32, eos: []const u32) !Serial.Batch {
        const h = bind(p);
        errdefer h.lost();
        const t0 = if (h.trace) std.Io.Clock.awake.now(h.io).toNanoseconds() else 0;
        defer if (h.trace) {
            const t1 = std.Io.Clock.awake.now(h.io).toNanoseconds();
            if (h.left_at > 0) h.outside += t0 - h.left_at;
            h.inside += t1 - t0;
            h.rounds += 1;
            h.left_at = t1;
        };
        const S = struct {
            threadlocal var out: [max_rows]u32 = undefined;
        };
        const pending = h.pending orelse {
            var pick: [1]u32 = undefined;
            var f = h.e.forward();
            try f.picks(1, &pick);
            h.pending = pick[0];
            S.out[0] = pick[0];
            return .{ .tokens = S.out[0..1], .stats = .{ .rounds = 1, .min_rows = 1 } };
        };
        const left = h.seq.capacity - h.seq.pos;
        if (left == 0 or budget == 0) return error.ContextFull;
        const room = @min(h.k, h.earned, budget - 1, left - 1);
        var d = try h.lookup(pending, room);
        for (d, 0..) |t, j| if (std.mem.indexOfScalar(u32, eos, t) != null) {
            d = d[0 .. j + 1]; // nothing past an end token: the next turn's prompt leaves it there
            break;
        };
        var rows: [max_rows]u32 = undefined;
        rows[0] = pending;
        @memcpy(rows[1 .. 1 + d.len], d);
        const W = 1 + d.len;
        const v0 = if (h.trace) std.Io.Clock.awake.now(h.io).toNanoseconds() else 0;
        const a = try h.e.verify(&h.seq, rows[0..W], &S.out);
        if (h.trace) h.gpu_ns += std.Io.Clock.awake.now(h.io).toNanoseconds() - v0;
        try h.fed.appendSlice(h.gpa, rows[0..a]);
        if (d.len > 0) h.earned = if (a == W) @min(2 * h.earned, max_rows - 1) else @max(1, a - 1);
        h.pending = S.out[a - 1];
        var n = a;
        for (S.out[0..a], 0..) |t, j| if (std.mem.indexOfScalar(u32, eos, t) != null) {
            n = j + 1;
            break;
        };
        return .{ .tokens = S.out[0..n], .stats = .{ .rounds = 1, .drafted = d.len, .accepted = a - 1, .min_rows = @intCast(W) } };
    }

    fn close(p: *anyopaque) void {
        const h: *Host = @ptrCast(@alignCast(p));
        h.serial.deinit();
        h.gpu.ctx.makeCurrent() catch {};
        h.forget();
        h.index.deinit(h.gpa);
        h.fed.deinit(h.gpa);
        h.queue.deinit(h.gpa);
        h.marks.deinit();
        h.gpa.free(h.marked);
        h.seq.deinit();
        h.e.deinit();
        h.gpu.ctx.deinit();
        h.gpu.driver.close();
        h.gpa.destroy(h.gpu);
        h.gpa.free(h.startup);
        h.gpa.destroy(h);
    }
};

/// llama.cpp's repeat penalty on a bf16 logits row: each distinct token of `recent` once, its logit divided by
/// `factor` when above 0 and multiplied otherwise (in f32, rounded back to bf16).
fn penalize(gpa: Allocator, row: []u16, recent: []const u32, factor: f32) !void {
    const ids = try gpa.dupe(u32, recent);
    defer gpa.free(ids);
    std.mem.sortUnstable(u32, ids, {}, std.sort.asc(u32));
    for (ids, 0..) |id, i| {
        if (i > 0 and ids[i - 1] == id) continue;
        if (id >= row.len) continue;
        const x: f32 = @bitCast(@as(u32, row[id]) << 16);
        row[id] = bf16(if (x <= 0) x * factor else x / factor);
    }
}

fn bf16(x: f32) u16 {
    const u: u32 = @bitCast(x);
    return @intCast((u + 0x7fff + ((u >> 16) & 1)) >> 16);
}

/// The row's greedy pick: the largest logit, ties to the lower id.
fn topPick(row: []const u16) !u32 {
    var top: f32 = -std.math.inf(f32);
    var at: u32 = 0;
    for (row, 0..) |bits, i| {
        const x: f32 = @bitCast(@as(u32, bits) << 16);
        if (!std.math.isFinite(x)) return error.NonfiniteLogits;
        if (x > top) {
            top = x;
            at = @intCast(i);
        }
    }
    return at;
}

test "repeat penalty on a bf16 row: each recent token once, by sign" {
    var row = [_]u16{ bf16(2.0), bf16(-1.0), bf16(0.5), bf16(1.9) };
    try penalize(std.testing.allocator, &row, &.{ 0, 1, 0, 0 }, 1.25);
    try std.testing.expectEqual(bf16(1.6), row[0]);
    try std.testing.expectEqual(bf16(-1.25), row[1]);
    try std.testing.expectEqual(bf16(0.5), row[2]);
    try std.testing.expectEqual(@as(u32, 3), try topPick(&row));
}

/// The sampler's draw from the row's `top_k` best logits alone (by value, ties to the lower id): it reads no
/// others, so the draw is the one the whole row gives, without sorting 128K candidates a token.
fn drawTopK(gpa: Allocator, row: []const u16, position: u64, s: api.Sampling) !u32 {
    const C = struct {
        v: f32,
        id: u32,
        fn worse(_: void, a: @This(), b: @This()) std.math.Order {
            if (a.v != b.v) return std.math.order(a.v, b.v);
            return std.math.order(b.id, a.id);
        }
    };
    var heap: std.PriorityQueue(C, void, C.worse) = .empty;
    defer heap.deinit(gpa);
    try heap.ensureTotalCapacity(gpa, s.top_k + 1);
    for (row, 0..) |bits, i| {
        const x: f32 = @bitCast(@as(u32, bits) << 16);
        if (!std.math.isFinite(x)) return error.NonfiniteLogits;
        const c: C = .{ .v = x, .id = @intCast(i) };
        if (heap.count() == s.top_k) {
            if (C.worse({}, c, heap.peek().?) != .gt) continue;
            _ = heap.pop();
        }
        try heap.push(gpa, c);
    }
    const n = heap.count();
    const values = try gpa.alloc(f64, n);
    defer gpa.free(values);
    const ids = try gpa.alloc(u64, n);
    defer gpa.free(ids);
    for (heap.items[0..n], values, ids) |c, *value, *id| {
        value.* = c.v;
        id.* = c.id;
    }
    return @intCast(try @import("lanes").sampling.choose(gpa, values, ids, position, s));
}

/// The engine over `gpu` (its context current here; the host owns it from now on) for `window` prompt plus reply
/// tokens; `drafts` false keeps every round one row.
pub fn open(gpa: Allocator, io: std.Io, gpu: *Gpu, dir: []const u8, window: u32, drafts: bool, startup: []const u8) !api.Opened {
    bound = &gpu.ctx;
    const e = try lfm2.Engine.init(gpa, io, &gpu.ctx, dir, .{ .context = window + max_rows });
    errdefer e.deinit();
    var seq = try e.sequence(window + max_rows);
    errdefer seq.deinit();
    var marks = try lfm2.state.Marks.init(e.ops(), e.w.g, window / mark_every + 1);
    errdefer marks.deinit();
    const marked = try gpa.alloc(bool, window / mark_every + 1);
    errdefer gpa.free(marked);
    @memset(marked, false);
    const h = try gpa.create(Host);
    errdefer gpa.destroy(h);
    const k: usize = if (!drafts) 0 else if (std.c.getenv("TF_LFM2_DRAFT")) |v| std.fmt.parseInt(usize, std.mem.span(v), 10) catch 15 else 15;
    h.* = .{ .gpa = gpa, .gpu = gpu, .e = e, .seq = seq, .marks = marks, .marked = marked, .k = @min(k, max_rows - 1), .io = io, .trace = std.c.getenv("TF_LFM2_TRACE") != null };
    h.startup = try std.fmt.allocPrint(gpa, "{s}; model {d:.2} GiB; one stream at a {d}-token window; prompt-lookup drafts {s}", .{ startup, @as(f64, @floatFromInt(e.w.bytes())) / (1 << 30), window, if (h.k > 0) "on" else "off" });
    errdefer gpa.free(h.startup);
    h.serial = try Serial.Host.init(gpa, io, .{
        .ctx = h,
        .info = .{ .name = "lfm2", .lanes = 1, .context_window = window, .prefill_step = @intCast(e.prompt_rows), .startup = h.startup },
        .reset = Host.reset,
        .prompt_chunk = Host.chunk,
        .draw = Host.draw,
        .advance = Host.advance,
        .penalize = Host.setPenalty,
        .draft_batch = Host.batch,
        .draft_begin = Host.begin,
        .fed = Host.fedCount,
    });
    return .{ .engine = h.serial.engine(), .close = Host.close, .ctx = h };
}
