//! LFM2-MoE's CUDA engine for one stream: the weights, kernels and scratch, prompt chunks and greedy serial decode.

const std = @import("std");
const cuda = @import("cuda");
const c = @import("shape.zig");
const kern = @import("kernels.zig");
const st = @import("state.zig");
const wts = @import("weights.zig");
const Forward = @import("forward.zig").Forward;

/// A prompt chunk's rows: the window attention's partials grow with them (rows x key tiles of the capacity).
pub const prompt_rows = 128;

pub const Options = struct { context: usize, prompt_rows: usize = prompt_rows };

pub const Engine = struct {
    gpa: std.mem.Allocator,
    ctx: *const cuda.Context,
    s: cuda.Stream,
    k: kern.Kernels,
    w: wts.Weights,
    scratch: st.Scratch,
    context: usize,
    prompt_rows: usize,
    graphs: [st.max_rows + 1]?cuda.graph.Exec = @splat(null), // [0] the one-row decode step, [W] a W-row verify window
    graph_key: u64 = 0, // the stream they were captured for

    pub fn init(gpa: std.mem.Allocator, io: std.Io, ctx: *const cuda.Context, dir: []const u8, o: Options) !*Engine {
        const cfg = try c.Config.read(gpa, io, dir);
        const e = try gpa.create(Engine);
        errdefer gpa.destroy(e);
        e.gpa = gpa;
        e.ctx = ctx;
        e.graphs = @splat(null);
        e.graph_key = 0;
        e.context = o.context;
        if (o.prompt_rows <= st.max_rows or o.prompt_rows > 4096) return error.BadPromptRows;
        e.prompt_rows = o.prompt_rows;
        e.s = try cuda.Stream.init(ctx.d, true);
        errdefer e.s.deinit();
        e.k = try kern.Kernels.load(gpa, io, ctx);
        errdefer e.k.deinit();
        e.w = try wts.load(gpa, io, e.ops(), dir, cfg);
        errdefer e.w.deinit();
        e.scratch = try st.Scratch.init(gpa, ctx.d, cfg.g, o.prompt_rows, o.context);
        errdefer e.scratch.deinit();
        try e.ops().b().fill32(e.scratch.moe.sync, 0, e.scratch.moe.sync_words);
        try e.ops().b().fill32(e.scratch.moe.counters, 0, e.scratch.moe.counter_words);
        try e.ops().b().fill32(e.scratch.attn_sync, 0, e.scratch.rows * cfg.g.kv_heads);
        try e.s.synchronize();
        return e;
    }

    pub fn deinit(e: *Engine) void {
        for (&e.graphs) |*g| if (g.*) |*x| x.deinit();
        e.scratch.deinit();
        e.w.deinit();
        e.k.deinit();
        e.s.deinit();
        e.gpa.destroy(e);
    }

    pub fn ops(e: *Engine) kern.Ops {
        return .{ .k = &e.k, .s = e.s };
    }

    pub fn forward(e: *Engine) Forward {
        return .{ .gpa = e.gpa, .w = &e.w, .ops = e.ops(), .s = &e.scratch };
    }

    pub fn sequence(e: *Engine, capacity: usize) !st.Seq {
        if (capacity > e.context) return error.ContextTooLong;
        return st.Seq.init(e.gpa, e.ops(), e.w.g, capacity);
    }

    /// The prompt from an empty state in even chunks of at most `prompt_rows` (more than a window's rows each).
    pub fn prefill(e: *Engine, seq: *st.Seq, prompt: []const u32) !void {
        if (prompt.len == 0) return error.EmptyPrompt;
        if (prompt.len >= seq.capacity) return error.PromptTooLong;
        try seq.reset(e.ops(), e.w.g);
        var f = e.forward();
        const n = (prompt.len + e.prompt_rows - 1) / e.prompt_rows;
        var at: usize = 0;
        for (0..n) |j| {
            const end = prompt.len * (j + 1) / n;
            try f.step(prompt[at..end], seq, false);
            at = end;
        }
    }

    /// Greedy one-row steps from `pending` until `count` tokens or an end token. Graphed (unless TF_LFM2_GRAPHS=0):
    /// `chunk` steps replay back to back, then their picks come back; an end token mid-chunk leaves the stream's
    /// state past it (the stream is done).
    pub fn decode(e: *Engine, seq: *st.Seq, pending: u32, count: usize, out: *std.ArrayList(u32)) !void {
        var f = e.forward();
        try out.append(e.gpa, pending);
        if (!graphsOn()) {
            var pick: [1]u32 = undefined;
            while (out.items.len < count and !e.w.isEos(out.items[out.items.len - 1]) and seq.pos < seq.capacity) {
                try f.step(out.items[out.items.len - 1 ..], seq, true);
                try f.picks(1, &pick);
                try out.append(e.gpa, pick[0]);
            }
            return;
        }
        if (out.items.len >= count or e.w.isEos(pending) or seq.pos >= seq.capacity) return;
        try f.stage(&.{pending}, seq.pos);
        const exec = try e.stepGraph(&f, seq, 0);
        var got: [chunk]u32 = undefined;
        while (true) {
            const n = @min(chunk, count - out.items.len, seq.capacity - seq.pos);
            for (0..n) |_| try exec.launchOn(e.s);
            try e.ops().b().download(std.mem.sliceAsBytes(got[0..n]), e.scratch.log + seq.pos * 4);
            try e.s.synchronize();
            seq.pos += n;
            for (got[0..n]) |t| {
                try out.append(e.gpa, t);
                if (e.w.isEos(t)) return;
            }
            if (out.items.len >= count or seq.pos >= seq.capacity) return;
        }
    }

    const chunk = 16; // graphed steps between host reads

    fn graphsOn() bool {
        const v = std.c.getenv("TF_LFM2_GRAPHS") orelse return true;
        return !std.mem.eql(u8, std.mem.span(v), "0");
    }

    /// Greedy decode with prompt-lookup drafts. A step drafts the tokens that followed the latest earlier match of
    /// the longest suffix of `prompt` + `out` (2 tokens at least), as many as the last draft earned (doubling after a
    /// fully accepted one, else the accepted count; at most `k`), and checks them in one window, keeping the picks
    /// they confirm plus one. A wider window costs ~0.4-0.9 ms a row on the 3090 (experts the drafts add), so only
    /// matches that keep paying get long drafts. Picks are the greedy ones; the wider windows' matmuls round
    /// differently, so near-ties can go the other way from `decode`'s.
    pub fn decodeDraft(e: *Engine, seq: *st.Seq, prompt: []const u32, pending: u32, count: usize, k: usize, out: *std.ArrayList(u32), stats: *DraftStats) !void {
        if (k == 0 or k >= st.max_rows) return error.BadDraft;
        var ctx: std.ArrayList(u32) = .empty;
        defer ctx.deinit(e.gpa);
        try ctx.appendSlice(e.gpa, prompt);
        try ctx.append(e.gpa, pending);
        try out.append(e.gpa, pending);
        var rows: [st.max_rows]u32 = undefined;
        var got: [st.max_rows]u32 = undefined;
        var earned: usize = 2;
        while (out.items.len < count and !e.w.isEos(out.items[out.items.len - 1]) and seq.pos < seq.capacity) {
            const room = @min(k, earned, count - out.items.len - 1, seq.capacity - seq.pos - 1);
            const d = lookup(ctx.items, room, 2);
            const W = 1 + d.len;
            rows[0] = ctx.items[ctx.items.len - 1];
            @memcpy(rows[1..W], d);
            const a = try e.verify(seq, rows[0..W], &got);
            if (d.len > 0) earned = if (a == W) @min(2 * earned, st.max_rows - 1) else @max(1, a - 1);
            stats.steps += 1;
            stats.drafted += d.len;
            stats.accepted += a - 1;
            for (got[0..a]) |t| {
                try out.append(e.gpa, t);
                try ctx.append(e.gpa, t);
                if (e.w.isEos(t) or out.items.len >= count) return;
            }
        }
    }

    /// One window at the stream's position: `rows` (the last token, then its drafts) committed as far as their greedy
    /// picks confirm them. The picks into `got`; returns how many count (the confirmed drafts plus one), the stream
    /// moved past that many rows. One row is the plain decode step.
    pub fn verify(e: *Engine, seq: *st.Seq, rows: []const u32, got: *[st.max_rows]u32) !usize {
        const W = rows.len;
        if (W < 1 or W > st.max_rows or seq.pos + W > seq.capacity) return error.BadWindow;
        var f = e.forward();
        try f.stage(rows, seq.pos);
        const exec = try e.stepGraph(&f, seq, if (W == 1) 0 else W); // no draft: the plain step (its conv commits)
        try exec.launchOn(e.s);
        try e.ops().b().download(std.mem.sliceAsBytes(got[0..W]), e.scratch.picks);
        var a: [1]u32 = .{1};
        if (W > 1) try e.ops().b().download(std.mem.sliceAsBytes(&a), e.scratch.accepted);
        try e.s.synchronize();
        seq.pos += a[0];
        return a[0];
    }

    /// One row through the head (the graphed decode step), its logits left in the scratch; no host read.
    pub fn advance(e: *Engine, seq: *st.Seq, token: u32) !void {
        if (seq.pos + 1 > seq.capacity) return error.BadWindow;
        var f = e.forward();
        try f.stage(&.{token}, seq.pos);
        const exec = try e.stepGraph(&f, seq, 0);
        try exec.launchOn(e.s);
        seq.pos += 1;
    }

    /// Up to `k` tokens that followed the latest earlier occurrence of the longest suffix of `ctx` (at most 64 long);
    /// none when it is shorter than `least`.
    pub fn lookup(ctx: []const u32, k: usize, least: usize) []const u32 {
        const n = ctx.len;
        if (k == 0 or n < 2) return ctx[0..0];
        var at: usize = 0;
        var best: usize = 0;
        var s = n - 1;
        while (s > 0) {
            s -= 1;
            if (ctx[s] != ctx[n - 1]) continue;
            var m: usize = 1;
            while (m < 64 and m <= s and ctx[s - m] == ctx[n - 1 - m]) m += 1;
            if (m > best) {
                best = m;
                at = s + 1;
                if (m == 64) break;
            }
        }
        if (best < least) return ctx[0..0];
        return ctx[at..@min(n, at + k)];
    }

    /// Benchmark: ms a W-row verify window takes at the stream's position (`iters` replays of its graph, state
    /// untouched but the conv commits), into ms[W - 1] for W = 1 ..= ms.len.
    pub fn benchWindows(e: *Engine, seq: *st.Seq, io: std.Io, iters: usize, ms: []f64) !void {
        var f = e.forward();
        const clock = std.Io.Clock.awake;
        var rows: [st.max_rows]u32 = @splat(0);
        for (ms, 1..) |*m, W| {
            for (rows[0..W], 0..) |*r, j| r.* = @intCast(1000 + 37 * j);
            try f.stage(rows[0..W], seq.pos);
            const exec = try e.stepGraph(&f, seq, W);
            try exec.launchOn(e.s);
            try e.s.synchronize();
            const t0 = clock.now(io).toNanoseconds();
            for (0..iters) |_| try exec.launchOn(e.s);
            try e.s.synchronize();
            m.* = @as(f64, @floatFromInt(clock.now(io).toNanoseconds() - t0)) / 1e6 / @as(f64, @floatFromInt(iters));
        }
    }

    pub const DraftStats = struct { steps: usize = 0, drafted: usize = 0, accepted: usize = 0 };

    /// Graph `which` for `seq` (its state pointers are baked in), captured on first use: 0 the one-row decode step,
    /// W a W-row verify window.
    fn stepGraph(e: *Engine, f: *Forward, seq: *st.Seq, which: usize) !cuda.graph.Exec {
        var h = std.hash.Wyhash.init(seq.capacity);
        h.update(std.mem.sliceAsBytes(seq.conv));
        h.update(std.mem.sliceAsBytes(seq.keys));
        h.update(std.mem.asBytes(&seq.head_in));
        const key = h.final();
        if (e.graph_key != key) {
            for (&e.graphs) |*g| if (g.*) |*x| {
                x.deinit();
                g.* = null;
            };
            e.graph_key = key;
        }
        if (e.graphs[which]) |g| return g;
        try cuda.graph.beginCapture(e.s, .thread_local);
        (if (which == 0) f.decodeStep(seq) else f.verifyStep(which, seq)) catch |err| {
            if (cuda.graph.endCapture(e.s)) |g| {
                var gg = g;
                gg.deinit();
            } else |_| {}
            return err;
        };
        var g = try cuda.graph.endCapture(e.s);
        defer g.deinit();
        const x = try g.instantiate();
        try x.upload(e.s);
        e.graphs[which] = x;
        return x;
    }

    /// The prompt's first pick after prefill.
    pub fn first(e: *Engine, seq: *st.Seq) !u32 {
        var f = e.forward();
        try f.headLast(seq);
        var pick: [1]u32 = undefined;
        try f.picks(1, &pick);
        return pick[0];
    }
};
