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
    graph: ?cuda.graph.Exec = null, // the one-row decode step, captured for `graph_key`'s stream
    graph_key: u64 = 0,

    pub fn init(gpa: std.mem.Allocator, io: std.Io, ctx: *const cuda.Context, dir: []const u8, o: Options) !*Engine {
        const cfg = try c.Config.read(gpa, io, dir);
        const e = try gpa.create(Engine);
        errdefer gpa.destroy(e);
        e.gpa = gpa;
        e.ctx = ctx;
        e.graph = null;
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
        if (e.graph) |*g| g.deinit();
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
        if (!graphs()) {
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
        const exec = try e.stepGraph(&f, seq);
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

    fn graphs() bool {
        const v = std.c.getenv("TF_LFM2_GRAPHS") orelse return true;
        return !std.mem.eql(u8, std.mem.span(v), "0");
    }

    /// The decode step's graph for `seq` (its state pointers are baked in): captured on first use.
    fn stepGraph(e: *Engine, f: *Forward, seq: *st.Seq) !cuda.graph.Exec {
        var h = std.hash.Wyhash.init(seq.capacity);
        h.update(std.mem.sliceAsBytes(seq.conv));
        h.update(std.mem.sliceAsBytes(seq.keys));
        h.update(std.mem.asBytes(&seq.head_in));
        const key = h.final();
        if (e.graph) |g| if (e.graph_key == key) return g;
        if (e.graph) |*g| g.deinit();
        e.graph = null;
        try cuda.graph.beginCapture(e.s, .thread_local);
        f.decodeStep(seq) catch |err| {
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
        e.graph = x;
        e.graph_key = key;
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
