//! A one-stream verify round's layers (embed through head) captured once a stream and shape, then replayed.

const std = @import("std");
const cuda = @import("cuda");
const c = @import("shape.zig");
const rnd = @import("round.zig");
const fw = @import("forward.zig");

/// Graphs are on unless TF_QWEN27_GRAPHS=0.
pub fn wanted() bool {
    const v = std.c.getenv("TF_QWEN27_GRAPHS") orelse return true;
    return !std.mem.eql(u8, std.mem.span(v), "0");
}

/// The captured rounds by key: the window's rows and GDN schedule, and every state pointer the layers bake in.
pub const Graphs = struct {
    gpa: std.mem.Allocator,
    map: std.AutoHashMapUnmanaged(u64, cuda.graph.Exec) = .empty,
    off: bool = false, // a capture failed: rounds run eagerly from then on

    pub fn deinit(x: *Graphs) void {
        var it = x.map.valueIterator();
        while (it.next()) |e| e.deinit();
        x.map.deinit(x.gpa);
    }

    /// One stream, a bf16 cache, no host reads or taps: every per-round input the layers read is staged on the device.
    pub fn usable(x: *const Graphs, f: *const fw.Forward, r: *const rnd.Round) bool {
        return !x.off and !r.multi and !c.kv4() and f.taps == null and f.dump == null and !fw.prof_on;
    }

    fn key(r: *const rnd.Round, parts: []const rnd.Part) u64 {
        var h = std.hash.Wyhash.init(0);
        for ([_]usize{ r.rows, r.slots, r.most }) |v| h.update(std.mem.asBytes(&v));
        const seq = parts[0].seq;
        h.update(std.mem.sliceAsBytes(seq.rec));
        h.update(std.mem.sliceAsBytes(seq.conv));
        return h.final();
    }

    /// The round's layers: replayed if this stream and shape were captured, else captured now and replayed.
    pub fn run(x: *Graphs, f: *fw.Forward, r: *const rnd.Round, parts: []const rnd.Part) !void {
        const s = f.ops.s;
        const k = key(r, parts);
        if (x.map.get(k)) |e| return e.launchOn(s);
        try cuda.graph.beginCapture(s, .thread_local);
        f.body(r, parts) catch |err| {
            if (cuda.graph.endCapture(s)) |g| {
                var dead = g;
                dead.deinit();
            } else |_| {}
            x.off = true;
            return err;
        };
        var g = try cuda.graph.endCapture(s);
        defer g.deinit();
        var e = try g.instantiate();
        errdefer e.deinit();
        try e.upload(s);
        try x.map.put(x.gpa, k, e);
        try e.launchOn(s);
    }
};
