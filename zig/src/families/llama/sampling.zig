const std = @import("std");
const cuda = @import("cuda");
const lanes = @import("lanes");
const Kernels = @import("kernels.zig").Kernels;

pub const Draw = struct {
    position: u32,
    sampling: ?lanes.Sampling = null,
    penalties: lanes.Penalties = .{},
    history: []const u32 = &.{},
    pending: ?u32 = null,
};

pub fn deviceBytes(vocab: usize, context: usize) usize {
    return entriesAt(vocab) + context * 12;
}

fn entriesAt(vocab: usize) usize {
    return std.mem.alignForward(usize, 48 + vocab * 2, 4);
}

pub const Sampler = struct {
    gpa: std.mem.Allocator,
    host: cuda.HostBuffer,
    dev: cuda.DeviceBuffer,
    counts: std.AutoArrayHashMapUnmanaged(u32, u32) = .empty,
    vocab: usize,

    pub fn init(a: std.mem.Allocator, d: *const cuda.Driver, vocab: usize, context: usize) !Sampler {
        var host = try cuda.HostBuffer.alloc(d, 48 + context * 12);
        errdefer host.free();
        return .{ .gpa = a, .host = host, .dev = try cuda.DeviceBuffer.alloc(d, deviceBytes(vocab, context)), .vocab = vocab };
    }

    pub fn deinit(x: *Sampler) void {
        x.counts.deinit(x.gpa);
        x.dev.free();
        x.host.free();
    }

    fn bump(x: *Sampler, token: u32) !void {
        const got = try x.counts.getOrPut(x.gpa, token);
        if (!got.found_existing) got.value_ptr.* = 0;
        got.value_ptr.* += 1;
    }

    pub fn draw(x: *Sampler, k: *const Kernels, s: cuda.Stream, logits: u64, out: u64, d: Draw) !void {
        errdefer s.synchronize() catch {};
        const sampled = if (d.sampling) |sp| sp.temperature > 0 else false;
        if (!sampled and (!d.penalties.active() or (d.history.len == 0 and d.pending == null)))
            return k.argmax(s, logits, out, x.vocab);
        const scores = x.dev.ptr + 48;
        try k.narrow(s, logits, scores, x.vocab);
        if (d.penalties.active()) {
            x.counts.clearRetainingCapacity();
            for (d.history) |t| try x.bump(t);
            if (d.pending) |t| try x.bump(t);
            const n = x.counts.count();
            if (n * 12 > x.host.bytes.len - 48) return error.PastContext;
            const words = x.host.slice(u32)[12..];
            for (x.counts.keys(), x.counts.values(), 0..) |t, count, i| {
                if (t >= x.vocab) return error.BadToken;
                words[i] = 0;
                words[n + i] = t;
                words[2 * n + i] = @bitCast(d.penalties.of(count));
            }
            const entries = entriesAt(x.vocab);
            try x.dev.uploadAsync(entries, x.host.bytes[48..][0 .. n * 12], s.handle);
            try k.draws.penalize(s, scores, x.vocab, x.dev.ptr + entries, n);
        }
        if (!sampled) {
            try k.widen(s, scores, logits, x.vocab);
            return k.argmax(s, logits, out, x.vocab);
        }
        const rule = cuda.sample.Rule.from(lanes.gpu_rule.rule(d.sampling.?));
        @memcpy(x.host.bytes[0..32], std.mem.asBytes(&rule));
        const meta = [4]i32{ @as(i32, @intCast(d.position)) - 1, 0, 0, 0 };
        @memcpy(x.host.bytes[32..48], std.mem.asBytes(&meta));
        try x.dev.uploadAsync(0, x.host.bytes[0..48], s.handle);
        try k.draws.draw(s, scores, x.vocab, x.dev.ptr, x.dev.ptr + 32, 0, out, 1, null, null);
    }
};
