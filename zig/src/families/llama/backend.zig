const std = @import("std");
const lanes = @import("lanes");
const Model = @import("forward.zig").Model;
const be = lanes.backend;
const Stream = lanes.Stream;

pub const Llama = struct {
    m: *Model,
    owner: ?*Stream = null,
    len: usize = 0,
    base: usize = 0,
    filling: bool = false,
    handles: [1024]u32 = undefined,
    next: u64 = 0,

    pub fn backend(x: *Llama) be.Backend {
        return .{ .ptr = x, .vtable = &vtable, .penalties = x.m.k.draws.penalize_fn != null };
    }

    pub fn facts(_: *const Llama) lanes.Model {
        return .{ .exact_width = 1, .copy_drafts = false, .gpu_tokens = false, .mtp = false, .speculate = false, .batch_rows = 1, .max_streams = 1 };
    }

    const vtable: be.Backend.VTable = .{ .prefill = prefill, .prefill_step = prefillStep, .first = first, .queue = queue, .read = read, .verify = verify, .keep = keep, .draft = draft, .release = release, .reports_fill = true };

    fn self(p: *anyopaque) *Llama {
        return @ptrCast(@alignCast(p));
    }

    fn owned(x: *Llama, s: *Stream) !void {
        if (x.owner != s) return error.UnknownStream;
    }

    fn opening(x: *Llama, s: *Stream) !void {
        if (x.owner != null and x.owner != s) return error.NoFreeSlot;
        const p = s.prompt();
        if (p.len == 0) return error.EmptyPrompt;
        if (p[0] != x.m.w.config.bos) return error.MissingBos;
        if (p.len > x.m.context or s.max_new > x.m.context - p.len) return error.PastContext;
        x.owner = s;
        x.len = 0;
        x.filling = true;
        s.cached = 0;
        s.fill_at.store(0, .monotonic);
    }

    fn prefillStep(p: *anyopaque, s: *Stream) anyerror!bool {
        const x = self(p);
        if (!x.filling) try x.opening(s);
        try x.owned(s);
        errdefer {
            x.owner = null;
            x.filling = false;
        }
        const prompt = s.prompt();
        const end = @min(x.len + 32, prompt.len);
        while (x.len < end) : (x.len += 1) try x.m.step(prompt[x.len], x.len, x.len + 1 == prompt.len);
        try x.m.s.synchronize();
        s.fill_at.store(@intCast(x.len), .monotonic);
        x.filling = x.len < prompt.len;
        return !x.filling;
    }

    fn prefill(p: *anyopaque, s: *Stream) anyerror!void {
        try self(p).opening(s);
        while (!try prefillStep(p, s)) {}
    }

    fn pick(x: *Llama, s: *Stream, position: u64, pending: ?u32) !u32 {
        return x.m.pick(.{ .position = @intCast(position), .sampling = s.sampling, .penalties = s.penalties, .history = s.context.items[s.prompt_len..x.len], .pending = pending });
    }

    fn draw(x: *Llama, s: *Stream, position: u64, pending: ?u32) !u64 {
        const h = x.next;
        x.handles[@intCast(h % x.handles.len)] = try x.pick(s, position, pending);
        x.next += 1;
        return h;
    }

    fn first(p: *anyopaque, s: *Stream, position: u64) anyerror!u64 {
        const x = self(p);
        try x.owned(s);
        if (x.filling or position != x.len) return error.PositionMismatch;
        return x.draw(s, position, null);
    }

    fn read(p: *anyopaque, handle: u64) anyerror!u32 {
        const x = self(p);
        if (handle >= x.next or x.next - handle > x.handles.len) return error.NoSuchToken;
        return x.handles[@intCast(handle % x.handles.len)];
    }

    fn queue(p: *anyopaque, s: *Stream, feed: be.Feed, position: u64) anyerror!u64 {
        const x = self(p);
        try x.owned(s);
        if (position != x.len + 1) return error.PositionMismatch;
        const token = switch (feed) {
            .value => |v| v,
            .handle => |h| try read(p, h),
        };
        try x.m.step(token, x.len, true);
        const h = try x.draw(s, position, token);
        x.len += 1;
        return h;
    }

    fn verify(p: *anyopaque, windows: []const be.Window, out: []be.Verified) anyerror!void {
        const x = self(p);
        if (windows.len != 1 or out.len != 1) return error.LlamaSerialOnly;
        const w = windows[0];
        try x.owned(w.stream);
        if (w.rows() != 1 or w.parents != null or w.masks.len != 0 or w.early) return error.LlamaSerialOnly;
        if (w.positions.len != 1 or w.positions[0] != x.len + 1) return error.PositionMismatch;
        x.base = x.len;
        try x.m.step(w.pending, x.len, true);
        out[0].sampled[0] = try x.pick(w.stream, w.positions[0], w.pending);
        x.len += 1;
    }

    fn keep(p: *anyopaque, windows: []const be.Window, paths: []const []const u32) anyerror!void {
        const x = self(p);
        if (windows.len != 1 or paths.len != 1) return error.LlamaSerialOnly;
        try x.owned(windows[0].stream);
        if (paths[0].len != 1 or paths[0][0] != 0) return error.LlamaSerialOnly;
        x.len = x.base + 1;
    }

    fn draft(_: *anyopaque, requests: []const be.DraftRequest) anyerror!void {
        for (requests) |r| if (r.depth != 0 or r.lanes != null) return error.NoDraftHead;
    }

    fn release(p: *anyopaque, s: *Stream) void {
        const x = self(p);
        if (x.owner == s) {
            x.owner = null;
            x.filling = false;
            x.len = 0;
        }
    }
};
