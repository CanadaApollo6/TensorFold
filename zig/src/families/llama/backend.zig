const std = @import("std");
const lanes = @import("lanes");
const Model = @import("forward.zig").Model;
const be = lanes.backend;
const Stream = lanes.Stream;

pub const Llama = Adapter(Model, @import("draft.zig").Attachment);

pub fn Adapter(comptime Target: type, comptime Draft: type) type {
    return struct {
        const Self = @This();
        m: *Target,
        drafter: ?*Draft = null,
        owner: ?*Stream = null,
        len: usize = 0,
        base: usize = 0,
        filling: bool = false,
        handles: [1024]u32 = undefined,
        next: u64 = 0,

        pub fn backend(x: *Self) be.Backend {
            return .{ .ptr = x, .vtable = &vtable, .penalties = x.m.k.draws.penalize_fn != null };
        }

        pub fn facts(x: *const Self) lanes.Model {
            if (x.drafter != null) return .{ .exact_width = 8, .copy_drafts = false, .gpu_tokens = false, .mtp = true, .speculate = true, .speculate_early = false, .drafts = 7, .batch_rows = 8, .max_streams = 1 };
            return .{ .exact_width = 1, .copy_drafts = false, .gpu_tokens = false, .mtp = false, .speculate = false, .batch_rows = 1, .max_streams = 1 };
        }

        const vtable: be.Backend.VTable = .{ .prefill = prefill, .prefill_step = prefillStep, .first = first, .queue = queue, .read = read, .verify = verify, .keep = keep, .draft = draft, .release = release, .reports_fill = true, .tree = tree };

        fn self(p: *anyopaque) *Self {
            return @ptrCast(@alignCast(p));
        }

        fn owned(x: *Self, s: *Stream) !void {
            if (x.owner != s) return error.UnknownStream;
        }

        fn opening(x: *Self, s: *Stream) !void {
            if (x.owner != null and x.owner != s) return error.NoFreeSlot;
            const p = s.prompt();
            if (p.len == 0) return error.EmptyPrompt;
            if (p[0] != x.m.w.config.bos) return error.MissingBos;
            if (p.len > x.m.context or s.max_new > x.m.context - p.len) return error.PastContext;
            if (x.drafter) |d| {
                d.r.reset();
                d.count = 0;
                const start = p.len -| (d.r.graph.config.window - 1);
                d.r.context = .{ .cache = 1, .begin = start, .end = start };
            }
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
            while (x.len < end) : (x.len += 1) {
                const capture = if (x.drafter) |d| x.len >= d.r.context.end else false;
                if (x.drafter) |d| {
                    x.m.tap_buffer = 0;
                    if (capture) d.capture(0);
                }
                try x.m.step(prompt[x.len], x.len, x.len + 1 == prompt.len);
                if (capture) try x.drafter.?.ingest(0, x.len);
            }
            try x.m.s.synchronize();
            s.fill_at.store(@intCast(x.len), .monotonic);
            x.filling = x.len < prompt.len;
            return !x.filling;
        }

        fn prefill(p: *anyopaque, s: *Stream) anyerror!void {
            try self(p).opening(s);
            while (!try prefillStep(p, s)) {}
        }

        fn pick(x: *Self, s: *Stream, position: u64, pending: ?u32) !u32 {
            return x.m.pick(.{ .position = @intCast(position), .sampling = s.sampling, .penalties = s.penalties, .history = s.context.items[s.prompt_len..x.len], .pending = pending });
        }

        fn draw(x: *Self, s: *Stream, position: u64, pending: ?u32) !u64 {
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
            if (x.drafter) |d| d.capture(0);
            try x.m.step(token, x.len, true);
            if (x.drafter) |d| try d.ingest(0, x.len);
            const h = try x.draw(s, position, token);
            x.len += 1;
            return h;
        }

        fn verify(p: *anyopaque, windows: []const be.Window, out: []be.Verified) anyerror!void {
            const x = self(p);
            if (windows.len != 1 or out.len != 1) return error.LlamaSerialOnly;
            const w = windows[0];
            try x.owned(w.stream);
            if (x.drafter == null) {
                if (w.rows() != 1 or w.parents != null or w.masks.len != 0 or w.early) return error.LlamaSerialOnly;
                if (w.positions.len != 1 or w.positions[0] != x.len + 1) return error.PositionMismatch;
                x.base = x.len;
                try x.m.step(w.pending, x.len, true);
                out[0].sampled[0] = try x.pick(w.stream, w.positions[0], w.pending);
                x.len += 1;
                return;
            }
            const d = x.drafter.?;
            if (w.rows() > 8 or w.held != 0 or w.parents != null or w.masks.len != 0 or w.early) return error.LlamaSerialOnly;
            if (w.positions.len != w.rows()) return error.PositionMismatch;
            x.base = x.len;
            var history: std.ArrayList(u32) = .empty;
            defer history.deinit(x.m.gpa);
            try history.appendSlice(x.m.gpa, w.stream.context.items[w.stream.prompt_len..x.len]);
            var tokens: [8]u32 = undefined;
            tokens[0] = w.pending;
            @memcpy(tokens[1..w.rows()], w.tokens);
            d.capture(0);
            if (w.rows() == 1) try x.m.step(tokens[0], x.len, true) else try x.m.batch(tokens[0..w.rows()], x.len);
            const logits = x.m.logits;
            defer x.m.logits = logits;
            for (0..w.rows()) |row| {
                if (w.positions[row] != x.len + 1) return error.PositionMismatch;
                const token = tokens[row];
                x.m.logits = logits + row * x.m.w.config.vocab * 4;
                out[0].sampled[row] = try x.m.pick(.{ .position = @intCast(w.positions[row]), .sampling = w.stream.sampling, .penalties = w.stream.penalties, .history = history.items, .pending = token });
                try history.append(x.m.gpa, token);
                x.len += 1;
            }
        }

        fn keep(p: *anyopaque, windows: []const be.Window, paths: []const []const u32) anyerror!void {
            const x = self(p);
            if (windows.len != 1 or paths.len != 1) return error.LlamaSerialOnly;
            try x.owned(windows[0].stream);
            if (paths[0].len == 0 or paths[0].len > windows[0].rows()) return error.LlamaSerialOnly;
            for (paths[0], 0..) |row, i| {
                if (row != i) return error.LlamaSerialOnly;
            }
            x.len = x.base + paths[0].len;
        }

        fn draft(p: *anyopaque, requests: []const be.DraftRequest) anyerror!void {
            const x = self(p);
            for (requests) |r| {
                try x.owned(r.stream);
                const d = x.drafter orelse {
                    if (r.depth != 0 or r.lanes != null) return error.NoDraftHead;
                    continue;
                };
                if (r.lanes != null) return error.LlamaSerialOnly;
                if (r.rows) |rows| {
                    if (r.start != d.r.context.end or r.start + rows.len != x.len) return error.DraftPositionMismatch;
                    for (rows, 0..) |row, i| {
                        if (row != i) return error.LlamaSerialOnly;
                        try d.ingest(row, @intCast(r.start + i));
                    }
                }
                if (d.r.context.end != x.len or r.position != x.len + 1) return error.DraftPositionMismatch;
                std.debug.assert(d.r.context.end == x.len);
                const pending = if (r.follow.len > 0) r.follow[r.follow.len - 1] else switch (r.first orelse return error.NoFirstToken) {
                    .value => |v| v,
                    .handle => |h| try read(p, h),
                };
                try d.propose(pending, @intCast(@min(r.depth, (x.m.context - x.len) -| 1)));
                if (std.c.getenv("TF_LLAMA_DRAFT_TRACE") != null and d.count > 0) std.log.info("Llama draft offered={d} cache={d}", .{ d.count, x.len });
            }
        }

        fn tree(p: *anyopaque, s: *Stream, a: std.mem.Allocator) anyerror!?lanes.stream.Held {
            const x = self(p);
            try x.owned(s);
            const d = x.drafter orelse return null;
            return .{ .count = @intCast(d.count), .tokens = try a.dupe(u32, d.held[0..d.count]) };
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
}
