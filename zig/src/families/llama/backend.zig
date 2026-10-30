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
        cached: std.ArrayList(u32) = .empty,

        pub fn backend(x: *Self) be.Backend {
            return .{ .ptr = x, .vtable = &vtable, .penalties = x.m.k.draws.penalize_fn != null };
        }

        pub fn facts(x: *const Self) lanes.Model {
            // DFlash2 drafts its whole block in one forward: the timed verify windows alone price a depth
            if (x.drafter) |d| return .{ .exact_width = 8, .copy_drafts = false, .gpu_tokens = false, .mtp = true, .speculate = true, .speculate_early = false, .drafts = 7, .batch_rows = 8, .max_streams = 1, .window_costs = &d.windows };
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
            var reused: usize = 0;
            const limit = @min(x.cached.items.len, p.len - 1);
            while (reused < limit and x.cached.items[reused] == p[reused]) : (reused += 1) {}
            if (x.drafter) |d| {
                reused = try d.reuse(reused, p.len);
                d.r.reset();
                d.count = 0;
                const start = p.len -| (d.r.graph.config.window - 1);
                d.r.context = .{ .cache = 1, .begin = start, .end = start };
                try d.restore(start, reused);
            }
            x.owner = s;
            x.len = reused;
            x.cached.shrinkRetainingCapacity(reused);
            try x.cached.ensureTotalCapacity(x.m.gpa, p.len);
            x.filling = true;
            s.cached = @intCast(reused);
            s.fill_at.store(@intCast(reused), .monotonic);
        }

        fn prefillStep(p: *anyopaque, s: *Stream) anyerror!bool {
            const x = self(p);
            if (!x.filling) try x.opening(s);
            try x.owned(s);
            errdefer {
                x.owner = null;
                x.filling = false;
                x.cached.clearRetainingCapacity();
            }
            const prompt = s.prompt();
            const end = @import("prefill_plan.zig").end(x.len, prompt.len, x.m.prompt_rows);
            x.m.tap_buffer = 0;
            if (x.drafter) |d| d.capture(0);
            try x.m.prefill(prompt[x.len..end], x.len, end == prompt.len);
            if (x.drafter) |d| {
                for (x.len..end) |position| if (position >= d.r.context.end) {
                    try d.ingest(position - x.len, position);
                };
            }
            x.cached.appendSliceAssumeCapacity(prompt[x.len..end]);
            x.len = end;
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
            try x.cached.append(x.m.gpa, token);
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
                try x.cached.append(x.m.gpa, w.pending);
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
            try x.cached.appendSlice(x.m.gpa, tokens[0..w.rows()]);
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
            x.cached.shrinkRetainingCapacity(x.len);
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
                x.cached.shrinkRetainingCapacity(@min(x.cached.items.len, x.len));
                x.len = 0;
            }
        }
    };
}
