const std = @import("std");
const lanes = @import("lanes");
const be = lanes.backend;

pub const Forced = struct {
    inner: be.Backend,
    script: []const u32,
    prompt_len: usize = 0,
    proposal_start: usize = 0,
    handles: [1024]u32 = undefined,
    pub fn backend(f: *Forced) be.Backend {
        return .{ .ptr = f, .vtable = &.{ .prefill = prefill, .first = first, .queue = queue, .read = read, .verify = verify, .keep = keep, .draft = draft, .tree = tree, .release = release } };
    }
    fn self(p: *anyopaque) *Forced {
        return @ptrCast(@alignCast(p));
    }
    fn at(f: *Forced, pos: u64) u32 {
        return f.script[@min(pos - f.prompt_len, f.script.len - 1)];
    }
    fn prefill(p: *anyopaque, s: *lanes.Stream) !void {
        const f = self(p);
        f.prompt_len = s.prompt().len;
        try f.inner.prefill(s);
    }
    fn first(p: *anyopaque, s: *lanes.Stream, pos: u64) !u64 {
        const f = self(p);
        const h = try f.inner.first(s, pos);
        f.handles[@intCast(h % 1024)] = f.at(pos);
        return h;
    }
    fn queue(p: *anyopaque, s: *lanes.Stream, feed: be.Feed, pos: u64) !u64 {
        const f = self(p);
        const value = switch (feed) {
            .value => |v| v,
            .handle => |h| f.handles[@intCast(h % 1024)],
        };
        const h = try f.inner.queue(s, .{ .value = value }, pos);
        f.handles[@intCast(h % 1024)] = f.at(pos);
        return h;
    }
    fn read(p: *anyopaque, h: u64) !u32 {
        return self(p).handles[@intCast(h % 1024)];
    }
    fn verify(p: *anyopaque, windows: []const be.Window, out: []be.Verified) !void {
        const f = self(p);
        try f.inner.verify(windows, out);
        // Fixed draws isolate stopping at exact offsets while the real GPU still executes every verification row.
        for (windows, out) |w, o| for (w.positions, 0..) |pos, i| {
            o.sampled[i] = f.at(pos);
        };
    }
    fn keep(p: *anyopaque, windows: []const be.Window, paths: []const []const u32) !void {
        try self(p).inner.keep(windows, paths);
    }
    fn draft(p: *anyopaque, requests: []const be.DraftRequest) !void {
        const f = self(p);
        if (requests.len > 0) f.proposal_start = @intCast(requests[0].position - f.prompt_len);
        try f.inner.draft(requests);
    }
    fn tree(p: *anyopaque, s: *lanes.Stream, a: std.mem.Allocator) !?lanes.stream.Held {
        const f = self(p);
        const held = (try f.inner.vtable.tree.?(f.inner.ptr, s, a)) orelse return null;
        for (held.tokens orelse return error.NoForcedProposals, 0..) |*token, i| token.* = f.script[@min(f.proposal_start + i, f.script.len - 1)];
        return held;
    }
    fn release(p: *anyopaque, s: *lanes.Stream) void {
        self(p).inner.vtable.release(self(p).inner.ptr, s);
    }
};

test "forced draws put EOS and decoded stop at every accepted row" {
    const a = std.testing.allocator;
    for (0..2) |kind| for (2..9) |cut| {
        var fake: lanes.fake.Fake = .{ .gpa = a };
        defer fake.deinit();
        var vtable = fake.backend().vtable.*;
        // like the Llama drafter: the held tree is as deep as the last draft request asked
        const Held = struct {
            var depth: u32 = 0;
            var inner: @TypeOf(vtable.draft) = undefined;
            fn draft(p: *anyopaque, requests: []const be.DraftRequest) anyerror!void {
                if (requests.len > 0) depth = requests[0].depth;
                try inner(p, requests);
            }
            fn tree(_: *anyopaque, _: *lanes.Stream, allocator: std.mem.Allocator) !?lanes.stream.Held {
                const tokens = try allocator.alloc(u32, depth);
                @memset(tokens, 2);
                return .{ .count = depth, .tokens = tokens };
            }
        };
        Held.inner = vtable.draft;
        vtable.draft = Held.draft;
        vtable.tree = Held.tree;
        var inner = fake.backend();
        inner.vtable = &vtable;
        var ids: [16]u32 = @splat(1);
        ids[cut - 1] = 11;
        var forced: Forced = .{ .inner = inner, .script = &ids };
        const facts: lanes.Model = .{ .exact_width = 8, .copy_drafts = false, .gpu_tokens = false, .mtp = true, .speculate = true, .speculate_early = false, .drafts = 7, .batch_rows = 8, .max_streams = 1 };
        var cfg = try lanes.Config.init(a, facts, 8, 7);
        defer cfg.deinit(a);
        var clock: lanes.fake.FixedClock = .{};
        var engine = lanes.Engine.init(a, &cfg, forced.backend(), clock.clock());
        defer engine.deinit();
        var marker: u32 = 11;
        var stream = try lanes.Stream.init(a, .{ .id = "cut", .prompt = &.{1}, .max_new = 16, .eos = if (kind == 0) &.{11} else &.{}, .drafts = true, .stop_check = if (kind == 1) .{ .ptr = &marker, .check = struct {
            fn stop(p: *anyopaque, tokens: []const u32) bool {
                const token: *u32 = @ptrCast(@alignCast(p));
                return tokens[tokens.len - 1] == token.*;
            }
        }.stop } else null });
        defer stream.deinit(a);
        try engine.addStream(&stream);
        while (engine.activeCount() > 0) try engine.step();
        try std.testing.expectEqual(cut, stream.emitted().len);
        try std.testing.expectEqual(.stop, stream.reason);
        try std.testing.expect(stream.accepted >= cut - 1);
    };
}
