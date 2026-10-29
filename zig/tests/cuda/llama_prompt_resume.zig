const std = @import("std");
const llama = @import("llama");
const lanes = @import("lanes");

fn fill(a: std.mem.Allocator, b: *llama.backend.Llama, tokens: []const u32, stop: ?usize) !usize {
    var stream = try lanes.Stream.init(a, .{ .id = "resume-gate", .prompt = tokens, .max_new = 1, .eos = &.{} });
    defer stream.deinit(a);
    const backend = b.backend();
    defer backend.release(&stream);
    while (true) {
        const done = try backend.vtable.prefill_step.?(backend.ptr, &stream);
        if (done or (stop != null and stream.fill_at.load(.monotonic) >= stop.?)) break;
    }
    return @intCast(stream.cached);
}

pub fn run(a: std.mem.Allocator, m: *llama.forward.Model, d: ?*llama.draft.Attachment, seq: []const u32, kind: []const u8) !void {
    var b: llama.backend.Llama = .{ .m = m, .drafter = d };
    defer b.cached.deinit(a);
    const overlap = @min(1000, seq.len / 4);
    const extended = try a.alloc(u32, seq.len + overlap);
    defer a.free(extended);
    @memcpy(extended[0..seq.len], seq);
    @memset(extended[seq.len..], seq[seq.len - 1]);
    if (std.mem.eql(u8, kind, "resume")) {
        _ = try fill(a, &b, seq[0 .. seq.len - overlap], null);
    } else if (std.mem.eql(u8, kind, "shorten")) {
        _ = try fill(a, &b, extended[0 .. seq.len + 1], null);
    } else if (std.mem.eql(u8, kind, "resume-shorten")) {
        // The final drafter window starts at a tap kept from the first turn through the longer second one.
        _ = try fill(a, &b, seq[0 .. seq.len - overlap], null);
        _ = try fill(a, &b, extended[0 .. seq.len + 1], null);
    } else if (std.mem.eql(u8, kind, "cancel-shorten")) {
        const window = if (d) |draft| draft.r.graph.config.window else 2048;
        _ = try fill(a, &b, extended, (extended.len -| (window - 1)) + m.prompt_rows);
    } else if (std.mem.eql(u8, kind, "cancel-extend")) {
        const window = if (d) |draft| draft.r.graph.config.window else 2048;
        const prefix = seq[0 .. seq.len - @min(50, overlap)];
        _ = try fill(a, &b, prefix, (seq.len -| (window - 1)) + m.prompt_rows);
    } else if (std.mem.eql(u8, kind, "rewrite-cancel")) {
        const common = seq.len - overlap;
        for (extended[common..]) |*token| token.* = (token.* + 1) % @as(u32, @intCast(m.w.config.vocab));
        _ = try fill(a, &b, extended, null);
        @memcpy(extended[0..seq.len], seq);
        @memset(extended[seq.len..], seq[seq.len - 1]);
        _ = try fill(a, &b, extended, common + m.prompt_rows);
    } else return error.BadResumeCase;
    const reused = try fill(a, &b, seq, null);
    if (reused == 0) return error.PrefixNotReused;
    if (std.mem.eql(u8, kind, "resume") or std.mem.eql(u8, kind, "resume-shorten")) if (d) |draft| {
        const start = seq.len -| (draft.r.graph.config.window - 1);
        if (reused <= start) return error.NoRestoredDraftRows;
    };
}
