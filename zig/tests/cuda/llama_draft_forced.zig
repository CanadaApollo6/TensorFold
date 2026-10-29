const std = @import("std");
const llama = @import("llama");
const lanes = @import("lanes");
const check = @import("check.zig");

pub fn run(gpu: check.Gpu, model: []const u8, draft: []const u8, prompt: []const u32, expected: []const u32) !void {
    if (expected.len < 24) return error.ForcedCoverageTooShort;
    const loaded = try llama.native.open(gpu.gpa, gpu.io, gpu.ctx, model, null, .{
        .context = prompt.len + expected.len + 8,
        .drafts = true,
        .drafter = draft,
    });
    defer loaded.deinit(loaded.ctx);
    var cfg = try lanes.Config.init(gpu.gpa, loaded.facts, 8, 7);
    defer cfg.deinit(gpu.gpa);
    var clock: lanes.fake.FixedClock = .{};
    var engine = lanes.Engine.init(gpu.gpa, &cfg, loaded.backend, clock.clock());
    defer engine.deinit();
    var stream = try lanes.Stream.init(gpu.gpa, .{ .id = "forced", .prompt = prompt, .max_new = @intCast(expected.len), .eos = &.{}, .drafts = true });
    defer stream.deinit(gpu.gpa);
    try engine.addStream(&stream);
    const first = stream.next.?.tokens orelse return error.NoForcedDrafts;
    if (first.len == 0) return error.NoForcedDrafts;
    @memcpy(first, expected[stream.emitted().len..][0..first.len]);
    try engine.step();
    if (stream.accepted == 0 or stream.accepted != stream.drafted) return error.FullAcceptanceNotExercised;
    const second = stream.next.?.tokens orelse return error.NoForcedDrafts;
    if (second.len == 0) return error.NoForcedDrafts;
    const c = try llama.Config.read(gpu.gpa, gpu.io, model);
    second[0] = (expected[stream.emitted().len] + 1) % @as(u32, @intCast(c.vocab));
    try engine.step();
    if (stream.drafted <= stream.accepted) return error.RejectionNotExercised;
    while (engine.activeCount() > 0) try engine.step();
    if (!std.mem.eql(u32, expected, stream.emitted())) return error.ForcedTokensDiffer;
    std.debug.print("FORCED full_accept_then_reject tokens={d} equal\n", .{expected.len});
}
