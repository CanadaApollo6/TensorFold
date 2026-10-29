const std = @import("std");
const llama = @import("llama");
const lanes = @import("lanes");
const check = @import("check.zig");
const Result = struct {
    tokens: []u32,
    seconds: f64,
    prefill_seconds: f64,
    rounds: u64,
    drafted: u64,
    accepted: u64,
};
fn generate(gpu: check.Gpu, model: []const u8, draft: ?[]const u8, prompt: []const u32, count: usize, seed: ?u64) !Result {
    const context = prompt.len + count + 8;
    const loaded = try llama.native.open(gpu.gpa, gpu.io, gpu.ctx, model, null, .{ .context = context, .drafts = draft != null, .drafter = draft });
    defer loaded.deinit(loaded.ctx);
    var cfg = try lanes.Config.init(gpu.gpa, loaded.facts, if (draft != null) 8 else 1, if (draft != null) 7 else 0);
    defer cfg.deinit(gpu.gpa);
    var clock: lanes.backend.WallClock = .{ .io = gpu.io };
    var engine = lanes.Engine.init(gpu.gpa, &cfg, loaded.backend, clock.clock());
    defer engine.deinit();
    var stream = try lanes.Stream.init(gpu.gpa, .{ .id = "gate", .prompt = prompt, .max_new = @intCast(count), .eos = &.{}, .drafts = draft != null, .sampling = if (seed) |n| .{ .seed = n, .temperature = 0.7, .top_k = 20, .top_p = 0.95 } else null });
    defer stream.deinit(gpu.gpa);
    const t0 = check.now(gpu.io);
    try engine.addStream(&stream);
    const t1 = check.now(gpu.io);
    while (engine.activeCount() > 0) try engine.step();
    const t2 = check.now(gpu.io);
    if (stream.reason != .length or stream.emitted().len != count) return error.ShortGeneration;
    return .{ .tokens = try gpu.gpa.dupe(u32, stream.emitted()), .seconds = @as(f64, @floatFromInt(t2 - t1)) / 1e9, .prefill_seconds = @as(f64, @floatFromInt(t1 - t0)) / 1e9, .rounds = stream.rounds, .drafted = stream.drafted, .accepted = stream.accepted };
}
pub fn run(gpu: check.Gpu, model: []const u8, draft: []const u8, prompts_file: []const u8, output: []const u8, count: usize, seeds: usize, reps: usize) !void {
    const raw = try std.Io.Dir.cwd().readFileAlloc(gpu.io, prompts_file, gpu.gpa, .limited(16 << 20));
    defer gpu.gpa.free(raw);
    const prompts = try std.json.parseFromSlice([][]u32, gpu.gpa, raw, .{});
    defer prompts.deinit();
    if (prompts.value.len != 3 or reps < 5 or count < 256) return error.BadGateShape;
    var records: std.ArrayList(Record) = .empty;
    defer {
        for (records.items) |record| {
            gpu.gpa.free(record.plain.tokens);
            gpu.gpa.free(record.drafted.tokens);
        }
        records.deinit(gpu.gpa);
    }
    for (prompts.value, 0..) |prompt, index| for (0..seeds + 1) |sample| for (0..if (sample == 0) reps else 1) |rep| {
        const seed: ?u64 = if (sample == 0) null else sample - 1;
        var plain: Result = undefined;
        var drafted: Result = undefined;
        if (rep % 2 == 0) {
            plain = try generate(gpu, model, null, prompt, count, seed);
            drafted = try generate(gpu, model, draft, prompt, count, seed);
        } else {
            drafted = try generate(gpu, model, draft, prompt, count, seed);
            plain = try generate(gpu, model, null, prompt, count, seed);
        }
        try records.append(gpu.gpa, .{ .prompt = index, .seed = seed, .rep = rep, .plain = plain, .drafted = drafted });
        try check.expect(std.mem.eql(u32, plain.tokens, drafted.tokens), "draft output differs prompt={d} sample={d} rep={d}", .{ index, sample, rep });
        if (drafted.drafted == 0) return error.DraftDidNotRun;
        std.debug.print("EQUALITY prompt={d} sample={d} rep={d} tokens={d} plain_tps={d:.3} draft_tps={d:.3} drafted={d} accepted={d} rounds={d}\n", .{ index, sample, rep, count, @as(f64, @floatFromInt(count - 1)) / plain.seconds, @as(f64, @floatFromInt(count - 1)) / drafted.seconds, drafted.drafted, drafted.accepted, drafted.rounds });
    };
    const text = try std.json.Stringify.valueAlloc(gpu.gpa, records.items, .{});
    defer gpu.gpa.free(text);
    try std.Io.Dir.cwd().writeFile(gpu.io, .{ .sub_path = output, .data = text });
    std.debug.print("SUMMARY full drafted/plain greedy and keyed sampled equality pairs={d} prompts=3 tokens={d}\n", .{ records.items.len, count });
}
const Record = struct { prompt: usize, seed: ?u64, rep: usize, plain: Result, drafted: Result };
