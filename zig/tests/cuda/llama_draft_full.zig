const std = @import("std");
const llama = @import("llama");
const lanes = @import("lanes");
const check = @import("check.zig");
const temperatures = [_]f32{ 0.3, 0.7, 1.0, 1.3 };
const top_ks = [_]u32{ 0, 3, 20, 50 };
const top_ps = [_]f32{ 1.0, 0.8, 0.95, 0.9 };
const Result = struct {
    tokens: []u32,
    seconds: f64,
    prefill_seconds: f64,
    rounds: u64,
    drafted: u64,
    accepted: u64,
    stopped: bool,
};
fn generate(gpu: check.Gpu, model: []const u8, draft: ?[]const u8, prompt: []const u32, count: usize, seed: ?u64, variant: usize) !Result {
    const context = prompt.len + count + 8;
    const model_config = try llama.Config.read(gpu.gpa, gpu.io, model);
    const loaded = try llama.native.open(gpu.gpa, gpu.io, gpu.ctx, model, null, .{ .context = context, .drafts = draft != null, .drafter = draft });
    defer loaded.deinit(loaded.ctx);
    var cfg = try lanes.Config.init(gpu.gpa, loaded.facts, if (draft != null) 8 else 1, if (draft != null) 7 else 0);
    defer cfg.deinit(gpu.gpa);
    var clock: lanes.backend.WallClock = .{ .io = gpu.io };
    var engine = lanes.Engine.init(gpu.gpa, &cfg, loaded.backend, clock.clock());
    defer engine.deinit();
    var stream = try lanes.Stream.init(gpu.gpa, .{ .id = "gate", .prompt = prompt, .max_new = @intCast(count), .eos = model_config.eos[0..model_config.eos_count], .drafts = draft != null, .penalties = if (variant == 3) .{ .presence = 0.7, .frequency = 0.3 } else .{}, .sampling = if (seed) |n| .{ .seed = n, .temperature = temperatures[variant], .top_k = top_ks[variant], .top_p = top_ps[variant] } else null });
    defer stream.deinit(gpu.gpa);
    const t0 = check.now(gpu.io);
    try engine.addStream(&stream);
    const t1 = check.now(gpu.io);
    while (engine.activeCount() > 0) try engine.step();
    const t2 = check.now(gpu.io);
    if (stream.reason != .length and stream.reason != .stop) return error.ShortGeneration;
    if (stream.emitted().len == 0) return error.ShortGeneration;
    return .{ .stopped = stream.reason == .stop, .tokens = try gpu.gpa.dupe(u32, stream.emitted()), .seconds = @as(f64, @floatFromInt(t2 - t1)) / 1e9, .prefill_seconds = @as(f64, @floatFromInt(t1 - t0)) / 1e9, .rounds = stream.rounds, .drafted = stream.drafted, .accepted = stream.accepted };
}
pub fn run(gpu: check.Gpu, model: []const u8, draft: []const u8, prompts_file: []const u8, output: []const u8, count: usize, samples: usize, reps: usize, checks: bool) !void {
    const raw = try std.Io.Dir.cwd().readFileAlloc(gpu.io, prompts_file, gpu.gpa, .limited(16 << 20));
    defer gpu.gpa.free(raw);
    const prompts = try std.json.parseFromSlice([][]u32, gpu.gpa, raw, .{});
    defer prompts.deinit();
    if (prompts.value.len != 3 or reps < 1 or count < 256 or samples < 16) return error.BadGateShape;
    var records: std.ArrayList(Record) = .empty;
    defer {
        for (records.items) |record| {
            gpu.gpa.free(record.plain.tokens);
            gpu.gpa.free(record.drafted.tokens);
        }
        records.deinit(gpu.gpa);
    }
    var distributions: std.Io.Writer.Allocating = .init(gpu.gpa);
    defer distributions.deinit();
    for (prompts.value, 0..) |prompt, index| for (0..reps) |rep| {
        const seed: ?u64 = null;
        const variant = 0;
        var plain: Result = undefined;
        var drafted: Result = undefined;
        if (rep % 2 == 0) {
            plain = try generate(gpu, model, null, prompt, count, seed, variant);
            drafted = try generate(gpu, model, draft, prompt, count, seed, variant);
        } else {
            drafted = try generate(gpu, model, draft, prompt, count, seed, variant);
            plain = try generate(gpu, model, null, prompt, count, seed, variant);
        }
        if (checks and rep == 0) try @import("llama_draft_forced.zig").run(gpu, model, draft, prompt, plain.tokens);
        try records.append(gpu.gpa, .{ .prompt = index, .seed = seed, .variant = variant, .rep = rep, .plain = plain, .drafted = drafted });
        if (checks and rep == 0) try @import("llama_draft_distribution.zig").run(gpu, model, draft, prompt, plain.tokens, index, samples, &distributions.writer);
        try check.expect(std.mem.eql(u32, plain.tokens, drafted.tokens), "draft output differs prompt={d} rep={d}", .{ index, rep });
        if (plain.stopped != drafted.stopped) return error.FinishMismatch;
        if (drafted.drafted == 0) return error.DraftDidNotRun;
        std.debug.print("EQUALITY prompt={d} rep={d} tokens={d} plain_tps={d:.3} draft_tps={d:.3} drafted={d} accepted={d} rounds={d}\n", .{ index, rep, drafted.tokens.len, @as(f64, @floatFromInt(plain.tokens.len - 1)) / plain.seconds, @as(f64, @floatFromInt(drafted.tokens.len - 1)) / drafted.seconds, drafted.drafted, drafted.accepted, drafted.rounds });
    };
    const stats_file = try std.fmt.allocPrint(gpu.gpa, "{s}.distribution.jsonl", .{output});
    defer gpu.gpa.free(stats_file);
    try std.Io.Dir.cwd().writeFile(gpu.io, .{ .sub_path = stats_file, .data = distributions.written() });
    const text = try std.json.Stringify.valueAlloc(gpu.gpa, records.items, .{});
    defer gpu.gpa.free(text);
    try std.Io.Dir.cwd().writeFile(gpu.io, .{ .sub_path = output, .data = text });
    std.debug.print("SUMMARY full drafted/plain greedy equality and fixed-position distributions pairs={d} prompts=3 tokens={d}\n", .{ records.items.len, count });
}
const Record = struct { prompt: usize, seed: ?u64, rep: usize, variant: usize, plain: Result, drafted: Result };

pub fn profile(gpu: check.Gpu, model: []const u8, draft: []const u8, prompts_file: []const u8, count: usize) !void {
    if (count < 16 or count > 256) return error.BadProfileShape;
    const raw = try std.Io.Dir.cwd().readFileAlloc(gpu.io, prompts_file, gpu.gpa, .limited(16 << 20));
    defer gpu.gpa.free(raw);
    const prompts = try std.json.parseFromSlice([][]u32, gpu.gpa, raw, .{});
    defer prompts.deinit();
    if (prompts.value.len == 0 or prompts.value[0].len == 0) return error.BadProfileShape;
    const result = try generate(gpu, model, draft, prompts.value[0], count, null, 0);
    defer gpu.gpa.free(result.tokens);
    std.debug.print("DFLASH_PROBE tokens={d} rounds={d} proposed={d} accepted={d} decode_s={d:.3}\n", .{ result.tokens.len, result.rounds, result.drafted, result.accepted, result.seconds });
}
