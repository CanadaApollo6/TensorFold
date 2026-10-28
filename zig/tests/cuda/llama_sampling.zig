const std = @import("std");
const cuda = @import("cuda");
const llama = @import("llama");
const lanes = @import("lanes");
const check = @import("check.zig");
const tests = @import("llama_tests.zig");

const Case = struct { name: []const u8, sampling: lanes.Sampling, penalties: lanes.Penalties = .{}, wide: bool = false };
const cases = [_]Case{
    .{ .name = "pilot", .sampling = .{ .seed = 1234, .temperature = 1, .top_p = 0.95, .top_k = 0 } },
    .{ .name = "temperature", .sampling = .{ .seed = 1234, .temperature = 0.7, .top_p = 1, .top_k = 0 } },
    .{ .name = "top_k", .sampling = .{ .seed = 1234, .temperature = 1, .top_p = 1, .top_k = 3 } },
    .{ .name = "min_p", .sampling = .{ .seed = 1234, .temperature = 1, .top_p = 1, .top_k = 0, .min_p = 0.2 } },
    .{ .name = "combined", .sampling = .{ .seed = 1234, .temperature = 0.75, .top_p = 0.9, .top_k = 5, .min_p = 0.2 } },
    .{ .name = "penalties", .sampling = .{ .seed = 1234, .temperature = 1, .top_p = 0.95, .top_k = 0 }, .penalties = .{ .presence = 0.5, .frequency = 0.25 } },
    .{ .name = "wide_nucleus", .sampling = .{ .seed = 1234, .temperature = 1, .top_p = 0.95, .top_k = 0 }, .wide = true },
};

fn pick(sampler: *llama.sampling.Sampler, k: *const llama.kernels.Kernels, s: cuda.Stream, logits: u64, token: *cuda.DeviceBuffer, draw: llama.sampling.Draw) !u32 {
    try sampler.draw(k, s, logits, token.ptr, draw);
    try s.synchronize();
    var out: u32 = undefined;
    try token.download(0, std.mem.asBytes(&out));
    return out;
}

pub fn distribution(gpu: check.Gpu, out: []const u8) !void {
    var k = try llama.kernels.Kernels.load(gpu.ctx);
    defer k.deinit();
    var s = try cuda.Stream.init(gpu.d, false);
    defer s.deinit();
    var logits = try cuda.DeviceBuffer.alloc(gpu.d, 2048 * 4);
    defer logits.free();
    var token = try cuda.DeviceBuffer.alloc(gpu.d, 4);
    defer token.free();
    var sampler = try llama.sampling.Sampler.init(gpu.gpa, gpu.d, 2048, 32);
    defer sampler.deinit();
    var data: std.Io.Writer.Allocating = .init(gpu.gpa);
    defer data.deinit();
    for (cases) |case| {
        sampler.vocab = if (case.wide) 2048 else 8;
        var values: [2048]f32 = @splat(0);
        if (!case.wide) for (values[0..8], 0..) |*v, i| {
            v.* = @as(f32, @floatFromInt(i)) * 0.5;
        };
        try logits.upload(0, std.mem.sliceAsBytes(values[0..sampler.vocab]));
        var counts: [2048]u32 = @splat(0);
        const draws: usize = if (case.wide) 8192 else 32768;
        for (0..draws) |i| {
            const id = try pick(&sampler, &k, s, logits.ptr, &token, .{
                .position = @intCast(i + 1),
                .sampling = case.sampling,
                .penalties = case.penalties,
                .history = &.{ 7, 7, 6 },
                .pending = 7,
            });
            if (id >= sampler.vocab) return error.BadSample;
            counts[id] += 1;
        }
        try data.writer.print("{f}\n", .{std.json.fmt(.{ .name = case.name, .sampling = case.sampling, .penalties = case.penalties, .logits = values[0..sampler.vocab], .counts = counts[0..sampler.vocab], .draws = draws }, .{})});
        std.debug.print("SAMPLED {s} draws={d}\n", .{ case.name, draws });
    }
    sampler.vocab = 8;
    const values = [_]f32{ 1.0001, 1.0002, 0, 0, 0, 0, 0, 0 };
    try logits.upload(0, std.mem.sliceAsBytes(&values));
    for (0..128) |i| {
        const id = try pick(&sampler, &k, s, logits.ptr, &token, .{ .position = @intCast(i), .sampling = .{ .seed = i, .temperature = 0 } });
        if (id != 1) return error.GreedyRounded;
    }
    const penalized = [_]f32{ 0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5 };
    try logits.upload(0, std.mem.sliceAsBytes(&penalized));
    const no_history = try pick(&sampler, &k, s, logits.ptr, &token, .{ .position = 8, .penalties = cases[5].penalties });
    if (no_history != 7) return error.PromptPenalized;
    const with_history = try pick(&sampler, &k, s, logits.ptr, &token, .{ .position = 8, .penalties = cases[5].penalties, .history = &.{ 7, 7, 6 }, .pending = 7 });
    if (with_history != 5) return error.GreedyPenaltyMismatch;
    const equal = [_]f32{ 0, 0, 0, 0, 0, 0, 0, 0 };
    try logits.upload(0, std.mem.sliceAsBytes(&equal));
    var same: usize = 0;
    var different: usize = 0;
    for (0..256) |i| {
        const draw: llama.sampling.Draw = .{ .position = @intCast(i), .sampling = cases[0].sampling };
        const first = try pick(&sampler, &k, s, logits.ptr, &token, draw);
        const second = try pick(&sampler, &k, s, logits.ptr, &token, draw);
        var other = draw;
        other.sampling.?.seed += 1;
        const third = try pick(&sampler, &k, s, logits.ptr, &token, other);
        same += @intFromBool(first == second);
        different += @intFromBool(first != third);
    }
    if (same != 256 or different == 0) return error.SeedMismatch;
    try std.Io.Dir.cwd().writeFile(gpu.io, .{ .sub_path = out, .data = data.written() });
    std.debug.print("SUMMARY llama sampling greedy_fp32=128/128 same_seed={d}/256 different_seed={d}/256 cases={d}\n", .{ same, different, cases.len });
}

pub fn decode(gpu: check.Gpu, model: []const u8, refs: []const u8, out: []const u8) !void {
    const seqs = try tests.sequences(gpu, refs);
    defer seqs.deinit();
    const prompt = seqs.value[seqs.value.len - 1];
    var rig: tests.Rig = undefined;
    try rig.init(gpu, model, null, 4096);
    defer rig.deinit();
    var data: std.Io.Writer.Allocating = .init(gpu.gpa);
    defer data.deinit();
    for ([_]lanes.Penalties{ .{}, .{ .presence = 1, .frequency = 0.5 } }) |penalties| {
        var tokens: [32]u32 = undefined;
        const sampling = cases[0].sampling;
        for (prompt, 0..) |t, i| try rig.m.step(t, i, i + 1 == prompt.len);
        for (&tokens, 0..) |*t, i| {
            if (i > 0) try rig.m.step(tokens[i - 1], prompt.len + i - 1, true);
            t.* = try rig.m.pick(.{ .position = @intCast(prompt.len + i), .sampling = sampling, .penalties = penalties, .history = tokens[0..i] });
        }
        var backend: llama.backend.Llama = .{ .m = &rig.m };
        var cfg = try lanes.Config.init(gpu.gpa, backend.facts(), 2, 1);
        defer cfg.deinit(gpu.gpa);
        var clock: lanes.fake.FixedClock = .{};
        var engine = lanes.Engine.init(gpu.gpa, &cfg, backend.backend(), clock.clock());
        defer engine.deinit();
        var stream = try lanes.Stream.init(gpu.gpa, .{ .id = "sampled", .prompt = prompt, .max_new = tokens.len, .drafts = false, .sampling = sampling, .penalties = penalties });
        defer stream.deinit(gpu.gpa);
        try engine.addStream(&stream);
        while (engine.activeCount() > 0) try engine.step();
        if (!std.mem.eql(u32, &tokens, stream.emitted())) return error.SampledLaneMismatch;
        var verified = try lanes.Stream.init(gpu.gpa, .{ .id = "verified", .prompt = prompt, .max_new = tokens.len, .drafts = false, .sampling = sampling, .penalties = penalties });
        defer verified.deinit(gpu.gpa);
        const be = backend.backend();
        try be.prefill(&verified);
        defer be.release(&verified);
        const first = try be.read(try be.first(&verified, prompt.len));
        if (first != tokens[0]) return error.SampledFirstMismatch;
        _ = try verified.commit(gpu.gpa, &.{first}, &.{});
        for (tokens[1..], 1..) |want, i| {
            const windows = [_]lanes.backend.Window{.{ .stream = &verified, .pending = tokens[i - 1], .held = 0, .tokens = &.{}, .parents = null, .positions = &.{prompt.len + i} }};
            var sampled: [1]u32 = undefined;
            var result = [_]lanes.backend.Verified{.{ .sampled = &sampled, .drafts = &.{} }};
            try be.verify(&windows, &result);
            try be.keep(&windows, &.{&.{0}});
            if (sampled[0] != want) return error.SampledVerifyMismatch;
            _ = try verified.commit(gpu.gpa, &sampled, &.{});
        }
        try data.writer.print("{f}\n", .{std.json.fmt(.{ .sampling = sampling, .penalties = penalties, .tokens = tokens }, .{})});
    }
    try std.Io.Dir.cwd().writeFile(gpu.io, .{ .sub_path = out, .data = data.written() });
    std.debug.print("SUMMARY llama sampled serial=lane=verify tokens=64 penalties=off,on\n", .{});
}
