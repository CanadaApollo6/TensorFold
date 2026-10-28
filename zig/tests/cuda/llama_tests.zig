const std = @import("std");
const cuda = @import("cuda");
const llama = @import("llama");
const lanes = @import("lanes");
const check = @import("check.zig");
const Gpu = check.Gpu;

const Rig = struct {
    gpu: Gpu,
    w: llama.weights.Weights,
    k: llama.kernels.Kernels,
    s: cuda.Stream,
    m: llama.forward.Model,

    fn init(r: *Rig, gpu: Gpu, dir: []const u8, count: ?usize, context: usize) !void {
        r.gpu = gpu;
        r.w = try llama.weights.load(gpu.gpa, gpu.io, gpu.ctx, dir, count, context);
        errdefer r.w.deinit();
        r.k = try llama.kernels.Kernels.load(gpu.ctx);
        errdefer r.k.deinit();
        r.s = try cuda.Stream.init(gpu.d, false);
        errdefer r.s.deinit();
        r.m = try llama.forward.Model.init(gpu.gpa, gpu.d, &r.w, &r.k, r.s, context);
    }

    fn deinit(r: *Rig) void {
        r.m.deinit();
        r.s.deinit();
        r.k.deinit();
        r.w.deinit();
    }
};

fn path(gpu: Gpu, dir: []const u8, comptime fmt: []const u8, args: anytype) ![]u8 {
    const name = try std.fmt.allocPrint(gpu.gpa, fmt, args);
    defer gpu.gpa.free(name);
    return std.fs.path.join(gpu.gpa, &.{ dir, name });
}

fn sequences(gpu: Gpu, dir: []const u8) !std.json.Parsed([][]u32) {
    const file = try path(gpu, dir, "sequences.json", .{});
    defer gpu.gpa.free(file);
    const text = try std.Io.Dir.cwd().readFileAlloc(gpu.io, file, gpu.gpa, .limited(1 << 22));
    defer gpu.gpa.free(text);
    const parsed = try std.json.parseFromSlice([][]u32, gpu.gpa, text, .{ .allocate = .alloc_always });
    errdefer parsed.deinit();
    if (parsed.value.len == 0) return error.BadReference;
    for (parsed.value) |seq| if (seq.len == 0) return error.BadReference;
    return parsed;
}

fn floats(gpu: Gpu, file: []const u8) ![]f32 {
    const bytes = try std.Io.Dir.cwd().readFileAlloc(gpu.io, file, gpu.gpa, .limited(1 << 30));
    defer gpu.gpa.free(bytes);
    if (bytes.len % 4 != 0) return error.BadReference;
    const out = try gpu.gpa.alloc(f32, bytes.len / 4);
    for (out, 0..) |*v, i| v.* = @bitCast(std.mem.readInt(u32, bytes[4 * i ..][0..4], .little));
    return out;
}

fn write(gpu: Gpu, dir: []const u8, comptime fmt: []const u8, args: anytype, bytes: []const u8) !void {
    const file = try path(gpu, dir, fmt, args);
    defer gpu.gpa.free(file);
    try std.Io.Dir.cwd().writeFile(gpu.io, .{ .sub_path = file, .data = bytes });
}

const Metrics = struct { kl: f64, top1: bool, max_abs: f64, rmse: f64, ref_top: usize, engine_top: usize, ref_margin: f64, engine_margin: f64 };

fn metrics(ref: []const f32, got: []const f32) !Metrics {
    if (ref.len != got.len or ref.len < 2) return error.BadReference;
    var ri: usize = 0;
    var gi: usize = 0;
    var max_abs: f64 = 0;
    var sq: f64 = 0;
    for (ref, got, 0..) |rv, gv, i| {
        if (!std.math.isFinite(rv) or !std.math.isFinite(gv)) return error.NonFinite;
        if (rv > ref[ri]) ri = i;
        if (gv > got[gi]) gi = i;
        const diff = @as(f64, rv) - gv;
        max_abs = @max(max_abs, @abs(diff));
        sq += diff * diff;
    }
    var rs: f64 = 0;
    var gs: f64 = 0;
    var rsecond = -std.math.inf(f32);
    var gsecond = -std.math.inf(f32);
    for (ref, got, 0..) |r, g, i| {
        rs += @exp(@as(f64, r) - ref[ri]);
        gs += @exp(@as(f64, g) - got[gi]);
        if (i != ri) rsecond = @max(rsecond, r);
        if (i != gi) gsecond = @max(gsecond, g);
    }
    const rz = @as(f64, ref[ri]) + @log(rs);
    const gz = @as(f64, got[gi]) + @log(gs);
    var kl: f64 = 0;
    for (ref, got) |r, g| {
        const lp = @as(f64, r) - rz;
        kl += @exp(lp) * (lp - (@as(f64, g) - gz));
    }
    return .{ .kl = @max(0, kl), .top1 = ri == gi, .max_abs = max_abs, .rmse = @sqrt(sq / @as(f64, @floatFromInt(ref.len))), .ref_top = ri, .engine_top = gi, .ref_margin = ref[ri] - rsecond, .engine_margin = got[gi] - gsecond };
}

pub fn layers(gpu: Gpu, model: []const u8, refs: []const u8, count_text: ?[]const u8, out: []const u8) !void {
    const count = if (count_text) |s| try std.fmt.parseInt(usize, s, 10) else (try llama.Config.read(gpu.gpa, gpu.io, model)).layers;
    const seqs = try sequences(gpu, refs);
    defer seqs.deinit();
    var context: usize = 1;
    for (seqs.value) |seq| context = @max(context, seq.len + 16);
    context = (context + 255) / 256 * 256;
    try std.Io.Dir.cwd().createDirPath(gpu.io, out);
    var r: Rig = undefined;
    try r.init(gpu, model, count, context);
    defer r.deinit();
    const c = r.w.config;
    var matched: usize = 0;
    var total: usize = 0;
    var max_kl: f64 = 0;
    var sum_kl: f64 = 0;
    var high: usize = 0;
    var report: std.Io.Writer.Allocating = .init(gpu.gpa);
    defer report.deinit();
    for (seqs.value, 0..) |seq, si| {
        const states = try gpu.gpa.alloc(f32, count * seq.len * c.hidden);
        defer gpu.gpa.free(states);
        const logits = try gpu.gpa.alloc(f32, seq.len * c.vocab);
        defer gpu.gpa.free(logits);
        for (seq, 0..) |token, pos| {
            try r.m.begin(token, pos);
            for (0..count) |li| {
                try r.m.layer(li);
                try r.m.download(r.m.res, std.mem.sliceAsBytes(states[(li * seq.len + pos) * c.hidden ..][0..c.hidden]));
            }
            try r.m.head();
            try r.m.download(r.m.logits, std.mem.sliceAsBytes(logits[pos * c.vocab ..][0..c.vocab]));
        }
        for (0..count) |li| {
            const file = try path(gpu, refs, "ref-{d}-s{d}-layer{d}.f32", .{ count, si, li });
            defer gpu.gpa.free(file);
            const ref = try floats(gpu, file);
            defer gpu.gpa.free(ref);
            if (ref.len != seq.len * c.hidden) return error.BadReference;
            const got = states[li * seq.len * c.hidden ..][0 .. seq.len * c.hidden];
            try write(gpu, out, "engine-{d}-s{d}-layer{d}.f32", .{ count, si, li }, std.mem.sliceAsBytes(got));
            var worst: f64 = 0;
            var squared_error: f64 = 0;
            var squared_ref: f64 = 0;
            for (ref, got) |rv, gv| {
                if (!std.math.isFinite(rv) or !std.math.isFinite(gv)) return error.NonFinite;
                const diff = @as(f64, rv) - gv;
                worst = @max(worst, @abs(diff));
                squared_error += diff * diff;
                squared_ref += @as(f64, rv) * rv;
            }
            const rms = @sqrt(squared_error / @as(f64, @floatFromInt(ref.len)));
            const relative = @sqrt(squared_error / @max(squared_ref, 1e-30));
            std.debug.print("LAYER sequence={d} layer={d} rmse={e} max_abs={e} relative_l2={e}\n", .{ si, li, rms, worst, relative });
        }
        const file = try path(gpu, refs, "ref-{d}-s{d}-logits.f32", .{ count, si });
        defer gpu.gpa.free(file);
        const ref = try floats(gpu, file);
        defer gpu.gpa.free(ref);
        if (ref.len != logits.len) return error.BadReference;
        try write(gpu, out, "engine-{d}-s{d}-logits.f32", .{ count, si }, std.mem.sliceAsBytes(logits));
        for (0..seq.len) |pos| {
            const met = try metrics(ref[pos * c.vocab ..][0..c.vocab], logits[pos * c.vocab ..][0..c.vocab]);
            try report.writer.print("{{\"sequence\":{d},\"position\":{d},\"kl\":{e},\"top1\":{},\"ref_top\":{d},\"engine_top\":{d},\"ref_margin\":{e},\"engine_margin\":{e},\"max_abs\":{e}}}\n", .{ si, pos, met.kl, met.top1, met.ref_top, met.engine_top, met.ref_margin, met.engine_margin, met.max_abs });
            matched += @intFromBool(met.top1);
            total += 1;
            sum_kl += met.kl;
            max_kl = @max(max_kl, met.kl);
            high += @intFromBool(met.kl > 0.01);
        }
    }
    try write(gpu, out, "metrics-{d}.jsonl", .{count}, report.written());
    var tokens: [8]u32 = undefined;
    try generate(&r, seqs.value[0], &tokens);
    try laneCheck(&r, seqs.value[0], &tokens);
    const u = cuda.usage(false);
    std.debug.print("SUMMARY llama layers={d} positions={d} mean_kl={e} max_kl={e} top1={d}/{d} kl_above_0.01={d} device_peak={d} pinned={d}\n", .{ count, total, sum_kl / @as(f64, @floatFromInt(total)), max_kl, matched, total, high, u.peak, u.host });
    if (high != 0) return error.LlamaLogitDrift;
}

fn generate(r: *Rig, prompt: []const u32, out: []u32) !void {
    if (prompt.len + out.len > r.m.context or out.len == 0) return error.PastContext;
    for (prompt, 0..) |token, pos| try r.m.step(token, pos, pos + 1 == prompt.len);
    out[0] = try r.m.greedy();
    for (1..out.len) |i| {
        try r.m.step(out[i - 1], prompt.len + i - 1, true);
        out[i] = try r.m.greedy();
    }
}

pub fn greedy(gpu: Gpu, model: []const u8, refs: []const u8, count_text: []const u8, out: []const u8) !void {
    const seqs = try sequences(gpu, refs);
    defer seqs.deinit();
    if (seqs.value.len == 0) return error.BadReference;
    const prompt = seqs.value[seqs.value.len - 1];
    const count = try std.fmt.parseInt(usize, count_text, 10);
    var r: Rig = undefined;
    try r.init(gpu, model, null, @max(4096, prompt.len + count));
    defer r.deinit();
    const tokens = try gpu.gpa.alloc(u32, count);
    defer gpu.gpa.free(tokens);
    try generate(&r, prompt, tokens);
    var text: std.Io.Writer.Allocating = .init(gpu.gpa);
    defer text.deinit();
    try text.writer.print("{f}\n", .{std.json.fmt(tokens, .{})});
    try std.Io.Dir.cwd().writeFile(gpu.io, .{ .sub_path = out, .data = text.written() });
    const file = try path(gpu, refs, "ref-{d}-greedy.json", .{r.w.config.layers});
    defer gpu.gpa.free(file);
    const bytes = try std.Io.Dir.cwd().readFileAlloc(gpu.io, file, gpu.gpa, .limited(1 << 20));
    defer gpu.gpa.free(bytes);
    const ref = try std.json.parseFromSlice([]u32, gpu.gpa, bytes, .{});
    defer ref.deinit();
    if (ref.value.len != count) return error.BadReference;
    const divergence = std.mem.indexOfDiff(u32, ref.value, tokens);
    std.debug.print("SUMMARY llama greedy tokens={d} first_divergence={any} device_peak={d}\n", .{ count, divergence, cuda.usage(false).peak });
    try laneCheck(&r, prompt, tokens);
    if (divergence != null) return error.LlamaGreedyDiverged;
}

fn laneCheck(r: *Rig, prompt: []const u32, expected: []const u32) !void {
    var b: llama.backend.Llama = .{ .m = &r.m };
    var cfg = try lanes.Config.init(r.gpu.gpa, b.facts(), 2, 1);
    defer cfg.deinit(r.gpu.gpa);
    var clock: lanes.fake.FixedClock = .{};
    var engine = lanes.Engine.init(r.gpu.gpa, &cfg, b.backend(), clock.clock());
    defer engine.deinit();
    var stream = try lanes.Stream.init(r.gpu.gpa, .{ .id = "llama-p1", .prompt = prompt, .max_new = @intCast(expected.len), .drafts = false });
    defer stream.deinit(r.gpu.gpa);
    try engine.addStream(&stream);
    while (engine.activeCount() > 0) try engine.step();
    try check.expect(std.mem.eql(u32, expected, stream.emitted()), "Llama lane backend differs from serial greedy", .{});
    check.pass("Llama lane backend equals serial greedy for {d} tokens", .{expected.len});
}

pub fn bench(gpu: Gpu, model: []const u8, refs: []const u8) !void {
    const seqs = try sequences(gpu, refs);
    defer seqs.deinit();
    if (seqs.value.len == 0) return error.BadReference;
    const chat = seqs.value[seqs.value.len - 1];
    const prompt = chat[0..@min(chat.len, 32)];
    var r: Rig = undefined;
    try r.init(gpu, model, null, 4096);
    defer r.deinit();
    var warm: [128]u32 = undefined;
    try generate(&r, prompt, &warm);
    var rates: [3]f64 = undefined;
    for (&rates, 0..) |*rate, rep| {
        for (prompt, 0..) |token, pos| try r.m.step(token, pos, pos + 1 == prompt.len);
        var tokens: [128]u32 = undefined;
        try r.s.synchronize();
        const start = check.now(gpu.io);
        tokens[0] = try r.m.greedy();
        for (1..tokens.len) |i| {
            try r.m.step(tokens[i - 1], prompt.len + i - 1, true);
            tokens[i] = try r.m.greedy();
        }
        const seconds = @as(f64, @floatFromInt(check.now(gpu.io) - start)) / 1e9;
        rate.* = 127.0 / seconds;
        try check.expect(std.mem.eql(u32, &tokens, &warm), "Llama benchmark repetition changed tokens", .{});
        std.debug.print("BENCH rep={d} forwards=127 tokens=128 seconds={d:.6} tok_s={d:.3}\n", .{ rep, seconds, rate.* });
    }
    const median = check.median(&rates);
    const u = cuda.usage(false);
    std.debug.print("SUMMARY llama bench warmups=1 repetitions=3 median_tok_s={d:.3} range={d:.3}..{d:.3} context=4096 prompt_tokens={d} device_peak={d} pinned={d}\n", .{ median, rates[0], rates[2], prompt.len, u.peak, u.host });
}
