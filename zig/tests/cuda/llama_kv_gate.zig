const std = @import("std");
const cuda = @import("cuda");
const llama = @import("llama");
const common = @import("llama_tests.zig");
const check = @import("check.zig");
const prefill = @import("llama_prefill.zig");

pub fn dump(gpu: check.Gpu, model: []const u8, refs: []const u8, bits: []const u8, out: []const u8, rows: usize) !void {
    return run(gpu, model, refs, bits, out, rows, 1);
}

pub fn run(gpu: check.Gpu, model: []const u8, refs: []const u8, bits: []const u8, out: []const u8, rows: usize, reps: usize) !void {
    return runLengths(gpu, model, refs, bits, out, rows, reps, false);
}

pub fn runLengths(gpu: check.Gpu, model: []const u8, refs: []const u8, bits: []const u8, out: []const u8, rows: usize, reps: usize, sweep: bool) !void {
    const paired = std.mem.eql(u8, bits, "pair");
    const mode = if (paired) .bf16 else try cuda.kv4.Mode.parse(bits);
    const seqs = try common.sequences(gpu, refs);
    defer seqs.deinit();
    if (seqs.value.len != 1 or reps == 0) return error.BadGateShape;
    const seq = seqs.value[0];
    const context = @max(256, seq.len + 128);
    var w = try llama.weights.loadMode(gpu.gpa, gpu.io, gpu.ctx, model, null, context, mode);
    defer w.deinit();
    var k = try llama.kernels.Kernels.load(gpu.ctx);
    defer k.deinit();
    var s = try cuda.Stream.init(gpu.d, false);
    defer s.deinit();
    var m = try llama.forward.Model.initRows(gpu.gpa, gpu.d, &w, &k, s, context, mode, 8);
    defer m.deinit();
    var q: ?llama.forward.Model = if (paired) try llama.forward.Model.initRows(gpu.gpa, gpu.d, &w, &k, s, context, .q4, 8) else null;
    defer if (q) |*qm| qm.deinit();
    if (rows != 0 and rows != 512) return error.BadScoredRows;
    if (seq.len < rows or seq.len == 0 or seq[0] != w.config.bos) return error.BadReference;
    const logits = try gpu.gpa.alloc(f32, w.config.vocab * rows);
    defer gpu.gpa.free(logits);
    var records: std.ArrayList(Record) = .empty;
    defer records.deinit(gpu.gpa);
    var models = [_]*llama.forward.Model{ &m, if (q) |*qm| qm else &m };
    for (models[0..if (paired) 2 else 1]) |target| {
        try s.synchronize();
        const start = check.now(gpu.io);
        try prefill.run(target, seq, 0, &.{});
        try s.synchronize();
        const seconds = @as(f64, @floatFromInt(check.now(gpu.io) - start)) / 1e9;
        try records.append(gpu.gpa, .{ .mode = @tagName(target.kv_mode), .prefill_seconds = seconds, .prefill_tok_s = @as(f64, @floatFromInt(seq.len)) / seconds });
    }
    const lengths: []const usize = if (sweep) &.{ 100000, 60000, 20000 } else &.{seq.len};
    if (sweep and seq.len != 100000) return error.BadSweepLength;
    for (lengths) |length| {
        const folder = if (sweep) try common.path(gpu, out, "{d}", .{length}) else try gpu.gpa.dupe(u8, out);
        defer gpu.gpa.free(folder);
        for (models[0..if (paired) 2 else 1]) |target| {
            const scores = if (paired) try common.path(gpu, folder, "{s}/0", .{@tagName(target.kv_mode)}) else try gpu.gpa.dupe(u8, folder);
            defer gpu.gpa.free(scores);
            try std.Io.Dir.cwd().createDirPath(gpu.io, scores);
            try prefill.score(target, seq, length, rows, logits);
            const file = try common.path(gpu, scores, "logits-0.f32", .{});
            defer gpu.gpa.free(file);
            try std.Io.Dir.cwd().writeFile(gpu.io, .{ .sub_path = file, .data = std.mem.sliceAsBytes(logits) });
        }
        for (0..reps) |rep| for (0..if (paired) @as(usize, 2) else 1) |side| {
            const index = if (paired and rep % 2 == 1) 1 - side else side;
            const target = models[index];
            // Rewriting the tail leaves the immutable prompt cache resident for every timing pair.
            try target.step(seq[length - 1], length - 1, true);
            var tokens: [128]u32 = undefined;
            tokens[0] = try target.greedy();
            const start = check.now(gpu.io);
            for (1..tokens.len) |i| {
                try target.step(tokens[i - 1], length + i - 1, true);
                tokens[i] = try target.greedy();
            }
            const seconds = @as(f64, @floatFromInt(check.now(gpu.io) - start)) / 1e9;
            const decoded = if (paired) try common.path(gpu, folder, "{s}/{d}", .{ @tagName(target.kv_mode), rep }) else try gpu.gpa.dupe(u8, folder);
            defer gpu.gpa.free(decoded);
            try std.Io.Dir.cwd().createDirPath(gpu.io, decoded);
            const file = try common.path(gpu, decoded, "greedy-0.json", .{});
            defer gpu.gpa.free(file);
            const text = try std.json.Stringify.valueAlloc(gpu.gpa, tokens, .{});
            defer gpu.gpa.free(text);
            try std.Io.Dir.cwd().writeFile(gpu.io, .{ .sub_path = file, .data = text });
            try records.append(gpu.gpa, .{ .mode = @tagName(target.kv_mode), .rep = rep, .history_tokens = length, .decode_tok_s = 127 / seconds });
        };
    }
    const file = try common.path(gpu, out, "timings.json", .{});
    defer gpu.gpa.free(file);
    const text = try std.json.Stringify.valueAlloc(gpu.gpa, records.items, .{});
    defer gpu.gpa.free(text);
    try std.Io.Dir.cwd().writeFile(gpu.io, .{ .sub_path = file, .data = text });
}
const Record = struct { mode: []const u8, rep: ?usize = null, history_tokens: usize = 0, prefill_seconds: f64 = 0, prefill_tok_s: f64 = 0, decode_tok_s: f64 = 0 };
