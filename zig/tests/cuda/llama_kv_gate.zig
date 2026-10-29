const std = @import("std");
const cuda = @import("cuda");
const llama = @import("llama");
const common = @import("llama_tests.zig");
const check = @import("check.zig");

pub fn dump(gpu: check.Gpu, model: []const u8, refs: []const u8, bits: []const u8, out: []const u8, rows: usize) !void {
    const mode = try cuda.kv4.Mode.parse(bits);
    const seqs = try common.sequences(gpu, refs);
    defer seqs.deinit();
    var context: usize = 256;
    for (seqs.value) |seq| context = @max(context, seq.len + 128);
    var w = try llama.weights.loadMode(gpu.gpa, gpu.io, gpu.ctx, model, null, context, mode);
    defer w.deinit();
    var k = try llama.kernels.Kernels.load(gpu.ctx);
    defer k.deinit();
    var s = try cuda.Stream.init(gpu.d, false);
    defer s.deinit();
    var m = try llama.forward.Model.initMode(gpu.gpa, gpu.d, &w, &k, s, context, mode);
    defer m.deinit();
    try std.Io.Dir.cwd().createDirPath(gpu.io, out);
    if (rows != 0 and rows != 512) return error.BadScoredRows;
    const logits = try gpu.gpa.alloc(f32, w.config.vocab * rows);
    defer gpu.gpa.free(logits);
    for (seqs.value, 0..) |seq, si| {
        if (seq.len < rows or seq[0] != w.config.bos) return error.BadReference;
        try s.synchronize();
        const start = check.now(gpu.io);
        for (seq, 0..) |token, pos| {
            const scored = pos + rows >= seq.len;
            try m.step(token, pos, scored);
            if (scored) try m.download(m.logits, std.mem.sliceAsBytes(logits[(pos + rows - seq.len) * w.config.vocab ..][0..w.config.vocab]));
        }
        try s.synchronize();
        const prefill_s = @as(f64, @floatFromInt(check.now(gpu.io) - start)) / 1e9;
        const file = try common.path(gpu, out, "logits-{d}.f32", .{si});
        defer gpu.gpa.free(file);
        try std.Io.Dir.cwd().writeFile(gpu.io, .{ .sub_path = file, .data = std.mem.sliceAsBytes(logits) });
        var tokens: [128]u32 = undefined;
        tokens[0] = try m.greedy();
        const decode_start = check.now(gpu.io);
        for (1..tokens.len) |i| {
            try m.step(tokens[i - 1], seq.len + i - 1, true);
            tokens[i] = try m.greedy();
        }
        const decode_s = @as(f64, @floatFromInt(check.now(gpu.io) - decode_start)) / 1e9;
        const token_file = try common.path(gpu, out, "greedy-{d}.json", .{si});
        defer gpu.gpa.free(token_file);
        var text: std.Io.Writer.Allocating = .init(gpu.gpa);
        defer text.deinit();
        try text.writer.print("{f}\n", .{std.json.fmt(&tokens, .{})});
        try std.Io.Dir.cwd().writeFile(gpu.io, .{ .sub_path = token_file, .data = text.written() });
        std.debug.print("QUALITY mode={s} sequence={d} prompt={d} scored={d} prefill_seconds={d:.6} prefill_tok_s={d:.3} decode_forwards=127 decode_seconds={d:.6} decode_tok_s={d:.3}\n", .{ @tagName(mode), si, seq.len, rows, prefill_s, @as(f64, @floatFromInt(seq.len)) / prefill_s, decode_s, 127 / decode_s });
    }
    std.debug.print("SUMMARY kv-quality mode={s} sequences={d} context={d} peak_bytes={d}\n", .{ @tagName(mode), seqs.value.len, context, cuda.usage(false).peak });
}
