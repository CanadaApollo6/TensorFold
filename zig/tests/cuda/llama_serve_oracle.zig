const std = @import("std");
const check = @import("check.zig");
const llama = @import("llama_tests.zig");

pub fn run(gpu: check.Gpu, model: []const u8, refs: []const u8, count_text: []const u8, out: []const u8) !void {
    const seqs = try llama.sequences(gpu, refs);
    defer seqs.deinit();
    if (seqs.value.len == 0) return error.BadReference;
    const prompt = seqs.value[seqs.value.len - 1];
    const count = try std.fmt.parseInt(usize, count_text, 10);
    var rig: llama.Rig = undefined;
    try rig.init(gpu, model, null, @max(4096, prompt.len + count));
    defer rig.deinit();
    const tokens = try gpu.gpa.alloc(u32, count);
    defer gpu.gpa.free(tokens);
    try llama.generate(&rig, prompt, tokens);
    const bytes = try std.json.Stringify.valueAlloc(gpu.gpa, tokens, .{});
    defer gpu.gpa.free(bytes);
    try std.Io.Dir.cwd().writeFile(gpu.io, .{ .sub_path = out, .data = bytes });
    std.debug.print("SUMMARY llama serve oracle tokens={d} context={d}\n", .{ count, rig.m.context });
}
