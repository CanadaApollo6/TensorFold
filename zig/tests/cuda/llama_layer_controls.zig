const std = @import("std");
const cuda = @import("cuda");
const llama = @import("llama");
const check = @import("check.zig");
const common = @import("llama_tests.zig");

const Control = enum { clean, zeros, rope_theta, norm_plus_one, swapped_kv };

pub fn run(gpu: check.Gpu, model: []const u8, refs: []const u8, name: []const u8, out: []const u8) !void {
    const control = std.meta.stringToEnum(Control, name) orelse return error.BadControl;
    const seqs = try common.sequences(gpu, refs);
    defer seqs.deinit();
    var context: usize = 256;
    for (seqs.value) |seq| context = @max(context, (seq.len + 255) / 256 * 256);
    var r: common.Rig = undefined;
    try r.init(gpu, model, 12, context);
    defer r.deinit();
    const c = r.w.config;
    const state = try gpu.gpa.alloc(f32, c.hidden);
    defer gpu.gpa.free(state);
    var inv: [64]f32 = undefined;
    var bad_inv: [64]f32 = undefined;
    try r.m.download(r.m.inv, std.mem.asBytes(&inv));
    for (&bad_inv, 0..) |*v, i| v.* = 1.0 / std.math.pow(f32, 10000, @as(f32, @floatFromInt(2 * i)) / 128.0);
    var norm: ?cuda.DeviceBuffer = null;
    defer if (norm) |*b| b.free();
    switch (control) {
        .norm_plus_one => {
            try r.m.download(r.w.layers[11].input_norm, std.mem.sliceAsBytes(state));
            for (state) |*v| v.* += 1;
            norm = try cuda.DeviceBuffer.fromHost(gpu.d, std.mem.sliceAsBytes(state));
            r.w.layers[11].input_norm = norm.?.ptr;
        },
        .swapped_kv => std.mem.swap(cuda.exl3.Weight, &r.w.layers[11].k, &r.w.layers[11].v),
        else => {},
    }
    var report: std.Io.Writer.Allocating = .init(gpu.gpa);
    defer report.deinit();
    var worst_l2: f64 = 0;
    var worst_scaled: f64 = 0;
    var legacy_failures: usize = 0;
    var positions: usize = 0;
    var failures: usize = 0;
    for (seqs.value, 0..) |seq, si| {
        const file = try common.path(gpu, refs, "ref-12-s{d}-layer11.f32", .{si});
        defer gpu.gpa.free(file);
        const ref = try common.floats(gpu, file);
        defer gpu.gpa.free(ref);
        if (ref.len != seq.len * c.hidden) return error.BadReference;
        if (control == .zeros) @memset(ref, 0);
        const got = try gpu.gpa.alloc(f32, ref.len);
        defer gpu.gpa.free(got);
        for (seq, 0..) |token, pos| {
            try r.m.begin(token, pos);
            for (0..12) |li| {
                if (control == .rope_theta and li == 11) try r.m.upload(r.m.inv, std.mem.asBytes(&bad_inv));
                try r.m.layer(li);
            }
            try r.m.download(r.m.res, std.mem.sliceAsBytes(got[pos * c.hidden ..][0..c.hidden]));
            if (control == .rope_theta) try r.m.upload(r.m.inv, std.mem.asBytes(&inv));
            const met = try llama.validation.layer(ref[pos * c.hidden ..][0..c.hidden], got[pos * c.hidden ..][0..c.hidden]);
            const scaled = met.max_abs / @max(met.ref_peak, 1e-30);
            worst_l2 = @max(worst_l2, met.relative_l2);
            worst_scaled = @max(worst_scaled, scaled);
            positions += 1;
            llama.validation.require(met) catch {
                failures += 1;
            };
            try report.writer.print("{{\"control\":\"{s}\",\"sequence\":{d},\"position\":{d},\"relative_l2\":{e},\"max_abs\":{e},\"ref_peak\":{e},\"scaled_max\":{e}}}\n", .{ name, si, pos, met.relative_l2, met.max_abs, met.ref_peak, scaled });
        }
        const legacy = try llama.validation.layer(ref, got);
        legacy_failures += @intFromBool(legacy.rmse > 1 or legacy.max_abs > 1e-4 + 0.01 * legacy.ref_peak or legacy.relative_l2 > 0.01);
        std.debug.print("CONTROL control={s} sequence={d} legacy_rmse={e} legacy_max={e} legacy_l2={e}\n", .{ name, si, legacy.rmse, legacy.max_abs, legacy.relative_l2 });
    }
    try std.Io.Dir.cwd().writeFile(gpu.io, .{ .sub_path = out, .data = report.written() });
    std.debug.print("SUMMARY llama control={s} layer=11 positions={d} worst_relative_l2={e} worst_scaled_max={e} relative_l2_bound={e} max_error_fraction={e} failed_positions={d} legacy_failures={d}\n", .{ name, positions, worst_l2, worst_scaled, llama.validation.relative_l2_bound, llama.validation.max_error_fraction, failures, legacy_failures });
    if (failures != 0) return error.LlamaLayerDrift;
}
