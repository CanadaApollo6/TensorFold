const std = @import("std");
const cuda = @import("cuda");
const llama = @import("llama");
const check = @import("check.zig");
const common = @import("llama_tests.zig");

pub fn run(gpu: check.Gpu, model: []const u8, refs: []const u8, out: []const u8) !void {
    const positions = [_]usize{ 511, 512, 2047, 2048, 16383 };
    const seqs = try common.sequences(gpu, refs);
    defer seqs.deinit();
    if (seqs.value.len != 1 or seqs.value[0].len != 16384) return error.BadReference;
    var r: common.Rig = undefined;
    try r.init(gpu, model, 2, 16384);
    defer r.deinit();
    const c = r.w.config;
    if (c.heads != 56 or c.kv_heads != 8 or c.head_dim != 128) return error.BadReference;
    try std.Io.Dir.cwd().createDirPath(gpu.io, out);
    const state = try gpu.gpa.alloc(f32, c.hidden);
    defer gpu.gpa.free(state);
    const logits = try gpu.gpa.alloc(f32, c.vocab);
    defer gpu.gpa.free(logits);
    for (seqs.value[0], 0..) |token, pos| {
        try r.m.begin(token, pos);
        for (0..2) |li| {
            try r.m.layer(li);
            if (std.mem.indexOfScalar(usize, &positions, pos) != null) {
                try r.m.download(r.m.res, std.mem.sliceAsBytes(state));
                const file = try common.path(gpu, refs, "boundary-p{d}-layer{d}.f32", .{ pos, li });
                defer gpu.gpa.free(file);
                const ref = try common.floats(gpu, file);
                defer gpu.gpa.free(ref);
                const saved = try common.path(gpu, out, "engine-boundary-p{d}-layer{d}.f32", .{ pos, li });
                defer gpu.gpa.free(saved);
                try std.Io.Dir.cwd().writeFile(gpu.io, .{ .sub_path = saved, .data = std.mem.sliceAsBytes(state) });
                const met = try llama.validation.layer(ref, state);
                std.debug.print("BOUNDARY position={d} layer={d} rmse={e} max_abs={e} relative_l2={e}\n", .{ pos, li, met.rmse, met.max_abs, met.relative_l2 });
                try llama.validation.require(met);
            }
        }
        if (std.mem.indexOfScalar(usize, &positions, pos) != null) {
            try r.m.head();
            try r.m.download(r.m.logits, std.mem.sliceAsBytes(logits));
            const file = try common.path(gpu, refs, "boundary-p{d}-logits.f32", .{pos});
            defer gpu.gpa.free(file);
            const ref = try common.floats(gpu, file);
            defer gpu.gpa.free(ref);
            const saved = try common.path(gpu, out, "engine-boundary-p{d}-logits.f32", .{pos});
            defer gpu.gpa.free(saved);
            try std.Io.Dir.cwd().writeFile(gpu.io, .{ .sub_path = saved, .data = std.mem.sliceAsBytes(logits) });
            const met = try common.metrics(ref, logits);
            std.debug.print("BOUNDARY position={d} kl={e} top1={}\n", .{ pos, met.kl, met.top1 });
            if (met.kl > 0.01) return error.LlamaLogitDrift;
        }
    }
    if (r.m.step(c.bos, 16384, true)) |_| return error.ContextAccepted else |err| {
        if (err != error.PastContext) return err;
    }
    try r.m.step(c.bos, 0, true);
    try r.m.download(r.m.logits, std.mem.sliceAsBytes(logits));
    const fresh_file = try common.path(gpu, refs, "boundary-fresh-logits.f32", .{});
    defer gpu.gpa.free(fresh_file);
    const fresh = try common.floats(gpu, fresh_file);
    defer gpu.gpa.free(fresh);
    const met = try common.metrics(fresh, logits);
    if (met.kl > 0.01) return error.LlamaLogitDrift;
    try r.m.step(c.bos, 0, true);
    const again = try gpu.gpa.alloc(f32, c.vocab);
    defer gpu.gpa.free(again);
    try r.m.download(r.m.logits, std.mem.sliceAsBytes(again));
    try check.expect(std.mem.eql(f32, logits, again), "fresh request after refusal changed logits", .{});
    std.debug.print("SUMMARY llama boundary heads=56/8 layers=2 context=16384 positions=511,512,2047,2048,16383 refused=16384 fresh_kl={e} fresh_repeat=bit_equal device_peak={d}\n", .{ met.kl, cuda.usage(false).peak });
}
