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
    try r.initCache(gpu, model, 2, 16384, 52);
    defer r.deinit();
    const c = r.w.config;
    if (c.heads != 56 or c.kv_heads != 8 or c.head_dim != 128 or c.layers != 52) return error.BadReference;
    var offsets: [4]i64 = undefined;
    try r.m.download(r.m.offsets + 50 * 16, std.mem.asBytes(&offsets));
    for (50..52) |slot| {
        const i = (slot - 50) * 2;
        const k_offset: i64 = @intCast(slot * 16384 * c.kvDim());
        const v_offset = @divExact(@as(i64, @intCast(r.m.vc)) - @as(i64, @intCast(r.m.kc)), 2) + k_offset;
        try check.expect(offsets[i] == k_offset and offsets[i + 1] == v_offset, "late KV offsets disagree with cache allocations", .{});
    }
    try std.Io.Dir.cwd().createDirPath(gpu.io, out);
    const state = try gpu.gpa.alloc(f32, c.hidden);
    defer gpu.gpa.free(state);
    const logits = try gpu.gpa.alloc(f32, c.vocab);
    defer gpu.gpa.free(logits);
    for (seqs.value[0], 0..) |token, pos| {
        try r.m.begin(token, pos);
        for (0..2) |li| {
            try r.m.layer(50 + li);
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
                std.debug.print("BOUNDARY position={d} weight_layer={d} cache_slot={d} rmse={e} max_abs={e} relative_l2={e}\n", .{ pos, li, 50 + li, met.rmse, met.max_abs, met.relative_l2 });
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
    try freshRequest(&r);
    try r.m.download(r.m.logits, std.mem.sliceAsBytes(logits));
    const fresh_file = try common.path(gpu, refs, "boundary-fresh-logits.f32", .{});
    defer gpu.gpa.free(fresh_file);
    const fresh = try common.floats(gpu, fresh_file);
    defer gpu.gpa.free(fresh);
    const met = try common.metrics(fresh, logits);
    if (met.kl > 0.01) return error.LlamaLogitDrift;
    try freshRequest(&r);
    const again = try gpu.gpa.alloc(f32, c.vocab);
    defer gpu.gpa.free(again);
    try r.m.download(r.m.logits, std.mem.sliceAsBytes(again));
    try check.expect(std.mem.eql(f32, logits, again), "fresh request after refusal changed logits", .{});
    std.debug.print("SUMMARY llama boundary heads=56/8 weight_layers=2 cache_layers=52 slots=50,51 kv_bytes=3489660928 context=16384 positions=511,512,2047,2048,16383 refused=16384 fresh_kl={e} fresh_repeat=bit_equal device_peak={d}\n", .{ met.kl, cuda.usage(false).peak });
}

fn freshRequest(r: *common.Rig) !void {
    try r.m.begin(r.w.config.bos, 0);
    try r.m.layer(50);
    try r.m.layer(51);
    try r.m.head();
}
