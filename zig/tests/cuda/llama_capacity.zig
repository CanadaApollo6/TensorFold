const std = @import("std");
const llama = @import("llama");
const core = @import("core");
const cuda = @import("cuda");

pub fn run(a: std.mem.Allocator, io: std.Io, dir: []const u8, total_mib: usize, wanted: usize) !void {
    const c = try llama.Config.read(a, io, dir);
    var ck = try core.checkpoint.Checkpoint.openModel(a, io, dir);
    defer ck.close();
    var map = try llama.names.map(a, &ck, c);
    defer map.deinit(a);
    const weights = llama.weights.deviceBytes(map, c.layers);
    const pair = weights + (try llama.memory.runtimeBytesMode(c, c.layers, wanted, .bf16)) + (try llama.memory.runtimeBytesMode(c, c.layers, wanted, .q4)) + 2 * llama.memory.batchBytes(c, 8);
    std.debug.print("NEED mode=pair context={d} bytes={d} need_mib={d}\n", .{ wanted, pair, (pair + (1 << 20) - 1) >> 20 });
    const total = total_mib << 20;
    const reserve = 2 << 30;
    for ([_]cuda.kv4.Mode{ .bf16, .q4 }) |mode| {
        var lo: usize = 0;
        var hi: usize = c.max_context;
        while (lo < hi) {
            const mid = lo + (hi - lo + 1) / 2;
            if (weights + (try llama.memory.runtimeBytesMode(c, c.layers, mid, mode)) + reserve <= total) lo = mid else hi = mid - 1;
        }
        for ([_]usize{ 256, 20128, 60128, 100128, wanted }) |length| {
            const need = weights + try llama.memory.runtimeBytesMode(c, c.layers, length, mode);
            std.debug.print("NEED mode={s} context={d} bytes={d} need_mib={d}\n", .{ @tagName(mode), length, need, (need + (1 << 20) - 1) >> 20 });
        }
        const runtime = try llama.memory.runtimeBytesMode(c, c.layers, lo, mode);
        const next = try llama.memory.runtimeBytesMode(c, c.layers, lo + 1, mode);
        std.debug.print("CAPACITY mode={s} total_mib={d} reserve_mib=2048 weights={d} token_bytes={d} max_context={d} runtime={d} next_runtime={d} allocated_plus_reserve={d}\n", .{ @tagName(mode), total_mib, weights, try mode.bytes(c.layers, 1, c.kv_heads, c.head_dim), lo, runtime, next, weights + runtime + reserve });
    }
}
