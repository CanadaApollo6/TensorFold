const std = @import("std");
const llama = @import("llama");
const core = @import("core");
const cuda = @import("cuda");
pub fn run(a: std.mem.Allocator, io: std.Io, model: []const u8, dir: []const u8, total_mib: usize, allowance_mib: usize, wanted: usize) !void {
    const c = try llama.Config.read(a, io, model);
    var target = try core.Checkpoint.openModel(a, io, model);
    defer target.close();
    var map = try llama.names.map(a, &target, c);
    defer map.deinit(a);
    const target_bytes = llama.weights.deviceBytes(map, c.layers);
    const path = try std.fs.path.join(a, &.{ dir, "config.json" });
    defer a.free(path);
    const text = try std.Io.Dir.cwd().readFileAlloc(io, path, a, .limited(1 << 20));
    defer a.free(text);
    const cfg = try llama.draft.config.parse(a, text);
    var checkpoint = try core.Checkpoint.openModel(a, io, dir);
    defer checkpoint.close();
    const names = try llama.draft.weights.names(a, cfg);
    defer llama.draft.weights.freeNames(a, names);
    var graph = try llama.draft.weights.load(a, .{ .names = names, .source = llama.draft.checkpoint.Source.from(&checkpoint) }, cfg, .{ .dense = map.embed });
    defer graph.deinit();
    const budget = total_mib << 20;
    const reserve = (2048 + allowance_mib) << 20;
    for ([_]?llama.draft.storage.Format{ null, .bf16, .affine4 }) |format| {
        var footprint: llama.draft.storage.Footprint = .{};
        var extra: usize = 0;
        if (format) |f| {
            for (names) |name| footprint.add(name, try checkpoint.get(name), f);
            const pool = try llama.draft.plan.scratch(a, &graph, f);
            const fixed = llama.draft.plan.fixed(cfg, footprint, pool);
            extra = @max(fixed, footprint.load_peak + llama.draft.plan.split_bytes) + llama.memory.batchBytes(c, 8);
            std.debug.print("DRAFT format={s} native={d} device_weights={d} host_codebooks={d} pool={d} live_fixed={d} total_extra={d} stage={d}\n", .{ @tagName(f), footprint.native, footprint.device, footprint.host_codebooks, pool, fixed, extra, footprint.stage });
        }
        for ([_]cuda.kv4.Mode{ .bf16, .q4 }) |mode| {
            if (wanted > 0) {
                const need = target_bytes + try llama.memory.runtimeBytesMode(c, c.layers, wanted, mode);
                std.debug.print("NEED draft={s} kv={s} context={d} bytes={d} need_mib={d}\n", .{ if (format) |f| @tagName(f) else "none", @tagName(mode), wanted, need + extra, ((need + extra) + (1 << 20) - 1) >> 20 });
            }
            var lo: usize = 0;
            var hi = c.max_context;
            while (lo < hi) {
                const mid = lo + (hi - lo + 1) / 2;
                const need = target_bytes + try llama.memory.runtimeBytesMode(c, c.layers, mid, mode);
                if (need + extra + reserve <= budget) lo = mid else hi = mid - 1;
            }
            std.debug.print("CAPACITY draft={s} kv={s} total_mib={d} allowance_mib={d} weights={d} extra={d} max_context={d}\n", .{ if (format) |f| @tagName(f) else "none", @tagName(mode), total_mib, allowance_mib, target_bytes, extra, lo });
        }
    }
}
