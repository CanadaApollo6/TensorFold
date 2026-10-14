//! tf-lfm2-cuda: LFM2-MoE's Zig CUDA engine, greedy runs timed and teacher-forced logits for the reference.

const std = @import("std");
const cuda = @import("cuda");
const lfm2 = @import("lfm2");

const usage =
    \\usage: tf-lfm2-cuda run MODEL --prompts prompts.json [--max-tokens N] [--context N] [--expect results.json] [--save out.json]
    \\  Greedy: each prompt's prefill time, decode tokens a second, tokens and their sha (against --expect's);
    \\  --draft K checks up to K prompt-lookup drafts a step.
    \\       tf-lfm2-cuda logits MODEL --prompts ids.json --out logits.bin [--prefill P] [--window N]
    \\  Teacher-forced over one id list: ids[0..P) as prompt chunks, the rest in windows of N rows; each position's
    \\  logits as bf16 rows (positions before P - 1 zero), for 01-inference/lfm25-24b/compare_logits.py.
    \\
;

const Options = struct {
    model: []const u8,
    prompts: []const u8 = "",
    expect: ?[]const u8 = null,
    save: ?[]const u8 = null,
    out: ?[]const u8 = null,
    dump: ?[]const u8 = null, // logits: the first window's norms, row 0 (bf16 [norms, hidden]), as ref.py --dump's
    max_tokens: usize = 128,
    context: usize = 8192,
    prefill: usize = 0,
    window: usize = 1,
    draft: usize = 0, // run: prompt-lookup drafts a step (0: plain greedy steps)
    device: c_int = 0,
};

pub fn main(init: std.process.Init) !u8 {
    const gpa = init.gpa;
    const arena = init.arena.allocator();
    const args = try init.minimal.args.toSlice(arena);
    const logits = args.len >= 2 and std.mem.eql(u8, args[1], "logits");
    if (args.len < 3 or !(std.mem.eql(u8, args[1], "run") or logits)) {
        std.debug.print("{s}", .{usage});
        return 2;
    }
    var o: Options = .{ .model = args[2] };
    var i: usize = 3;
    while (i + 1 < args.len) : (i += 2) {
        const k = args[i];
        const v = args[i + 1];
        const n = std.fmt.parseInt(usize, v, 10) catch 0;
        if (std.mem.eql(u8, k, "--prompts")) o.prompts = v else if (std.mem.eql(u8, k, "--expect")) o.expect = v else if (std.mem.eql(u8, k, "--save")) o.save = v else if (std.mem.eql(u8, k, "--out")) o.out = v else if (std.mem.eql(u8, k, "--dump")) o.dump = v else if (std.mem.eql(u8, k, "--max-tokens")) o.max_tokens = n else if (std.mem.eql(u8, k, "--context")) o.context = n else if (std.mem.eql(u8, k, "--prefill")) o.prefill = n else if (std.mem.eql(u8, k, "--window")) o.window = n else if (std.mem.eql(u8, k, "--draft")) o.draft = n else if (std.mem.eql(u8, k, "--device")) o.device = @intCast(n) else {
            std.debug.print("unknown option {s}\n{s}", .{ k, usage });
            return 2;
        }
    }
    if (o.prompts.len == 0 or (logits and o.out == null)) {
        std.debug.print("{s}", .{usage});
        return 2;
    }
    const prompts = try readJson(arena, init.io, o.prompts);
    const expect = if (o.expect) |p| try readJson(arena, init.io, p) else null;
    var driver = try cuda.Driver.open();
    defer driver.close();
    var ctx = try cuda.Context.init(&driver, o.device);
    defer ctx.deinit();
    const clock = std.Io.Clock.awake;
    var t0 = clock.now(init.io).toNanoseconds();
    const rows = if (std.c.getenv("TF_LFM2_PROMPT_ROWS")) |v| std.fmt.parseInt(usize, std.mem.span(v), 10) catch return error.BadPromptRows else lfm2.engine.prompt_rows;
    const e = try lfm2.Engine.init(gpa, init.io, &ctx, o.model, .{ .context = o.context, .prompt_rows = rows });
    defer e.deinit();
    std.debug.print("loaded in {d:.1}s, weights {d:.2} GiB\n", .{ seconds(clock.now(init.io).toNanoseconds() - t0), @as(f64, @floatFromInt(e.w.bytes())) / (1 << 30) });
    var seq = try e.sequence(o.context);
    defer seq.deinit();
    if (logits) return dumpLogits(gpa, init.io, e, &seq, try tokenList(arena, prompts), o);
    if (std.c.getenv("TF_LFM2_BENCH_WINDOWS") != null) { // after each prompt's prefill: ms a verify window of 1 ..= 16 rows
        var it = prompts.object.iterator();
        while (it.next()) |entry| {
            try e.prefill(&seq, try tokenList(arena, entry.value_ptr.*));
            var ms: [16]f64 = undefined;
            try e.benchWindows(&seq, init.io, 30, &ms);
            std.debug.print("{s} at {d}:", .{ entry.key_ptr.*, seq.pos });
            for (ms) |m| std.debug.print(" {d:.2}", .{m});
            std.debug.print("\n", .{});
        }
        return 0;
    }
    var mismatches: usize = 0;
    var saved: std.ArrayList(u8) = .empty;
    try saved.appendSlice(arena, "{\"results\": {");
    var it = prompts.object.iterator();
    while (it.next()) |entry| {
        const name = entry.key_ptr.*;
        const ids = try tokenList(arena, entry.value_ptr.*);
        var out: std.ArrayList(u32) = .empty;
        defer out.deinit(gpa);
        t0 = clock.now(init.io).toNanoseconds();
        try e.prefill(&seq, ids);
        const first = try e.first(&seq);
        const t1 = clock.now(init.io).toNanoseconds();
        var ds: lfm2.Engine.DraftStats = .{};
        if (o.draft > 0) try e.decodeDraft(&seq, ids, first, @min(o.max_tokens, seq.capacity - ids.len), o.draft, &out, &ds) else try e.decode(&seq, first, @min(o.max_tokens, seq.capacity - ids.len), &out);
        const t2 = clock.now(init.io).toNanoseconds();
        const sha = try sha12(arena, out.items);
        var verdict: []const u8 = "";
        if (expect) |x| blk: {
            const want_v = (x.object.get("results") orelse break :blk).object.get(name) orelse break :blk;
            const want = try tokenList(arena, want_v.object.get("tokens").?);
            const same = std.mem.eql(u32, want, out.items);
            if (!same) mismatches += 1;
            var at: usize = 0;
            while (at < @min(want.len, out.items.len) and want[at] == out.items[at]) at += 1;
            verdict = if (same) " MATCH" else try std.fmt.allocPrint(arena, " DIFFER at token {d}", .{at});
        }
        const steps = out.items.len - 1;
        std.debug.print("{s}: prompt {d} in {d:.3}s ({d:.0} tok/s), {d} tokens sha {s}, decode {d:.1} tok/s{s}\n", .{ name, ids.len, seconds(t1 - t0), @as(f64, @floatFromInt(ids.len)) / seconds(t1 - t0), out.items.len, sha, @as(f64, @floatFromInt(steps)) / seconds(t2 - t1), verdict });
        if (ds.steps > 0) std.debug.print("  drafts: {d} steps, {d:.2} tokens a step, {d} drafted, {d} accepted\n", .{ ds.steps, @as(f64, @floatFromInt(steps)) / @as(f64, @floatFromInt(ds.steps)), ds.drafted, ds.accepted });
        if (saved.items.len > 13) try saved.appendSlice(arena, ", ");
        try saved.print(arena, "\"{s}\": {{\"tokens\": [", .{name});
        for (out.items, 0..) |t, j| try saved.print(arena, "{s}{d}", .{ if (j > 0) ", " else "", t });
        try saved.appendSlice(arena, "]}");
    }
    try saved.appendSlice(arena, "}}\n");
    if (o.save) |path| try std.Io.Dir.cwd().writeFile(init.io, .{ .sub_path = path, .data = saved.items });
    return if (mismatches == 0) 0 else 1;
}

fn seconds(ns: i96) f64 {
    return @as(f64, @floatFromInt(ns)) / 1e9;
}

fn dumpLogits(gpa: std.mem.Allocator, io: std.Io, e: *lfm2.Engine, seq: *lfm2.state.Seq, ids: []const u32, o: Options) !u8 {
    const vocab = e.w.g.vocab;
    const T = ids.len;
    if (o.window < 1 or o.window > lfm2.state.max_rows or o.prefill >= T) return error.BadWindow;
    const all = try gpa.alloc(u16, T * vocab);
    defer gpa.free(all);
    @memset(all, 0);
    var f = e.forward();
    var at: usize = 0;
    try seq.reset(e.ops(), e.w.g);
    if (o.prefill > 0) {
        try e.prefill(seq, ids[0..o.prefill]);
        try f.headLast(seq);
        try e.ops().b().download(std.mem.sliceAsBytes(all[(o.prefill - 1) * vocab ..][0..vocab]), e.scratch.logits);
        at = o.prefill;
    }
    var rows: std.ArrayList(u16) = .empty;
    defer rows.deinit(gpa);
    while (at < T) {
        const n = @min(o.window, T - at);
        f.dump = if (o.dump != null and at == 0) &rows else null;
        try f.step(ids[at..][0..n], seq, true);
        try e.ops().b().download(std.mem.sliceAsBytes(all[at * vocab ..][0 .. n * vocab]), e.scratch.logits);
        try e.s.synchronize();
        at += n;
    }
    try e.s.synchronize();
    if (o.dump) |p| try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = p, .data = std.mem.sliceAsBytes(rows.items) });
    try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = o.out.?, .data = std.mem.sliceAsBytes(all) });
    std.debug.print("{d} positions (prefill {d}, windows of {d}) -> {s}\n", .{ T, o.prefill, o.window, o.out.? });
    return 0;
}

fn readJson(a: std.mem.Allocator, io: std.Io, path: []const u8) !std.json.Value {
    const bytes = try std.Io.Dir.cwd().readFileAlloc(io, path, a, .limited(1 << 30));
    return (try std.json.parseFromSliceLeaky(std.json.Value, a, bytes, .{}));
}

fn tokenList(a: std.mem.Allocator, v: std.json.Value) ![]u32 {
    if (v != .array) return error.BadPromptFile;
    const out = try a.alloc(u32, v.array.items.len);
    for (v.array.items, out) |x, *t| t.* = @intCast(x.integer);
    return out;
}

/// sha256 of json.dumps(tokens), "[1, 2, 3]", its first 12 hex digits.
fn sha12(a: std.mem.Allocator, tokens: []const u32) ![]const u8 {
    var text: std.ArrayList(u8) = .empty;
    try text.append(a, '[');
    for (tokens, 0..) |t, i| {
        if (i > 0) try text.appendSlice(a, ", ");
        try text.print(a, "{d}", .{t});
    }
    try text.append(a, ']');
    var digest: [32]u8 = undefined;
    std.crypto.hash.sha2.Sha256.hash(text.items, &digest, .{});
    const hex = std.fmt.bytesToHex(digest, .lower);
    return a.dupe(u8, hex[0..12]);
}
