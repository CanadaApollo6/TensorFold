const std = @import("std");
const cuda = @import("cuda");
const llama = @import("llama");
const check = @import("check.zig");

fn write(gpu: check.Gpu, dir: []const u8, name: []const u8, data: []const u8) !void {
    const file = try std.fs.path.join(gpu.gpa, &.{ dir, name });
    defer gpu.gpa.free(file);
    try std.Io.Dir.cwd().writeFile(gpu.io, .{ .sub_path = file, .data = data });
}

pub fn run(gpu: check.Gpu, dir: []const u8, prompts: []const u8, out: []const u8, draft_dir: []const u8, kind: []const u8) !void {
    const raw = try std.Io.Dir.cwd().readFileAlloc(gpu.io, prompts, gpu.gpa, .limited(16 << 20));
    defer gpu.gpa.free(raw);
    const seqs = try std.json.parseFromSlice([][]u32, gpu.gpa, raw, .{});
    defer seqs.deinit();
    if (seqs.value.len != 1 or seqs.value[0].len == 0) return error.BadReference;
    const seq = seqs.value[0];
    const mode = try llama.native.kvMode();
    var w = try llama.weights.loadMode(gpu.gpa, gpu.io, gpu.ctx, dir, null, seq.len + @min(1000, seq.len / 4) + 1, mode);
    defer w.deinit();
    var k = try llama.kernels.Kernels.load(gpu.ctx);
    defer k.deinit();
    var s = try cuda.Stream.init(gpu.d, false);
    defer s.deinit();
    var m = try llama.forward.Model.initRows(gpu.gpa, gpu.d, &w, &k, s, seq.len + @min(1000, seq.len / 4) + 1, mode, 8);
    defer m.deinit();
    var attachment: ?llama.draft.Attachment = null;
    defer if (attachment) |*d| d.deinit();
    if (!std.mem.eql(u8, draft_dir, "none")) {
        const format = try llama.draft.storage.Format.parse(if (std.c.getenv("TF_LLAMA_DRAFT_FORMAT")) |v| std.mem.span(v) else "affine4");
        attachment = try llama.draft.Attachment.init(gpu.gpa, gpu.io, &m, draft_dir, format, null);
    }
    try std.Io.Dir.cwd().createDirPath(gpu.io, out);
    const serial = std.mem.eql(u8, kind, "serial");
    const timing = std.mem.eql(u8, kind, "timing");
    var elapsed: f64 = 0;
    for (0..if (timing) @as(usize, 4) else 1) |rep| {
        if (attachment) |*d| {
            d.r.reset();
            d.retained = .{};
            const start = seq.len -| (d.r.graph.config.window - 1);
            d.r.context = .{ .cache = 1, .begin = start, .end = start };
        }
        try s.synchronize();
        const t0 = std.Io.Timestamp.now(gpu.io, .awake);
        if (!serial and !timing and !std.mem.eql(u8, kind, "batch")) {
            try @import("llama_prompt_resume.zig").run(gpu.gpa, &m, if (attachment) |*d| d else null, seq, kind);
        } else {
            var pos: usize = 0;
            while (pos < seq.len) {
                const n = @min(if (serial) 1 else m.prompt_rows, seq.len - pos);
                m.tap_buffer = 0;
                if (attachment) |*d| d.capture(0);
                if (serial) try m.step(seq[pos], pos, pos + n == seq.len) else try m.prefill(seq[pos..][0..n], pos, pos + n == seq.len);
                if (attachment) |*d| for (pos..pos + n) |position| {
                    if (position >= d.r.context.end) try d.ingest(position - pos, position);
                };
                pos += n;
            }
        }
        try s.synchronize();
        const seconds = @as(f64, @floatFromInt(t0.durationTo(std.Io.Timestamp.now(gpu.io, .awake)).nanoseconds)) / 1e9;
        if (!timing or rep > 0) elapsed += seconds;
    }
    if (timing) elapsed /= 3;
    const logits = try gpu.gpa.alloc(f32, w.config.vocab);
    defer gpu.gpa.free(logits);
    try m.download(m.logits, std.mem.sliceAsBytes(logits));
    try write(gpu, out, "logits.f32", std.mem.sliceAsBytes(logits));
    if (!timing) {
        const stride = m.context * w.config.kv_heads * mode.rowBytes(w.config.head_dim);
        const data = try gpu.gpa.alloc(u8, seq.len * w.config.kv_heads * mode.rowBytes(w.config.head_dim));
        defer gpu.gpa.free(data);
        for (0..w.layers.len) |layer| for ([_]u64{ m.kc, m.vc }, 0..) |base, plane| {
            try m.download(base + layer * stride, data);
            const name = try std.fmt.allocPrint(gpu.gpa, "cache-{d}-{d}.bin", .{ layer, plane });
            defer gpu.gpa.free(name);
            try write(gpu, out, name, data);
        };
        if (m.first != 0) {
            const bos = try gpu.gpa.alloc(u8, w.layers.len * w.config.kvDim() * 4);
            defer gpu.gpa.free(bos);
            try m.download(m.first, bos);
            try write(gpu, out, "bos.bf16", bos);
        }
        if (attachment) |*d| {
            const width = d.r.graph.config.tapWidth() * 2;
            const begin = seq.len -| (d.r.graph.config.window - 1);
            const taps = try gpu.gpa.alloc(u8, (seq.len - begin) * width);
            defer gpu.gpa.free(taps);
            for (begin..seq.len) |position| try m.download(d.history.ptr + (position % d.r.graph.config.window) * width, taps[(position - begin) * width ..][0..width]);
            try write(gpu, out, "taps.bf16", taps);
            const c = d.r.graph.config;
            const ctx = d.r.context;
            if (ctx.begin != begin or ctx.end != seq.len) return error.BadDraftContext;
            const row_bytes = c.kvWidth() * 2;
            const plane_bytes = @as(usize, c.window) * row_bytes;
            const rows = try gpu.gpa.alloc(u8, @as(usize, c.layers) * 2 * (seq.len - begin) * row_bytes);
            defer gpu.gpa.free(rows);
            var at: usize = 0;
            for (0..c.layers * 2) |plane| for (begin..seq.len) |position| {
                try m.download(d.r.ring.ptr + plane * plane_bytes + (position % c.window) * row_bytes, rows[at..][0..row_bytes]);
                at += row_bytes;
            };
            try write(gpu, out, "draft-kv.bf16", rows);
            var metadata: [256]u8 = undefined;
            const context = try std.fmt.bufPrint(&metadata, "{{\"begin\":{d},\"end\":{d},\"layers\":{d},\"width\":{d}}}\n", .{ begin, seq.len, c.layers, c.kvWidth() });
            try write(gpu, out, "draft-context.json", context);
        }
    }
    var buf: [512]u8 = undefined;
    const report = try std.fmt.bufPrint(&buf, "{{\"state\":\"PASS\",\"tokens\":{d},\"seconds\":{d},\"tok_s\":{d},\"chunk\":{d}}}\n", .{ seq.len, elapsed, @as(f64, @floatFromInt(seq.len)) / elapsed, m.prompt_rows });
    try write(gpu, out, "summary.json", report);
}
