const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const weights = @import("weights.zig");
const Config = @import("config.zig").Config;

pub const Format = enum {
    bf16,
    affine4,
    pub fn parse(raw: []const u8) !Format {
        if (std.mem.eql(u8, raw, "bf16") or std.mem.eql(u8, raw, "16")) return .bf16;
        if (std.mem.eql(u8, raw, "affine4") or std.mem.eql(u8, raw, "4")) return .affine4;
        return error.BadDraftFormat;
    }
};
pub const Linear = union(enum) { bf16: u64, affine4: cuda.qlinear.Affine4 };
pub const Entry = struct { linear: ?Linear = null, raw: u64 = 0, norm: u64 = 0 };
pub const Footprint = struct {
    native: usize = 0,
    device: usize = 0,
    host_codebooks: usize = 0,
    stage: usize = 0,
    load_peak: usize = 0,
    linears: usize = 0,
    pub fn add(f: *Footprint, name: []const u8, t: weights.Tensor, format: Format) void {
        f.native += t.bytes.len;
        if (codebook(name)) {
            f.host_codebooks += t.bytes.len;
        } else if (t.rank == 2) {
            f.linears += 1;
            if (format == .bf16) {
                f.device += t.bytes.len;
            } else {
                const l = cuda.qlinear.Affine4.layout(t.shape[0], t.shape[1]);
                f.device += l.words + 2 * l.scales;
                const stage = t.bytes.len / 4 + t.bytes.len / 32;
                f.stage = @max(f.stage, stage);
                f.load_peak = @max(f.load_peak, f.device + stage);
            }
        } else {
            f.device += t.bytes.len;
            if (t.rank == 1) f.device += t.shape[0] * 4;
        }
        f.load_peak = @max(f.load_peak, f.device);
    }
};
pub fn codebook(name: []const u8) bool {
    return std.mem.eql(u8, name, "candidate_selector.predecessor_codebook") or
        std.mem.eql(u8, name, "candidate_selector.successor_codebook");
}
pub const Store = struct {
    a: std.mem.Allocator,
    d: *const cuda.Driver,
    format: Format,
    entries: std.AutoHashMapUnmanaged(usize, Entry) = .empty,
    buffers: std.ArrayList(cuda.DeviceBuffer) = .empty,
    footprint: Footprint = .{},

    pub fn deinit(w: *Store) void {
        for (w.buffers.items) |*b| b.free();
        w.buffers.deinit(w.a);
        w.entries.deinit(w.a);
    }
    fn upload(w: *Store, bytes: []const u8) !u64 {
        var b = try cuda.DeviceBuffer.alloc(w.d, bytes.len);
        errdefer b.free();
        try b.upload(0, bytes);
        try w.buffers.append(w.a, b);
        return b.ptr;
    }
    pub fn get(w: *const Store, t: weights.Tensor) !Entry {
        return w.entries.get(@intFromPtr(t.bytes.ptr)) orelse error.MissingDraftWeight;
    }
    pub fn load(a: std.mem.Allocator, d: *const cuda.Driver, s: cuda.Stream, ck: *core.Checkpoint, c: Config, format: Format, affine: *const cuda.qlinear.Affine4Kernels) !Store {
        var w: Store = .{ .a = a, .d = d, .format = format };
        errdefer w.deinit();
        const names = try weights.names(a, c);
        defer weights.freeNames(a, names);
        for (names) |name| {
            const t = try ck.get(name);
            w.footprint.add(name, t, format);
            if (codebook(name)) continue;
            var e: Entry = .{};
            if (t.rank == 2 and format == .affine4) {
                var q = try core.affine4.prepare(a, @alignCast(std.mem.bytesAsSlice(u16, t.bytes)), t.shape[0], t.shape[1]);
                defer q.deinit();
                const raw = [_][]const u8{ std.mem.sliceAsBytes(q.words), std.mem.sliceAsBytes(q.scales), std.mem.sliceAsBytes(q.biases) };
                var staging: [3]cuda.DeviceBuffer = undefined;
                var made: usize = 0;
                defer for (staging[0..made]) |*b| b.free();
                for (&staging, raw) |*b, bytes| {
                    b.* = try cuda.DeviceBuffer.alloc(d, bytes.len);
                    made += 1;
                    try b.upload(0, bytes);
                }
                const layout = cuda.qlinear.Affine4.layout(q.n, q.k);
                const sizes = [_]usize{ layout.words, layout.scales, layout.scales };
                var pointers: [3]u64 = undefined;
                for (&pointers, sizes) |*ptr, bytes| {
                    var b = try cuda.DeviceBuffer.alloc(d, bytes);
                    errdefer b.free();
                    try w.buffers.append(a, b);
                    ptr.* = b.ptr;
                }
                const out: cuda.qlinear.Affine4 = .{ .w = pointers[0], .s = pointers[1], .b = pointers[2], .n = q.n, .k = q.k, .npad = layout.npad };
                try affine.pack(s, staging[0].ptr, staging[1].ptr, staging[2].ptr, out);
                try s.synchronize();
                e.linear = .{ .affine4 = out };
            } else {
                e.raw = try w.upload(t.bytes);
                if (t.rank == 2) e.linear = .{ .bf16 = e.raw };
                if (t.rank == 1) {
                    const values = try a.alloc(f32, t.shape[0]);
                    defer a.free(values);
                    for (values, 0..) |*v, i| v.* = core.affine4.asFloat(std.mem.readInt(u16, t.bytes[i * 2 ..][0..2], .little));
                    e.norm = try w.upload(std.mem.sliceAsBytes(values));
                }
            }
            try w.entries.put(a, @intFromPtr(t.bytes.ptr), e);
        }
        return w;
    }
};

test "draft storage is explicit and codebooks do not occupy device memory" {
    try std.testing.expectEqual(Format.affine4, try Format.parse("4"));
    try std.testing.expectError(error.BadDraftFormat, Format.parse("8"));
    const t: weights.Tensor = .{ .rank = 2, .dtype = .bf16, .shape = .{ 128, 64, 1, 1, 1 }, .bytes = &(@as([16384]u8, @splat(0))) };
    var f: Footprint = .{};
    f.add("candidate_selector.predecessor_codebook", t, .affine4);
    try std.testing.expectEqual(@as(usize, 0), f.device);
    try std.testing.expectEqual(t.bytes.len, f.host_codebooks);
    f.add("fc.weight", t, .affine4);
    try std.testing.expectEqual(@as(usize, 4608), f.device);
}
