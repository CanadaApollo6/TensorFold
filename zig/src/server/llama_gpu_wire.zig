const std = @import("std");
const cuda = @import("cuda");
const llama = @import("llama");
const lanes = @import("lanes");
const api = @import("engine_api");
const json = @import("json");
const Forced = @import("llama_gpu_forced.zig").Forced;

const Harness = struct {
    gpa: std.mem.Allocator,
    io: std.Io,
    backend: lanes.backend.Backend,
    facts: lanes.Model,
    drafted: bool = false,
    cut: usize = 0,
    fn engine(h: *Harness) api.Engine {
        return .{ .ctx = h, .vtable = &.{ .info = info, .submit = submit, .cancel = cancel, .status = status, .memory = memory } };
    }
    fn info(_: *anyopaque) api.Info {
        return .{ .context_window = 512 };
    }
    fn submit(p: *anyopaque, id: api.Id, request: *const api.Request, sink: api.Sink) api.SubmitError!void {
        const h: *Harness = @ptrCast(@alignCast(p));
        h.generate(id, request, sink) catch |err| {
            std.debug.print("forced wire error {t}\n", .{err});
            return error.InvalidRequest;
        };
    }
    fn generate(h: *Harness, id: api.Id, req: *const api.Request, sink: api.Sink) !void {
        var facts = h.facts;
        if (!h.drafted) {
            facts.exact_width = 1;
            facts.batch_rows = 1;
            facts.drafts = 0;
            facts.speculate = false;
            facts.mtp = false;
        }
        var cfg = try lanes.Config.init(h.gpa, facts, if (h.drafted) 8 else 1, if (h.drafted) 7 else 0);
        defer cfg.deinit(h.gpa);
        var clock: lanes.fake.FixedClock = .{};
        var runner = lanes.Engine.init(h.gpa, &cfg, h.backend, clock.clock());
        defer runner.deinit();
        var stream = try lanes.Stream.init(h.gpa, .{ .id = "forced", .prompt = req.prompt, .max_new = req.max_tokens, .eos = req.eos, .drafts = h.drafted, .stop_check = if (req.stop) |stop| .{ .ptr = stop.ctx, .check = stop.check } else null });
        defer stream.deinit(h.gpa);
        try runner.addStream(&stream);
        while (runner.activeCount() > 0) try runner.step();
        if (stream.emitted().len != h.cut or (h.drafted and stream.accepted < h.cut - 1)) return error.OffsetNotAccepted;
        sink.event(sink.ctx, id, &.{ .prefilled = 0 });
        const emitted = stream.emitted();
        if (h.drafted) sink.event(sink.ctx, id, &.{ .tokens = emitted }) else for (emitted) |*token| sink.event(sink.ctx, id, &.{ .tokens = @as(*const [1]u32, token) });
        sink.event(sink.ctx, id, &.{ .finished = .{ .reason = if (stream.reason == .stop) .stop else .length } });
    }
    fn cancel(_: *anyopaque, _: api.Id) void {}
    fn status(_: *anyopaque, out: *api.Status, _: []u32) void {
        out.* = .{};
    }
    fn memory(_: *anyopaque, _: bool) ?api.Memory {
        return null;
    }
};

pub fn main(init: std.process.Init) !void {
    const args = try init.minimal.args.toSlice(init.arena.allocator());
    if (args.len != 3) return error.ExpectedModelAndDrafter;
    const a = init.arena.allocator();
    var driver = try cuda.Driver.open();
    defer driver.close();
    var ctx = try cuda.Context.init(&driver, 0);
    defer ctx.deinit();
    const loaded = try llama.native.open(a, init.io, &ctx, args[1], null, .{ .context = 512, .drafts = true, .drafter = args[2] });
    defer loaded.deinit(loaded.ctx);
    var problem: []const u8 = "";
    const text = try @import("hf_text.zig").HfText.load(a, init.io, args[1], null, a, &problem);
    defer text.deinit();
    try checkCases(a, init.io, text.text(), loaded.backend, loaded.facts);
}

pub fn checkCases(a: std.mem.Allocator, io: std.Io, text: anytype, inner: lanes.backend.Backend, facts: lanes.Model) !void {
    const aa = try text.encode(a, "a", false);
    const xx = try text.encode(a, "X", false);
    if (aa.len != 1 or xx.len != 1) return error.ExpectedSingleTokens;
    var ids: [16]u32 = @splat(aa[0]);
    var forced: Forced = .{ .inner = inner, .script = &ids };
    var harness: Harness = .{ .gpa = a, .io = io, .backend = forced.backend(), .facts = facts };
    const srv = try @import("server.zig").Server.init(a, io, harness.engine(), text, .{ .served_name = "nanbeige", .model_type = "llama", .model_ids = &.{"nanbeige"}, .use_drafts = true }, null);
    defer srv.deinit();
    var cases: usize = 0;
    for (0..2) |kind| for (2..9) |cut| for ([_]bool{ false, true }) |chat| {
        if (kind == 0 and chat) continue;
        @memset(&ids, aa[0]);
        ids[cut - 1] = if (kind == 0) xx[0] else srv.eos[0];
        const body = (try json.parse(a, "{\"messages\":[{\"role\":\"user\",\"content\":\"x\"}],\"prompt\":\"x\",\"temperature\":0,\"max_tokens\":16,\"stream_options\":{\"include_usage\":true}}")).ok;
        if (kind == 0) try body.object.put(a, "stop", .{ .string = "X" });
        harness.cut = cut;
        for ([_]bool{ false, true }) |streaming| {
            try body.object.put(a, "stream", .{ .bool = streaming });
            harness.drafted = false;
            const expected = try @import("llama_blocks_test.zig").canonical(a, try @import("llama_bytes_test.zig").wireAt(srv, a, body, 200, chat), chat);
            harness.drafted = true;
            const actual = try @import("llama_blocks_test.zig").canonical(a, try @import("llama_bytes_test.zig").wireAt(srv, a, body, 200, chat), chat);
            if (!std.mem.eql(u8, actual, expected)) return error.ForcedWireMismatch;
            var stop = false;
            var completion: ?i64 = null;
            var frames = std.mem.splitSequence(u8, actual, "\n\n");
            while (frames.next()) |frame| {
                if (frame.len == 0 or std.mem.eql(u8, frame, "data: [DONE]")) continue;
                const value = (try json.parse(a, if (streaming) frame[6..] else frame)).ok;
                if (value.get("choices")) |choices| for (choices.array) |choice| {
                    if (choice.strField("finish_reason")) |reason| stop = stop or std.mem.eql(u8, reason, "stop");
                };
                if (value.get("usage")) |usage| if (usage != .null) {
                    completion = usage.get("completion_tokens").?.int64();
                };
            }
            if (!stop) return error.StopNotObserved;
            if (completion != @as(i64, @intCast(cut))) return error.StopOffsetNotObserved;
            cases += 1;
        }
    };
    std.debug.print("FORCED_WIRE cases={d} eos_offsets=2..8 stop_offsets=2..8 accepted=true equal=true\n", .{cases});
}
