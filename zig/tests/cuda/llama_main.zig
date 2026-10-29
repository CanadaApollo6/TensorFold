const std = @import("std");
const cuda = @import("cuda");
const check = @import("check.zig");
const llama = @import("llama_tests.zig");

pub fn main(init: std.process.Init) !u8 {
    const args = try init.minimal.args.toSlice(init.arena.allocator());
    if (args.len < 2) return error.MissingCommand;
    if (std.mem.eql(u8, args[1], "llama-draft-capacity")) {
        try @import("llama_draft_capacity.zig").run(init.gpa, init.io, try arg(args, 2), try arg(args, 3), try std.fmt.parseInt(usize, try arg(args, 4), 10), try std.fmt.parseInt(usize, try arg(args, 5), 10), if (args.len > 6) try std.fmt.parseInt(usize, args[6], 10) else 0);
        return 0;
    }
    if (std.mem.eql(u8, args[1], "llama-capacity")) {
        try @import("llama_capacity.zig").run(init.gpa, init.io, try arg(args[2..], 0), try std.fmt.parseInt(usize, try arg(args[2..], 1), 10), if (args.len > 4) try std.fmt.parseInt(usize, args[4], 10) else 256);
        return 0;
    }
    var driver = try cuda.Driver.open();
    defer driver.close();
    var ctx = try cuda.Context.init(&driver, 0);
    defer ctx.deinit();
    const gpu: check.Gpu = .{ .d = &driver, .ctx = &ctx, .gpa = init.gpa, .io = init.io };
    run(gpu, args[1], args[2..]) catch |err| {
        std.debug.print("FAIL {s}: {t}\n", .{ args[1], err });
        return 1;
    };
    return 0;
}

fn arg(args: []const [:0]const u8, i: usize) ![]const u8 {
    if (i >= args.len) return error.MissingArgument;
    return args[i];
}

fn run(gpu: check.Gpu, command: []const u8, args: []const [:0]const u8) !void {
    if (std.mem.eql(u8, command, "llama-kv-sweep")) return @import("llama_kv_gate.zig").runLengths(gpu, try arg(args, 0), try arg(args, 1), "pair", try arg(args, 2), 512, try std.fmt.parseInt(usize, try arg(args, 3), 10), true);
    if (std.mem.eql(u8, command, "llama-kv-pair")) return @import("llama_kv_gate.zig").run(gpu, try arg(args, 0), try arg(args, 1), "pair", try arg(args, 2), 512, try std.fmt.parseInt(usize, try arg(args, 3), 10));
    if (std.mem.eql(u8, command, "llama-draft-full")) return @import("llama_draft_full.zig").run(gpu, try arg(args, 0), try arg(args, 1), try arg(args, 2), try arg(args, 3), try std.fmt.parseInt(usize, try arg(args, 4), 10), try std.fmt.parseInt(usize, try arg(args, 5), 10), try std.fmt.parseInt(usize, try arg(args, 6), 10));
    if (std.mem.eql(u8, command, "llama-draft-partial")) return @import("llama_draft_partial.zig").run(gpu, try arg(args, 0), try arg(args, 1), try arg(args, 2), try arg(args, 3), try arg(args, 4));
    if (std.mem.eql(u8, command, "llama-kv4-contract")) return @import("contract_kv4.zig").run(gpu);
    if (std.mem.eql(u8, command, "llama-kv-dump")) return @import("llama_kv_gate.zig").dump(gpu, try arg(args, 0), try arg(args, 1), try arg(args, 2), try arg(args, 3), if (args.len > 4) try std.fmt.parseInt(usize, args[4], 10) else 512);
    if (std.mem.eql(u8, command, "llama-kv4-synthetic")) return @import("llama_kv4.zig").synthetic(gpu);
    if (std.mem.eql(u8, command, "llama-kv4-real")) return @import("llama_kv4.zig").real(gpu, try arg(args, 0), try arg(args, 1), try std.fmt.parseInt(usize, try arg(args, 2), 10));
    if (std.mem.eql(u8, command, "llama-sampling")) return @import("llama_sampling.zig").distribution(gpu, try arg(args, 0));
    if (std.mem.eql(u8, command, "llama-sampled")) return @import("llama_sampling.zig").decode(gpu, try arg(args, 0), try arg(args, 1), try arg(args, 2));
    if (std.mem.eql(u8, command, "llama-layer-control")) return @import("llama_layer_controls.zig").run(gpu, try arg(args, 0), try arg(args, 1), try arg(args, 2), try arg(args, 3));
    if (std.mem.eql(u8, command, "llama-head")) return llama.head(gpu, try arg(args, 0), try arg(args, 1), try arg(args, 2));
    if (std.mem.eql(u8, command, "llama-boundary")) return @import("llama_boundary.zig").run(gpu, try arg(args, 0), try arg(args, 1), try arg(args, 2));
    if (std.mem.eql(u8, command, "llama-pointwise-contract")) return @import("contract_dense_pointwise.zig").run(gpu);
    if (std.mem.eql(u8, command, "llama-layers")) return llama.layers(gpu, try arg(args, 0), try arg(args, 1), try arg(args, 2), try arg(args, 3));
    if (std.mem.eql(u8, command, "llama-logits")) return llama.layers(gpu, try arg(args, 0), try arg(args, 1), null, try arg(args, 2));
    if (std.mem.eql(u8, command, "llama-serve-greedy")) return @import("llama_serve_oracle.zig").run(gpu, try arg(args, 0), try arg(args, 1), try arg(args, 2), try arg(args, 3));
    if (std.mem.eql(u8, command, "llama-greedy")) return llama.greedy(gpu, try arg(args, 0), try arg(args, 1), try arg(args, 2), try arg(args, 3));
    if (std.mem.eql(u8, command, "llama-greedy-slice")) return llama.greedyLayers(gpu, try arg(args, 0), try arg(args, 1), try arg(args, 3), try arg(args, 4), try std.fmt.parseInt(usize, try arg(args, 2), 10));
    if (std.mem.eql(u8, command, "llama-bench-once")) return llama.bench(gpu, try arg(args, 0), try arg(args, 1), true);
    if (std.mem.eql(u8, command, "llama-bench")) return llama.bench(gpu, try arg(args, 0), try arg(args, 1), false);
    return error.UnknownCommand;
}
