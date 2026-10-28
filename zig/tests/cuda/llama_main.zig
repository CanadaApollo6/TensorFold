const std = @import("std");
const cuda = @import("cuda");
const check = @import("check.zig");
const llama = @import("llama_tests.zig");

pub fn main(init: std.process.Init) !u8 {
    const args = try init.minimal.args.toSlice(init.arena.allocator());
    if (args.len < 2) return error.MissingCommand;
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
