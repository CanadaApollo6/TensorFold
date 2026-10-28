const std = @import("std");

pub fn module(b: *std.Build, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode, cuda: *std.Build.Module, core: *std.Build.Module, lanes: *std.Build.Module) *std.Build.Module {
    return b.createModule(.{ .root_source_file = b.path("zig/src/families/llama/cuda.zig"), .target = target, .optimize = optimize, .link_libc = true, .imports = &.{ .{ .name = "cuda", .module = cuda }, .{ .name = "core", .module = core }, .{ .name = "lanes", .module = lanes } } });
}

pub fn host(b: *std.Build, m: *std.Build.Module, all: *std.Build.Step) void {
    const run = b.addRunArtifact(b.addTest(.{ .root_module = m }));
    b.step("test-llama-host", "Llama config and EXL3 metadata without a GPU").dependOn(&run.step);
    all.dependOn(&run.step);
}
