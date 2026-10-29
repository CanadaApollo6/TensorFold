const std = @import("std");

pub fn module(b: *std.Build, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode, cuda: *std.Build.Module, core: *std.Build.Module, lanes: *std.Build.Module) *std.Build.Module {
    return b.createModule(.{ .root_source_file = b.path("zig/llama_cuda.zig"), .target = target, .optimize = optimize, .link_libc = true, .imports = &.{ .{ .name = "cuda", .module = cuda }, .{ .name = "core", .module = core }, .{ .name = "lanes", .module = lanes } } });
}

pub fn host(b: *std.Build, m: *std.Build.Module, all: *std.Build.Step) void {
    const run = b.addRunArtifact(b.addTest(.{ .root_module = m }));
    b.step("test-llama-host", "Llama config and EXL3 metadata without a GPU").dependOn(&run.step);
    all.dependOn(&run.step);
    const metal = b.createModule(.{ .root_source_file = b.path("zig/src/metal/metal.zig"), .target = b.graph.host });
    const dflash = b.createModule(.{ .root_source_file = b.path("zig/dflash2_host.zig"), .target = b.graph.host, .optimize = .debug, .link_libc = true, .imports = &.{ .{ .name = "core", .module = m.import_table.get("core").? }, .{ .name = "metal", .module = metal } } });
    b.step("test-dflash2-host", "Shared DFlash2 and stock Qwen entry point CPU contracts").dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = dflash })).step);
}

pub fn runner(b: *std.Build, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode, cuda: *std.Build.Module, core: *std.Build.Module, lanes: *std.Build.Module, controls: ?std.Build.LazyPath) void {
    const m = b.createModule(.{ .root_source_file = b.path("zig/tests/cuda/llama_main.zig"), .target = target, .optimize = optimize, .link_libc = true, .imports = &.{ .{ .name = "cuda", .module = cuda }, .{ .name = "core", .module = core }, .{ .name = "lanes", .module = lanes }, .{ .name = "llama", .module = module(b, target, optimize, cuda, core, lanes) } } });
    const options = b.addOptions();
    options.addOption(bool, "controls", controls != null);
    m.addOptions("contract_options", options);
    m.addAnonymousImport("fatbin_contract_controls", .{ .root_source_file = controls orelse b.addWriteFiles().add("no-controls.fatbin", "") });
    const exe = b.addExecutable(.{ .name = "tf-llama-test", .root_module = m });
    b.step("tf-llama-test", "Build bounded Llama model and pointwise contract checks").dependOn(&b.addInstallArtifact(exe, .{}).step);
}
