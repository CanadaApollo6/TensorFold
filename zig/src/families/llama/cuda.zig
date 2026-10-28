pub const Config = @import("config.zig").Config;
pub const config = @import("config.zig");
pub const names = @import("names.zig");
pub const weights = @import("weights.zig");
pub const kernels = @import("kernels.zig");
pub const forward = @import("forward.zig");
pub const backend = @import("backend.zig");
pub const native = @import("native.zig");
pub const memory = @import("memory.zig");
pub const validation = @import("validation.zig");

test {
    _ = config;
    _ = names;
    _ = weights;
    _ = kernels;
    _ = forward;
    _ = backend;
    _ = native;
    _ = memory;
    _ = validation;
}
