const shared = @import("../../dflash2/config.zig");
pub const repository = shared.repository;
pub const tap_ids = shared.tap_ids;
pub const Config = shared.Config;
pub fn parse(a: @import("std").mem.Allocator, raw: []const u8) !Config {
    const c = try shared.parse(a, raw);
    try c.requireStock();
    return c;
}
test {
    _ = shared;
}
