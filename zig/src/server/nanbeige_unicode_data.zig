pub const Family = struct {
    prefix: []const u8,
    insensitive: bool,
    ranges: []const [2]u21,
};

pub const families = [_]Family{
    .{
        .prefix = "CJK UNIFIED IDEOGRAPH-",
        .insensitive = false,
        .ranges = &.{
            .{ 0x3400, 0x4dbf },
            .{ 0x4e00, 0x9fff },
            .{ 0x20000, 0x2a6df },
            .{ 0x2a700, 0x2b739 },
            .{ 0x2b740, 0x2b81d },
            .{ 0x2b820, 0x2cea1 },
            .{ 0x2ceb0, 0x2ebe0 },
            .{ 0x30000, 0x3134a },
            .{ 0x31350, 0x323af },
        },
    },
    .{
        .prefix = "KHITAN SMALL SCRIPT CHARACTER-",
        .insensitive = true,
        .ranges = &.{
            .{ 0x18b00, 0x18cd5 },
        },
    },
    .{
        .prefix = "NUSHU CHARACTER-",
        .insensitive = true,
        .ranges = &.{
            .{ 0x1b170, 0x1b2fb },
        },
    },
};
