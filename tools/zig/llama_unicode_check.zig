const std = @import("std");
const Names = @import("names").Names;

test "every Python 3.12 name and alias, both cases, and negative names" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const path = std.mem.span(std.c.getenv("UNICODE_NAMES_INPUT") orelse return error.MissingCorpus);
    const corpus = try std.Io.Dir.cwd().readFileAlloc(std.testing.io, path, a, .limited(64 * 1024 * 1024));
    var lines = std.mem.splitScalar(u8, corpus, '\n');
    var names: Names = .{};
    var count: usize = 0;
    while (lines.next()) |line| {
        if (line.len == 0) continue;
        const tab = std.mem.indexOfScalar(u8, line, '\t').?;
        const expected = try std.fmt.parseInt(i32, line[0..tab], 10);
        const got = try names.lookup(a, line[tab + 1 ..]);
        errdefer std.debug.print("name {s}, expected {d}, got {?d}\n", .{ line[tab + 1 ..], expected, got });
        try std.testing.expectEqual(if (expected < 0) null else @as(?u21, @intCast(expected)), got);
        count += 1;
    }
    std.debug.print("Unicode Zig comparisons {d}, mismatches 0\n", .{count});
}
