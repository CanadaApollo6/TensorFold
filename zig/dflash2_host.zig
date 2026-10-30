const dflash = @import("src/families/qwen27/dflash/root.zig");
test {
    _ = dflash;
    _ = @import("src/families/dflash2/rotary.zig");
    _ = @import("src/families/dflash2/lattice.zig");
}
