const shared = @import("../../dflash2/execution.zig");
pub const append = shared.append;
pub const appendBlock = shared.appendBlock;
pub const Output = shared.Output;
pub const Consumer = shared.Consumer;
pub const Stream = shared.Stream;
pub const Absorb = shared.Absorb;
pub const absorbBlocks = shared.absorbBlocks;
pub const forwardToBlock = shared.forwardToBlock;
pub const forwardTo = shared.forwardTo;
pub const forwardBlock = shared.forwardBlock;
pub const forwardGiven = shared.forwardGiven;
pub const forward = shared.forward;
test {
    _ = shared;
}
