const shared = @import("../../dflash2/weights.zig");
pub const Tensor = shared.Tensor;
pub const Attention = shared.Attention;
pub const Conv = shared.Conv;
pub const Mlp = shared.Mlp;
pub const Layer = shared.Layer;
pub const Inventory = shared.Inventory;
pub const Graph = shared.Graph;
pub const names = shared.names;
pub const freeNames = shared.freeNames;
pub const load = shared.load;
test {
    _ = shared;
}
