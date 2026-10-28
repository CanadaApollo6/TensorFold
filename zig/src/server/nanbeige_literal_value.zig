const std = @import("std");
const json = @import("json");
const numbers = @import("nanbeige_numbers.zig");
const Allocator = std.mem.Allocator;
pub const Error = error{ Invalid, PodFailure, OutOfMemory };
pub const Entry = struct { key: Node, value: Node };
pub const Node = union(enum) {
    scalar: json.Value,
    bytes: []const u8,
    complex: struct { real: f64 = 0, imag: f64 },
    ellipsis,
    list: []const Node,
    tuple: []const Node,
    set: []const Node,
    dict: []const Entry,
    invalid,
    failure,

    pub fn validate(n: Node) Error!void {
        switch (n) {
            .invalid => return error.Invalid,
            .failure => return error.PodFailure,
            .list, .tuple, .set => |items| for (items) |item| {
                try item.validate();
                if (n == .set and !item.hashable()) return error.PodFailure;
            },
            .dict => |items| for (items) |item| {
                try item.key.validate();
                try item.value.validate();
                if (!item.key.hashable()) return error.PodFailure;
            },
            else => {},
        }
    }

    fn hashable(n: Node) bool {
        return switch (n) {
            .list, .set, .dict => false,
            .tuple => |items| for (items) |item| {
                if (!item.hashable()) break false;
            } else true,
            else => true,
        };
    }

    pub fn jsonValue(n: Node, a: Allocator) Error!json.Value {
        switch (n) {
            .scalar => |v| {
                if (v == .int and std.mem.trimStart(u8, v.int, "-").len > 4300) return error.PodFailure;
                return v;
            },
            .list, .tuple => |items| {
                const out = try a.alloc(json.Value, items.len);
                for (items, out) |item, *slot| slot.* = try item.jsonValue(a);
                return .{ .array = out };
            },
            .dict => |items| {
                var kept: std.ArrayList(Entry) = .empty;
                for (items) |item| {
                    for (kept.items) |*old| {
                        if (try equal(a, old.key, item.key)) {
                            old.value = item.value;
                            break;
                        }
                    } else try kept.append(a, item);
                }
                const out = try json.newObject(a);
                for (kept.items) |item| {
                    const k = try item.key.jsonValue(a);
                    const key = switch (k) {
                        .string => |s| s,
                        .int => |s| s,
                        .bool => |b| if (b) "true" else "false",
                        .null => "null",
                        .float => try json.stringify(a, k, .{}),
                        else => return error.PodFailure,
                    };
                    try out.entries.append(a, .{ .hash = std.array_hash_map.hashString(key), .key = key, .value = try item.value.jsonValue(a) });
                }
                return .{ .object = out };
            },
            .invalid => return error.Invalid,
            else => return error.PodFailure,
        }
    }
};

fn equal(a: Allocator, x: Node, y: Node) Allocator.Error!bool {
    if (x == .bytes or y == .bytes) return x == .bytes and y == .bytes and std.mem.eql(u8, x.bytes, y.bytes);
    if (x == .tuple or y == .tuple) {
        if (x != .tuple or y != .tuple or x.tuple.len != y.tuple.len) return false;
        for (x.tuple, y.tuple) |xx, yy| if (!try equal(a, xx, yy)) return false;
        return true;
    }
    if (x == .ellipsis or y == .ellipsis) return x == .ellipsis and y == .ellipsis;
    if (x == .complex or y == .complex) {
        if (x != .complex or y != .complex) {
            const c = if (x == .complex) x.complex else y.complex;
            const scalar = if (x == .scalar) x.scalar else y.scalar;
            if (scalar == .int) {
                if (c.imag != 0 or !std.math.isFinite(c.real) or @trunc(c.real) != c.real) return false;
                return std.mem.eql(u8, scalar.int, (try numbers.integralFloat(a, c.real)).int);
            }
        }
        const xx: @FieldType(Node, "complex") = if (x == .complex) x.complex else .{ .real = x.scalar.float64() orelse return false, .imag = 0 };
        const yy: @FieldType(Node, "complex") = if (y == .complex) y.complex else .{ .real = y.scalar.float64() orelse return false, .imag = 0 };
        return xx.real == yy.real and xx.imag == yy.imag;
    }
    const xx = x.scalar;
    const yy = y.scalar;
    if (xx == .string or yy == .string) return xx == .string and yy == .string and std.mem.eql(u8, xx.string, yy.string);
    if (xx == .null or yy == .null) return xx == .null and yy == .null;
    const xi = if (xx == .bool) (if (xx.bool) "1" else "0") else if (xx == .int) xx.int else null;
    const yi = if (yy == .bool) (if (yy.bool) "1" else "0") else if (yy == .int) yy.int else null;
    if (xi != null and yi != null) return std.mem.eql(u8, xi.?, yi.?);
    if (xx == .float and yy == .float) return xx.float == yy.float;
    const f = if (xx == .float) xx.float else yy.float;
    if (!std.math.isFinite(f) or @trunc(f) != f) return false;
    return std.mem.eql(u8, (try numbers.integralFloat(a, f)).int, xi orelse yi.?);
}
