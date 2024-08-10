# About

A [k-d tree](https://en.wikipedia.org/wiki/K-d_tree) implementation ported from https://github.com/gishi523/kd-tree

# API

* Floats only.  Accepts arrays or homogenous structures of any zig float type
* Search for single nearest neighbor
* Search for k nearest neighbors
* Search for all neighbors within radius
* Visit all nodes with user provided context
* Validate all nodes

# Usage

This package is meant to be consumed with the [zig build system](https://ziglang.org/learn/build-system) and imported as a module.

```console
# latest commit
zig fetch --save=kdtree git+https://github.com/travisstaloch/kd-tree
```
```console
# specific commit
zig fetch --save=kdtree https://github.com/travisstaloch/kd-tree/archive/<commit-hash>.tar.gz
```
```zig
// build.zig
exe.root_module.addImport("kdtree", b.dependency("kdtree", .{}).module("kdtree"));
```
```zig
// myapp.zig
const kdtree = @import("kdtree");
```

### Example test

This code is from [src/root.zig](src/root.zig) where there are also other similar tests.  And there is [demo raylib app](src/main.zig) too.

```zig
test "basic usage" {
    const MyPoint = struct { x: f32, y: f32 };
    const my_points: [8]MyPoint = .{
        .{ .x = 2.7175806e0, .y = 2.5162528e0 },
        .{ .x = 3.0172718e0, .y = 1.1619115e-1 },
        .{ .x = 3.9525537e0, .y = 2.9364395e-1 },
        .{ .x = 7.0706835e0, .y = 7.9604106e0 },
        .{ .x = 2.6941679e0, .y = 1.9487169 },
        .{ .x = 3.7812352e0, .y = 8.3156556e-1 },
        .{ .x = 7.326651e-1, .y = 1.2251115e0 },
        .{ .x = 8.401675e-1, .y = 2.9021428e0 },
    };

    // TODO: const KdTree = @import("kdtree").KdTree;
    const Tree = KdTree(MyPoint);
    var tree = try Tree.init(std.testing.allocator, &my_points);
    defer tree.deinit(std.testing.allocator);

    // validate
    try tree.validate();

    // visit
    var count: u8 = 0;
    tree.visit(.{}, &count, struct {
        fn visit(_: Tree, _: Tree.Node, c: *u8) bool {
            c.* += 1;
            return true;
        }
    }.visit);
    try testing.expectEqual(my_points.len, count);

    const target: MyPoint = .{ .x = 5, .y = 5 };

    // nearest
    const nearest = tree.nnSearch(target);
    try testing.expectEqual(0, nearest.nearest_point_idx);

    // knn
    var knn_result: [2]u32 = undefined;
    try tree.knnSearch(std.testing.allocator, target, &knn_result);
    try testing.expectEqual(0, knn_result[0]);
    try testing.expectEqual(3, knn_result[1]);

    // radius
    var radius_search_results = std.ArrayList(u32).init(std.testing.allocator);
    defer radius_search_results.deinit();
    const radius = 4;
    try tree.radiusSearch(target, radius, &radius_search_results);
    try testing.expectEqual(3, radius_search_results.items.len);
}
```