pub fn KdTree(comptime T: type) type {
    return struct {
        /// original points
        points: []const Point,
        /// storage for all nodes.  root node is always first.
        nodes: std.ArrayListUnmanaged(Node) = .{},

        const not_supported = @compileError("TODO support '" ++ @tagName(info) ++ "'");
        const Self = @This();
        const info = @typeInfo(Point);
        pub const len: u8 = switch (info) {
            .Array => |a| a.len,
            .Struct => |s| s.fields.len,
            else => not_supported,
        };
        pub const Point = T;
        pub const Child = switch (info) {
            .Array => |a| a.child,
            .Struct => |s| s.fields[0].type,
            else => not_supported,
        };
        const Fe = std.meta.FieldEnum(Point);

        pub const Node = struct {
            /// index to the original point
            point_idx: u32,
            /// pointers to the child nodes
            next: [2]Idx = .{ sentinel, sentinel },
            /// dimension's axis
            axis: u8,

            const sentinel: Idx = @enumFromInt(std.math.maxInt(u32));

            /// strongly typed u32 to prevent confusing with other integer types
            const Idx = enum(u32) {
                root = 0,
                _,
                pub inline fn int(i: Idx) u32 {
                    return @intFromEnum(i);
                }
            };
        };

        /// Builds a k-d tree from points.
        pub fn init(alloc: mem.Allocator, points: []const Point) !Self {
            const point_indices = try alloc.alloc(u32, points.len);
            defer alloc.free(point_indices);
            var i: u32 = 0;
            while (i < points.len) : (i += 1) point_indices[i] = i;

            var self = Self{ .points = points };
            try self.nodes.ensureTotalCapacity(alloc, points.len);
            _ = try self.initImpl(alloc, point_indices, 0);
            return self;
        }

        /// Free all nodes and set root to null.
        pub fn deinit(self: *Self, alloc: mem.Allocator) void {
            self.nodes.deinit(alloc);
        }

        /// Validates each node in the tree.
        pub fn validate(self: *const Self) !void {
            return self.validateImpl(.root, 0);
        }

        const child_max: Child = switch (@typeInfo(Child)) {
            .Float => std.math.floatMax(Child),
            else => not_supported,
        };

        /// Search for the single nearest neighbor.
        pub fn nnSearch(self: *const Self, target: Point) NnResult {
            var result: NnResult = .{
                .min_distance = child_max,
                .nearest_point_idx = undefined,
            };
            self.nnSearchImpl(target, .root, &result);

            return result;
        }

        /// Search for k-nearest neighbors where k = result_indices.len.
        /// Resulting point indices will be written to result_indices.
        pub fn knnSearch(
            self: *const Self,
            allocator: mem.Allocator,
            target: Point,
            result_indices: []u32,
        ) !void {
            var queue = KnnQueue.init(allocator, @intCast(result_indices.len));
            defer queue.deinit();

            try self.knnSearchRecursive(target, .root, &queue, @intCast(result_indices.len));

            for (0..queue.storage.items.len) |i| {
                result_indices[i] = queue.storage.items[i][1];
            }
        }

        /// Search for neighbors within radius.  clears result_indices before
        /// searching.
        pub fn radiusSearch(
            self: *const Self,
            target: Point,
            radius: Child,
            result_indices: *std.ArrayList(u32),
        ) !void {
            result_indices.clearRetainingCapacity();
            try self.radiusSearchRecursive(target, .root, result_indices, radius);
        }

        /// Visit all nodes calling visitor with each.
        /// Pass `options.start_node_idx = .root` which (default value) to
        /// search all nodes or @enumFromInt(node_idx) to start somewhere else.
        /// `visitor` may return false to prevent decending into child nodes.
        pub fn visit(
            self: Self,
            options: struct { start_node_idx: Node.Idx = .root },
            ctx: anytype,
            visitor: *const fn (tree: Self, node: Node, ctx: @TypeOf(ctx)) bool,
        ) void {
            const n = self.nodes.items[options.start_node_idx.int()];
            if (!visitor(self, n, ctx)) return;
            if (n.next[0] != Node.sentinel) {
                self.visit(.{ .start_node_idx = n.next[0] }, ctx, visitor);
            }
            if (n.next[1] != Node.sentinel) {
                self.visit(.{ .start_node_idx = n.next[1] }, ctx, visitor);
            }
        }

        /// returns field at axis.  this function exists to provide one api for
        /// accessing struct and array fields.
        fn pointField(t: Point, axis: u8) Child {
            return switch (info) {
                .Array => return t[axis],
                .Struct => {
                    const fe: Fe = @enumFromInt(axis);
                    return switch (fe) {
                        inline else => |tag| @field(t, @tagName(tag)),
                    };
                },
                else => not_supported,
            };
        }

        /// build tree recursively.
        fn initImpl(
            self: *Self,
            alloc: mem.Allocator,
            point_indices: []u32,
            depth: u32,
        ) !Node.Idx {
            if (point_indices.len == 0) return Node.sentinel;

            const axis: u8 = @truncate(depth % len);
            const mid = (point_indices.len - 1) / 2;

            // sort indices by points at axis
            const Ctx = struct { points: []const Point, axis: u8 };
            const _ctx = Ctx{ .points = self.points, .axis = axis };
            std.mem.sortUnstable(u32, point_indices, _ctx, struct {
                fn lessThan(ctx: Ctx, lhs: u32, rhs: u32) bool {
                    return pointField(ctx.points[lhs], ctx.axis) <
                        pointField(ctx.points[rhs], ctx.axis);
                }
            }.lessThan);

            const node_idx: Node.Idx = @enumFromInt(self.nodes.items.len);
            self.nodes.appendAssumeCapacity(.{
                .point_idx = point_indices[mid],
                .axis = axis,
            });

            const next0 = try self.initImpl(alloc, point_indices[0..mid], depth + 1);
            self.nodes.items[node_idx.int()].next[0] = next0;

            const next1 = try self.initImpl(alloc, point_indices[mid + 1 ..], depth + 1);
            self.nodes.items[node_idx.int()].next[1] = next1;

            return node_idx;
        }

        fn validateImpl(self: *const Self, node_idx: Node.Idx, depth: u32) !void {
            if (node_idx == Node.sentinel) return;

            const n = self.nodes.items[node_idx.int()];
            const has_next0 = n.next[0] != Node.sentinel;
            const has_next1 = n.next[1] != Node.sentinel;

            if (has_next0 and has_next1) {
                const next0 = self.nodes.items[n.next[0].int()];
                if (pointField(self.points[n.point_idx], n.axis) <
                    pointField(self.points[next0.point_idx], n.axis))
                    return error.Invalid;

                const next1 = self.nodes.items[n.next[1].int()];
                if (pointField(self.points[n.point_idx], n.axis) >
                    pointField(self.points[next1.point_idx], n.axis))
                    return error.Invalid;
            }

            if (has_next0) try self.validateImpl(n.next[0], depth + 1);

            if (has_next1) try self.validateImpl(n.next[1], depth + 1);
        }

        /// distance squared.  caller should sqrt result when necessary.
        fn distance2(p: Point, q: Point) Child {
            var dist: Child = 0;
            inline for (0..len) |axis| {
                const pf = pointField(p, axis);
                const qf = pointField(q, axis);
                const diff = pf - qf;
                dist += diff * diff;
            }
            return dist;
        }

        pub const NnResult = struct { nearest_point_idx: u32, min_distance: Child };
        /// Search the nearest neighbor recursively.
        fn nnSearchImpl(
            self: *const Self,
            target: Point,
            node_idx: Node.Idx,
            result: *NnResult,
        ) void {
            if (node_idx == Node.sentinel) return;

            const node = self.nodes.items[node_idx.int()];
            const train = self.points[node.point_idx];

            const dist = distance2(target, train);
            if (dist < result.min_distance) {
                result.min_distance = dist;
                result.nearest_point_idx = node.point_idx;
            }

            const axis = node.axis;
            const dir = pointField(train, axis) < pointField(target, axis);
            self.nnSearchImpl(target, node.next[@intFromBool(dir)], result);

            const diff = @abs(pointField(target, axis) - pointField(train, axis));
            if (diff < result.min_distance) {
                self.nnSearchImpl(target, node.next[@intFromBool(!dir)], result);
            }
        }

        pub const QueueItem = struct { Child, u32 };
        pub const KnnQueue = BoundedPriorityQueue(QueueItem);

        /// Search k-nearest neighbors recursively.
        fn knnSearchRecursive(
            self: *const Self,
            target: Point,
            node_idx: Node.Idx,
            queue: *KnnQueue,
            k: u32,
        ) !void {
            if (node_idx == Node.sentinel) return;

            const node = self.nodes.items[node_idx.int()];
            const train = self.points[node.point_idx];

            const dist = distance2(target, train);
            try queue.push(.{ dist, node.point_idx });

            const axis = node.axis;
            const dir = pointField(train, axis) < pointField(target, axis);
            try self.knnSearchRecursive(target, node.next[@intFromBool(dir)], queue, k);

            const diff = @abs(pointField(target, axis) - pointField(train, axis));
            if (queue.count() < k or diff < queue.last()[0])
                try self.knnSearchRecursive(target, node.next[@intFromBool(!dir)], queue, k);
        }

        /// Search nearest neighbors recursively.
        fn radiusSearchRecursive(
            self: *const Self,
            target: Point,
            node_idx: Node.Idx,
            result: *std.ArrayList(u32),
            radius: Child,
        ) !void {
            if (node_idx == Node.sentinel) return;

            const node = self.nodes.items[node_idx.int()];
            const train = self.points[node.point_idx];

            const dist = @sqrt(distance2(target, train));
            if (dist < radius) try result.append(node.point_idx);

            const axis = node.axis;
            const dir = pointField(train, axis) < pointField(target, axis);
            try self.radiusSearchRecursive(
                target,
                node.next[@intFromBool(dir)],
                result,
                radius,
            );

            const diff = @abs(pointField(target, axis) - pointField(train, axis));
            if (diff < radius) {
                try self.radiusSearchRecursive(
                    target,
                    node.next[@intFromBool(!dir)],
                    result,
                    radius,
                );
            }
        }
    };
}

/// a simple priority queue backed by an array list which is always sorted by
/// T's first field.  T must be an array or tuple.
pub fn BoundedPriorityQueue(comptime T: type) type {
    return struct {
        storage: std.ArrayListUnmanaged(T) = .{},
        alloc: mem.Allocator,
        bound: u16,

        const Self = @This();

        pub fn init(alloc: mem.Allocator, bound: u16) Self {
            std.debug.assert(bound != 0);
            return .{ .alloc = alloc, .bound = bound };
        }

        pub fn push(self: *Self, val: T) !void {
            const index = std.sort.lowerBound(T, self.storage.items, val, compareFn);
            try self.storage.insert(self.alloc, index, val);
            if (self.storage.items.len > self.bound) {
                self.storage.items.len -= 1;
                std.debug.assert(self.storage.items.len == self.bound);
            }
        }

        pub fn count(self: Self) usize {
            return self.storage.items.len;
        }

        pub fn last(self: Self) T {
            std.debug.assert(self.storage.items.len > 0);
            return self.storage.items[self.storage.items.len - 1];
        }

        pub fn deinit(self: *Self) void {
            self.storage.deinit(self.alloc);
        }

        fn compareFn(ctx: T, a: T) std.math.Order {
            return std.math.order(a[0], ctx[0]);
        }
    };
}

const talloc = std.testing.allocator;
fn TestPoint(comptime T: type) type {
    return extern struct {
        x: T,
        y: T,

        pub fn sub(a: @This(), b: @This()) @This() {
            return .{ .x = a.x - b.x, .y = a.y - b.y };
        }
        pub fn len2(a: @This()) T {
            return a.x * a.x + a.y * a.y;
        }
    };
}

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

/// generated with std.Random.DefaultPrng.init(0) and Random.float() * scale;
const test_points: [32][2]f32 = .{
    .{ 2.7175806e0, 2.5162528e0 },
    .{ 3.0172718e0, 1.1619115e-1 },
    .{ 3.9525537e0, 2.9364395e-1 },
    .{ 7.0706835e0, 7.9604106e0 },
    .{ 2.6941679e0, 1.9487169 },
    .{ 3.7812352e0, 8.3156556e-1 },
    .{ 7.326651e-1, 1.2251115e0 },
    .{ 8.401675e-1, 2.9021428e0 },
    .{ 9.8444605e0, 8.574822e-1 },
    .{ 3.5539699e0, 4.863959e0 },
    .{ 3.5543716e0, 1.6830349e0 },
    .{ 4.5475838e-1, 4.8341055e0 },
    .{ 3.1293128e0, 4.6348634e0 },
    .{ 2.1604078e0, 9.860038e0 },
    .{ 8.796754e0, 5.847299e-1 },
    .{ 4.85139e0, 2.514596e0 },
    .{ 4.023865e0, 8.238647e0 },
    .{ 8.925945e0, 1.8834783e0 },
    .{ 5.426202e0, 3.5715155e0 },
    .{ 5.426679e0, 1.8324531e0 },
    .{ 6.3002505e0, 3.0327914e0 },
    .{ 2.6707802e0, 5.755902e-1 },
    .{ 3.6478248e-1, 1.904851e0 },
    .{ 2.3302314e0, 1.367808e0 },
    .{ 2.3675838e0, 4.866859e0 },
    .{ 9.925309e0, 9.252941e0 },
    .{ 6.964834e0, 1.2647063e0 },
    .{ 1.8314324e-1, 5.777245e0 },
    .{ 7.2437263e0, 4.7264957e0 },
    .{ 3.7840703e0, 5.706524e0 },
    .{ 9.081715e0, 7.4064455e0 },
    .{ 8.429312e-2, 4.393729e0 },
};

/// run test with both struct and array types (TestPoint(T) and [2]T)
fn testWithT(comptime T: type) !void {
    try testWithKdTree(KdTree(TestPoint(T)));
    try testWithKdTree(KdTree([2]T));
}

fn testWithKdTree(comptime Tree: type) !void {
    const scale = 10;
    const Pt = Tree.Point;
    const T = Tree.Child;

    var points: [32]Pt = undefined;
    for (0..32) |i| points[i] = if (Tree.info == .Struct) .{
        .x = @floatCast(test_points[i][0]),
        .y = @floatCast(test_points[i][1]),
    } else if (Tree.info == .Array) .{
        @floatCast(test_points[i][0]),
        @floatCast(test_points[i][1]),
    } else unreachable;

    var tree = try Tree.init(talloc, &points);
    defer tree.deinit(talloc);

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
    try testing.expectEqual(32, count);

    // nnSearch
    {
        const p = if (Tree.info == .Struct)
            TestPoint(T){ .x = scale / 2, .y = scale / 2 }
        else
            .{ scale / 2, scale / 2 };
        const result = tree.nnSearch(p);
        try testing.expectEqual(29, result.nearest_point_idx);
        try testing.expectEqual(points[29], points[result.nearest_point_idx]);
    }
    {
        const p = if (Tree.info == .Struct)
            TestPoint(T){ .x = 0, .y = 0 }
        else
            .{ 0, 0 };
        const result = tree.nnSearch(p);
        try testing.expectEqual(6, result.nearest_point_idx);
        try testing.expectEqual(points[6], points[result.nearest_point_idx]);
    }
    {
        const p = if (Tree.info == .Struct)
            TestPoint(T){ .x = scale, .y = scale }
        else
            .{ scale, scale };
        const result = tree.nnSearch(p);
        try testing.expectEqual(25, result.nearest_point_idx);
        try testing.expectEqual(points[25], points[result.nearest_point_idx]);
    }
    {
        const p = if (Tree.info == .Struct)
            TestPoint(T){ .x = 0, .y = scale }
        else
            .{ 0, scale };
        const result = tree.nnSearch(p);
        try testing.expectEqual(13, result.nearest_point_idx);
        try testing.expectEqual(points[13], points[result.nearest_point_idx]);
    }
    {
        const p = if (Tree.info == .Struct)
            TestPoint(T){ .x = scale, .y = 0 }
        else
            .{ scale, 0 };
        const result = tree.nnSearch(p);
        try testing.expectEqual(8, result.nearest_point_idx);
        try testing.expectEqual(points[8], points[result.nearest_point_idx]);
    }

    const target = if (Tree.info == .Struct)
        TestPoint(T){ .x = scale / 2, .y = scale / 2 }
    else
        .{ scale / 2, scale / 2 };

    // knnSearch
    {
        var result: [4]u32 = undefined;
        try tree.knnSearch(talloc, target, &result);

        if (Tree.info == .Struct) {
            const p1 = points[result[0]];
            const p2 = points[result[1]];
            const p3 = points[result[2]];
            const p4 = points[result[3]];
            try testing.expect(target.sub(p1).len2() < target.sub(p2).len2());
            try testing.expect(target.sub(p2).len2() < target.sub(p3).len2());
            try testing.expect(target.sub(p3).len2() < target.sub(p4).len2());
        } else {
            const _target = TestPoint(T){ .x = target[0], .y = target[1] };
            const p1 = TestPoint(T){ .x = points[result[0]][0], .y = points[result[0]][1] };
            const p2 = TestPoint(T){ .x = points[result[1]][0], .y = points[result[1]][1] };
            const p3 = TestPoint(T){ .x = points[result[2]][0], .y = points[result[2]][1] };
            const p4 = TestPoint(T){ .x = points[result[3]][0], .y = points[result[3]][1] };
            try testing.expect(_target.sub(p1).len2() < _target.sub(p2).len2());
            try testing.expect(_target.sub(p2).len2() < _target.sub(p3).len2());
            try testing.expect(_target.sub(p3).len2() < _target.sub(p4).len2());
        }
    }

    // radiusSearch
    var result = std.ArrayList(u32).init(talloc);
    defer result.deinit();
    const radius = 3;
    try tree.radiusSearch(target, radius, &result);
    for (result.items) |i| {
        if (Tree.info == .Struct) {
            const p = points[i];
            try testing.expect(@sqrt(target.sub(p).len2()) < radius);
        } else {
            const _target = TestPoint(T){ .x = target[0], .y = target[1] };
            const p = TestPoint(T){ .x = points[i][0], .y = points[i][1] };
            try testing.expect(@sqrt(_target.sub(p).len2()) < radius);
        }
    }

    // check count outside radius
    var count2: u8 = 0;
    for (points) |p| {
        if (Tree.info == .Struct) {
            count2 += @intFromBool(@sqrt(target.sub(p).len2()) >= radius);
        } else {
            const _target = TestPoint(T){ .x = target[0], .y = target[1] };
            const p1 = TestPoint(T){ .x = p[0], .y = p[1] };
            count2 += @intFromBool(@sqrt(_target.sub(p1).len2()) >= radius);
        }
    }
    try testing.expectEqual(points.len - result.items.len, count2);
}

test KdTree {
    try testWithT(f16);
    try testWithT(f32);
    try testWithT(f64);
    try testWithT(f80);
    try testWithT(f128);
}

test "larger dataset" {
    var prng = std.Random.DefaultPrng.init(0);
    const rand = prng.random();
    const scale = 10_000;

    const size = 100_000;
    const points = try talloc.alloc(TestPoint(f32), size);
    defer talloc.free(points);
    for (points) |*p| p.* = .{
        .x = rand.float(f32) * scale,
        .y = rand.float(f32) * scale,
    };
    var tree = try KdTree(TestPoint(f32)).init(talloc, points);
    defer tree.deinit(talloc);

    // validate
    try tree.validate();

    // knnSearch
    const target = TestPoint(f32){ .x = scale / 2, .y = scale / 2 };
    {
        var result: [scale / 1000]u32 = undefined;
        try tree.knnSearch(talloc, target, &result);
        for (0..result.len - 1) |i| {
            const p1 = points[result[i]];
            const p2 = points[result[i + 1]];
            try testing.expect(target.sub(p1).len2() < target.sub(p2).len2());
        }
    }

    // radiusSearch
    var result = std.ArrayList(u32).init(talloc);
    defer result.deinit();
    const radius = scale / 1000;
    try tree.radiusSearch(target, radius, &result);
    for (result.items) |i| {
        const p = points[i];
        try testing.expect(@sqrt(target.sub(p).len2()) < radius);
    }
}

const std = @import("std");
const mem = std.mem;
const testing = std.testing;
