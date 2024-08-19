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

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const alloc = gpa.allocator();
    // var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    // const alloc = arena.allocator();

    var prng = std.Random.DefaultPrng.init(0);
    const rand = prng.random();
    const scale = 10_000;

    const size = 100_000;
    const points = try alloc.alloc(TestPoint(f32), size);
    defer alloc.free(points);
    for (points) |*p| p.* = .{
        .x = rand.float(f32) * scale,
        .y = rand.float(f32) * scale,
    };
    var timer = try std.time.Timer.start();
    var timer2 = try std.time.Timer.start();
    var tree = try KdTree(TestPoint(f32), .{
        // .sort_algorithm = .unstable,
        .sort_algorithm = .median_of_medians,
    }).init(alloc, points);

    std.debug.print("init time {}\n", .{std.fmt.fmtDuration(timer2.lap())});
    defer tree.deinit(alloc);

    // validate
    try tree.validate();
    std.debug.print("validate time {}\n", .{std.fmt.fmtDuration(timer2.lap())});

    const target = TestPoint(f32){ .x = scale / 2, .y = scale / 2 };
    // nnSearch
    _ = tree.nnSearch(target);
    std.debug.print("nnSearch time {}\n", .{std.fmt.fmtDuration(timer2.lap())});

    // knnSearch
    {
        var result: [scale / 1000]u32 = undefined;
        try tree.knnSearch(alloc, target, &result);
        std.debug.print("knnSearch time {}\n", .{std.fmt.fmtDuration(timer2.lap())});
    }

    // radiusSearch
    var result = std.ArrayList(u32).init(alloc);
    defer result.deinit();
    const radius = scale / 1000;
    try tree.radiusSearch(target, radius, &result);
    std.debug.print("radiusSearch time {}\n", .{std.fmt.fmtDuration(timer2.lap())});
    std.debug.print("time {} size {}\n", .{ std.fmt.fmtDuration(timer.read()), size });
}

const std = @import("std");
const mem = std.mem;
const testing = std.testing;
const kd_tree = @import("kd-tree");
const KdTree = kd_tree.KdTree;
