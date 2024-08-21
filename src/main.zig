const std = @import("std");
const kdtree = @import("kd-tree");
const rl = @cImport(@cInclude("raylib.h"));
const width = 1280;
const height = 920;
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

pub fn main() !void {
    rl.InitWindow(width, height, "kd-tree search results");
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    const alloc = gpa.allocator();
    var points = std.ArrayList(rl.Vector2).init(alloc);
    const npoints = 100;
    var prng = std.Random.DefaultPrng.init(0);
    const rand = prng.random();
    for (0..npoints) |_| {
        try points.append(.{
            .x = rand.float(f32) * width,
            .y = rand.float(f32) * height,
        });
    }
    // for (test_points) |p| {
    //     try points.append(.{ .x = p[0] / 10.0 * width, .y = p[1] / 10.0 * height });
    // }
    var tree = try kdtree.KdTree(rl.Vector2, .{}).init(alloc, points.items);
    defer tree.deinit(alloc);
    const target = rl.Vector2{ .x = width * 0.3, .y = height * 0.5 };
    const c_radius = 5;
    const closest = tree.nnSearch(target);
    var kclosest: [5]u32 = undefined;
    try tree.knnSearch(alloc, target, &kclosest);
    var rclosest = std.ArrayList(u32).init(alloc);
    defer rclosest.deinit();
    const search_radius = 300;
    try tree.radiusSearch(target, search_radius, &rclosest);
    const outline_mul = 2.0;
    while (!rl.WindowShouldClose()) {
        const key = rl.GetKeyPressed();
        if (key == rl.KEY_Q) break;

        rl.BeginDrawing();
        rl.ClearBackground(rl.BLACK);
        for (points.items) |p| {
            rl.DrawCircleV(p, c_radius, rl.GRAY);
        }
        for (rclosest.items) |i| {
            rl.DrawCircleV(points.items[i], c_radius, rl.BLUE);
        }
        for (kclosest) |i| {
            rl.DrawCircleV(points.items[i], c_radius, rl.GREEN);
        }
        rl.DrawCircleLinesV(target, search_radius, rl.RED);
        rl.DrawCircleLinesV(
            points.items[closest.nearest_point_idx],
            c_radius * outline_mul,
            rl.WHITE,
        );
        rl.DrawCircleV(target, c_radius, rl.RED);
        rl.EndDrawing();
    }
    rl.CloseWindow();
}
