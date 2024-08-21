set -xe

zig build -Doptimize=Debug
zig-out/bin/bench $1
zig build -Doptimize=ReleaseSafe
zig-out/bin/bench $1
zig build -Doptimize=ReleaseSmall
zig-out/bin/bench $1
zig build -Doptimize=ReleaseFast
zig-out/bin/bench $1