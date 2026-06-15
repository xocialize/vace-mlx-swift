// swift-tools-version: 6.2
// vace-mlx-swift — Swift/MLX port of Wan2.1-VACE-1.3B (the all-in-one consumer
// image-grounding model: first-frame i2v + reference + inpaint + control + v2v, one
// Apache model). A wan-core consumer: the backbone IS wan-core `WanModel` unchanged
// (VaceWanModel subclasses it), and the net-new is the VACE Context Adapter branch
// (vace_patch_embedding + 15 VaceWanAttentionBlocks = WanAttentionBlock + before/after_proj),
// whose per-block hints inject via wan-core's runBlocks(blockResiduals:) seam. Oracle /
// converted weights: /Volumes/DEV_ARCHIVE/vace-1.3b-measure. See ENH-vace-1.3b-consumer.md.

import PackageDescription

let package = Package(
    name: "VACE",
    platforms: [
        .macOS(.v26)
    ],
    products: [
        .library(name: "VACE", targets: ["VACE"]),
    ],
    dependencies: [
        .package(url: "https://github.com/ml-explore/mlx-swift.git", from: "0.30.0"),
        // The neutral Wan substrate (WanModel + 16-ch VAE + umT5 + RoPE + schedulers + loader).
        .package(path: "../wan-core-mlx-swift"),
    ],
    targets: [
        .target(
            name: "VACE",
            dependencies: [
                .product(name: "WanCore", package: "wan-core-mlx-swift"),
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
                .product(name: "MLXFast", package: "mlx-swift"),
                .product(name: "MLXRandom", package: "mlx-swift"),
            ],
            path: "Sources/VACE"
        ),
        .testTarget(
            name: "VACETests",
            dependencies: [
                "VACE",
                .product(name: "WanCore", package: "wan-core-mlx-swift"),
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
            ],
            path: "Tests/VACETests"
        ),
    ]
)
