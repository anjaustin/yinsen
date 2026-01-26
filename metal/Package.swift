// swift-tools-version:5.9
import PackageDescription

let package = Package(
    name: "YinsenMetal",
    platforms: [
        .macOS(.v14)
    ],
    products: [
        .executable(name: "yinsen-metal-tests", targets: ["YinsenMetalTests"]),
        .library(name: "YinsenMetal", targets: ["YinsenMetal"])
    ],
    targets: [
        .executableTarget(
            name: "YinsenMetalTests",
            dependencies: ["YinsenMetal"],
            path: "test"
        ),
        .target(
            name: "YinsenMetal",
            path: "Sources",
            resources: [
                .copy("../kernels/ternary_core.metal"),
                .copy("../kernels/activations.metal"),
                .copy("../kernels/layernorm.metal")
            ]
        )
    ]
)
