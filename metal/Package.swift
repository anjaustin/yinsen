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
            path: "test",
            sources: [
                "main.swift",
                "YinsenMetalTests.swift",
                "Test8x8.swift",
                "Benchmark4x4vs8x8.swift"
            ],
            swiftSettings: [
                .unsafeFlags(["-parse-as-library"])
            ]
        ),
        .target(
            name: "YinsenMetal",
            path: "Sources",
            resources: [
                .copy("../kernels/ternary_core.metal"),
                .copy("../kernels/ternary_8x8.metal"),
                .copy("../kernels/ternary_matvec_tiled.metal"),
                .copy("../kernels/activations.metal"),
                .copy("../kernels/layernorm.metal")
            ]
        )
    ]
)
