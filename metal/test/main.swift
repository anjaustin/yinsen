/*
 * YINSEN Metal Test Runner
 *
 * Runs all tests and benchmarks for ternary kernels.
 */

import Foundation

@main
struct Main {
    enum Mode: String {
        case test = "test"
        case benchmark = "bench"
        case all = "all"
        case test8x8 = "test8x8"
    }
    
    static func main() throws {
        let args = CommandLine.arguments
        let mode = args.count > 1 ? Mode(rawValue: args[1]) ?? .all : .all
        
        print("YINSEN Metal Test Suite")
        print(String(repeating: "=", count: 60))
        
        switch mode {
        case .test:
            try runTests()
            
        case .benchmark:
            try runBenchmarks()
            
        case .test8x8:
            try runTest8x8()
            
        case .all:
            try runTests()
            try runTest8x8()
            try runBenchmarks()
        }
    }
    
    static func runTests() throws {
        print("\n>>> Running 4×4 Tests (Exhaustive)\n")
        let tests = try YinsenMetalTests()
        let passed = try tests.runAllTests()
        if !passed {
            print("ERROR: 4×4 tests failed")
        }
    }
    
    static func runTest8x8() throws {
        print("\n>>> Running 8×8 Tests (Implementation Verification)\n")
        let tests = try Test8x8()
        let passed = tests.runAll()
        if !passed {
            print("ERROR: 8×8 tests failed")
        }
    }
    
    static func runBenchmarks() throws {
        print("\n>>> Running Benchmarks\n")
        let bench = try Benchmark4x4vs8x8()
        bench.runAll()
    }
}
