/*
 * YINSEN Metal - 8×8 Kernel Tests
 *
 * Implementation tests for the 8×8 ternary matvec kernel.
 * These are NOT exhaustive proofs (impossible at 10^30 configs).
 * They verify the implementation is correct via:
 *   1. Boundary cases (all +1, all -1, all 0, etc.)
 *   2. Structural cases (identity-like, checkerboard, etc.)
 *   3. Random sampling (statistical confidence)
 *   4. Property verification (linearity, etc.)
 */

import Metal
import Foundation

class Test8x8 {
    let device: MTLDevice
    let commandQueue: MTLCommandQueue
    let library: MTLLibrary
    
    // Pipeline states
    var pipeline8x8: MTLComputePipelineState?
    var pipeline8x8Single: MTLComputePipelineState?
    var pipelineTiled: MTLComputePipelineState?
    
    init() throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw TestError.noDevice
        }
        self.device = device
        
        guard let queue = device.makeCommandQueue() else {
            throw TestError.noCommandQueue
        }
        self.commandQueue = queue
        
        // Load kernel source
        let kernelPath = URL(fileURLWithPath: #file)
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .appendingPathComponent("kernels")
            .appendingPathComponent("ternary_8x8.metal")
        
        let source = try String(contentsOf: kernelPath)
        self.library = try device.makeLibrary(source: source, options: nil)
        
        // Create pipelines
        if let fn = library.makeFunction(name: "ternary_matvec_8x8") {
            pipeline8x8 = try device.makeComputePipelineState(function: fn)
        }
        if let fn = library.makeFunction(name: "ternary_matvec_8x8_single") {
            pipeline8x8Single = try device.makeComputePipelineState(function: fn)
        }
        if let fn = library.makeFunction(name: "ternary_matvec_tiled_8x8") {
            pipelineTiled = try device.makeComputePipelineState(function: fn)
        }
        
        print("Test8x8 initialized on \(device.name)")
    }
    
    // =========================================================================
    // CPU REFERENCE
    // =========================================================================
    
    /// Canonical encoding: 00=0, 01=+1, 10=-1, 11=reserved(0)
    func tritSign(_ encoding: UInt8) -> Int {
        if encoding == 1 { return 1 }
        if encoding == 2 { return -1 }
        return 0
    }
    
    func cpuDot8(packed: UInt16, x: [Float]) -> Float {
        var sum: Float = 0
        for i in 0..<8 {
            let encoding = UInt8((packed >> (i * 2)) & 0x3)
            let sign = tritSign(encoding)
            sum += Float(sign) * x[i]
        }
        return sum
    }
    
    func cpuMatvec8x8(W: [UInt16], x: [Float]) -> [Float] {
        var y = [Float](repeating: 0, count: 8)
        for row in 0..<8 {
            y[row] = cpuDot8(packed: W[row], x: x)
        }
        return y
    }
    
    /// Canonical encoding: +1 -> 01, -1 -> 10, 0 -> 00
    func encodeTrit(_ val: Int) -> UInt8 {
        if val > 0 { return 0x1 }
        if val < 0 { return 0x2 }  // 10 (canonical)
        return 0x0
    }
    
    func packRow(_ trits: [Int]) -> UInt16 {
        var packed: UInt16 = 0
        for i in 0..<8 {
            packed |= UInt16(encodeTrit(trits[i])) << (i * 2)
        }
        return packed
    }
    
    // =========================================================================
    // GPU EXECUTION
    // =========================================================================
    
    func gpuMatvec8x8(W: [UInt16], x: [Float]) -> [Float]? {
        guard let pipeline = pipeline8x8 else { return nil }
        
        let wBuffer = device.makeBuffer(bytes: W, length: W.count * 2, options: .storageModeShared)!
        let xBuffer = device.makeBuffer(bytes: x, length: x.count * 4, options: .storageModeShared)!
        let yBuffer = device.makeBuffer(length: 8 * 4, options: .storageModeShared)!
        
        guard let cmdBuffer = commandQueue.makeCommandBuffer(),
              let encoder = cmdBuffer.makeComputeCommandEncoder() else { return nil }
        
        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(wBuffer, offset: 0, index: 0)
        encoder.setBuffer(xBuffer, offset: 0, index: 1)
        encoder.setBuffer(yBuffer, offset: 0, index: 2)
        
        let gridSize = MTLSize(width: 8, height: 1, depth: 1)
        let threadgroupSize = MTLSize(width: 8, height: 1, depth: 1)
        encoder.dispatchThreads(gridSize, threadsPerThreadgroup: threadgroupSize)
        
        encoder.endEncoding()
        cmdBuffer.commit()
        cmdBuffer.waitUntilCompleted()
        
        let ptr = yBuffer.contents().bindMemory(to: Float.self, capacity: 8)
        return Array(UnsafeBufferPointer(start: ptr, count: 8))
    }
    
    // =========================================================================
    // BOUNDARY TESTS
    // =========================================================================
    
    func testBoundaries() -> Bool {
        print("\n=== 8×8 Boundary Tests ===")
        var passed = 0
        var failed = 0
        
        let x: [Float] = [1, 2, 3, 4, 5, 6, 7, 8]
        let xSum = x.reduce(0, +)  // 36
        
        // Test 1: All zeros
        let allZeros = [UInt16](repeating: 0x0000, count: 8)
        if let result = gpuMatvec8x8(W: allZeros, x: x) {
            let expected = [Float](repeating: 0, count: 8)
            if result == expected {
                print("  [PASS] All zeros")
                passed += 1
            } else {
                print("  [FAIL] All zeros: expected \(expected), got \(result)")
                failed += 1
            }
        }
        
        // Test 2: All +1 (row sums to xSum)
        // Encoding: 01 01 01 01 01 01 01 01 = 0x5555
        let allOnes = [UInt16](repeating: 0x5555, count: 8)
        if let result = gpuMatvec8x8(W: allOnes, x: x) {
            let expected = [Float](repeating: xSum, count: 8)
            if zip(result, expected).allSatisfy({ abs($0 - $1) < 1e-5 }) {
                print("  [PASS] All +1")
                passed += 1
            } else {
                print("  [FAIL] All +1: expected \(expected), got \(result)")
                failed += 1
            }
        }
        
        // Test 3: All -1 (row sums to -xSum)
        // Canonical encoding: 10 10 10 10 10 10 10 10 = 0xAAAA
        let allNegOnes = [UInt16](repeating: 0xAAAA, count: 8)
        if let result = gpuMatvec8x8(W: allNegOnes, x: x) {
            let expected = [Float](repeating: -xSum, count: 8)
            if zip(result, expected).allSatisfy({ abs($0 - $1) < 1e-5 }) {
                print("  [PASS] All -1")
                passed += 1
            } else {
                print("  [FAIL] All -1: expected \(expected), got \(result)")
                failed += 1
            }
        }
        
        // Test 4: Identity-like (diagonal +1)
        var identity = [UInt16](repeating: 0, count: 8)
        for i in 0..<8 {
            identity[i] = UInt16(0x1) << (i * 2)  // +1 at position i
        }
        if let result = gpuMatvec8x8(W: identity, x: x) {
            if zip(result, x).allSatisfy({ abs($0 - $1) < 1e-5 }) {
                print("  [PASS] Identity-like")
                passed += 1
            } else {
                print("  [FAIL] Identity-like: expected \(x), got \(result)")
                failed += 1
            }
        }
        
        // Test 5: Negation (diagonal -1)
        var negIdentity = [UInt16](repeating: 0, count: 8)
        for i in 0..<8 {
            negIdentity[i] = UInt16(0x2) << (i * 2)  // -1 at position i (canonical: 10)
        }
        if let result = gpuMatvec8x8(W: negIdentity, x: x) {
            let expected = x.map { -$0 }
            if zip(result, expected).allSatisfy({ abs($0 - $1) < 1e-5 }) {
                print("  [PASS] Negation")
                passed += 1
            } else {
                print("  [FAIL] Negation: expected \(expected), got \(result)")
                failed += 1
            }
        }
        
        // Test 6: Checkerboard (+1, -1, +1, -1, ...)
        // Row i: alternating starting with +1
        // Canonical: 01 10 01 10 01 10 01 10 = 0x9999
        let checkerboard = [UInt16](repeating: 0x9999, count: 8)
        if let result = gpuMatvec8x8(W: checkerboard, x: x) {
            // Expected: 1 - 2 + 3 - 4 + 5 - 6 + 7 - 8 = -4
            let expected = [Float](repeating: -4, count: 8)
            if zip(result, expected).allSatisfy({ abs($0 - $1) < 1e-5 }) {
                print("  [PASS] Checkerboard")
                passed += 1
            } else {
                print("  [FAIL] Checkerboard: expected \(expected), got \(result)")
                failed += 1
            }
        }
        
        // Test 7: First half +1, second half -1
        // Canonical: 01 01 01 01 10 10 10 10 = 0xAA55
        let halfHalf = [UInt16](repeating: 0xAA55, count: 8)
        if let result = gpuMatvec8x8(W: halfHalf, x: x) {
            // Expected: (1+2+3+4) - (5+6+7+8) = 10 - 26 = -16
            let expected = [Float](repeating: -16, count: 8)
            if zip(result, expected).allSatisfy({ abs($0 - $1) < 1e-5 }) {
                print("  [PASS] Half +1 / Half -1")
                passed += 1
            } else {
                print("  [FAIL] Half +1 / Half -1: expected \(expected), got \(result)")
                failed += 1
            }
        }
        
        print("  Boundary tests: \(passed) passed, \(failed) failed")
        return failed == 0
    }
    
    // =========================================================================
    // RANDOM SAMPLING
    // =========================================================================
    
    func testRandomSampling(iterations: Int = 100000) -> Bool {
        print("\n=== 8×8 Random Sampling (\(iterations) iterations) ===")
        
        var passed = 0
        var failed = 0
        var maxError: Float = 0
        
        for iter in 0..<iterations {
            // Random weights
            var W = [UInt16](repeating: 0, count: 8)
            for row in 0..<8 {
                var packed: UInt16 = 0
                for col in 0..<8 {
                    let trit = Int.random(in: -1...1)
                    packed |= UInt16(encodeTrit(trit)) << (col * 2)
                }
                W[row] = packed
            }
            
            // Random input
            let x = (0..<8).map { _ in Float.random(in: -10...10) }
            
            // CPU reference
            let cpuResult = cpuMatvec8x8(W: W, x: x)
            
            // GPU result
            guard let gpuResult = gpuMatvec8x8(W: W, x: x) else {
                print("  [FAIL] GPU execution failed at iteration \(iter)")
                failed += 1
                continue
            }
            
            // Compare
            var match = true
            for i in 0..<8 {
                let error = abs(cpuResult[i] - gpuResult[i])
                maxError = max(maxError, error)
                if error > 1e-4 {
                    match = false
                }
            }
            
            if match {
                passed += 1
            } else {
                failed += 1
                if failed <= 5 {
                    print("  [FAIL] Mismatch at iteration \(iter)")
                    print("    W: \(W)")
                    print("    x: \(x)")
                    print("    CPU: \(cpuResult)")
                    print("    GPU: \(gpuResult)")
                }
            }
        }
        
        print("  Results: \(passed) passed, \(failed) failed")
        print("  Max error: \(maxError)")
        
        return failed == 0
    }
    
    // =========================================================================
    // PROPERTY TESTS
    // =========================================================================
    
    func testLinearity() -> Bool {
        print("\n=== 8×8 Linearity Property Test ===")
        
        var passed = 0
        var failed = 0
        
        for _ in 0..<1000 {
            // Random weights
            var W = [UInt16](repeating: 0, count: 8)
            for row in 0..<8 {
                var packed: UInt16 = 0
                for col in 0..<8 {
                    let trit = Int.random(in: -1...1)
                    packed |= UInt16(encodeTrit(trit)) << (col * 2)
                }
                W[row] = packed
            }
            
            // Random inputs and scalars
            let x = (0..<8).map { _ in Float.random(in: -10...10) }
            let y = (0..<8).map { _ in Float.random(in: -10...10) }
            let a = Float.random(in: -5...5)
            let b = Float.random(in: -5...5)
            
            // Compute W @ (a*x + b*y)
            let combined = zip(x, y).map { a * $0 + b * $1 }
            guard let lhs = gpuMatvec8x8(W: W, x: combined) else { continue }
            
            // Compute a * (W @ x) + b * (W @ y)
            guard let wx = gpuMatvec8x8(W: W, x: x),
                  let wy = gpuMatvec8x8(W: W, x: y) else { continue }
            let rhs = zip(wx, wy).map { a * $0 + b * $1 }
            
            // Compare
            let match = zip(lhs, rhs).allSatisfy { abs($0 - $1) < 1e-3 }
            if match {
                passed += 1
            } else {
                failed += 1
            }
        }
        
        print("  Linearity: \(passed) passed, \(failed) failed")
        return failed == 0
    }
    
    // =========================================================================
    // RUN ALL
    // =========================================================================
    
    func runAll() -> Bool {
        var allPassed = true
        
        if !testBoundaries() { allPassed = false }
        if !testRandomSampling() { allPassed = false }
        if !testLinearity() { allPassed = false }
        
        print("\n" + String(repeating: "=", count: 50))
        if allPassed {
            print("  8×8 TESTS: ALL PASSED")
        } else {
            print("  8×8 TESTS: SOME FAILED")
        }
        print(String(repeating: "=", count: 50))
        
        return allPassed
    }
}

enum TestError: Error {
    case noDevice
    case noCommandQueue
}
