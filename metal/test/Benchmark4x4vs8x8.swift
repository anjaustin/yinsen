/*
 * YINSEN Metal - 4×4 vs 8×8 Performance Benchmark
 *
 * Empirically measures the performance difference between
 * 4×4 and 8×8 ternary matvec kernels on Apple Silicon.
 *
 * Hypothesis: 8×8 should be faster due to:
 * - 128-bit aligned weight loads
 * - Better SIMD utilization
 * - Reduced kernel launch overhead per element
 */

import Metal
import Foundation

class Benchmark4x4vs8x8 {
    let device: MTLDevice
    let commandQueue: MTLCommandQueue
    let library: MTLLibrary
    
    var pipeline4x4: MTLComputePipelineState?
    var pipeline8x8: MTLComputePipelineState?
    var pipeline8x8Single: MTLComputePipelineState?
    var pipelineTiled8x8: MTLComputePipelineState?
    var pipelineTiledCooperative: MTLComputePipelineState?
    
    init() throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw BenchmarkError.noDevice
        }
        self.device = device
        
        guard let queue = device.makeCommandQueue() else {
            throw BenchmarkError.noCommandQueue
        }
        self.commandQueue = queue
        
        // Load all kernel files
        let kernelPath = URL(fileURLWithPath: #file)
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .appendingPathComponent("kernels")
        
        let source4x4 = try String(contentsOf: kernelPath.appendingPathComponent("ternary_core.metal"))
        let source8x8 = try String(contentsOf: kernelPath.appendingPathComponent("ternary_8x8.metal"))
        let sourceTiled = try String(contentsOf: kernelPath.appendingPathComponent("ternary_matvec_tiled.metal"))
        
        let lib4x4 = try device.makeLibrary(source: source4x4, options: nil)
        let lib8x8 = try device.makeLibrary(source: source8x8, options: nil)
        let libTiled = try device.makeLibrary(source: sourceTiled, options: nil)
        self.library = lib8x8  // Use 8x8 as primary
        
        // Create pipelines
        if let fn = lib4x4.makeFunction(name: "ternary_matvec") {
            pipeline4x4 = try device.makeComputePipelineState(function: fn)
        }
        if let fn = lib8x8.makeFunction(name: "ternary_matvec_8x8") {
            pipeline8x8 = try device.makeComputePipelineState(function: fn)
        }
        if let fn = lib8x8.makeFunction(name: "ternary_matvec_8x8_single") {
            pipeline8x8Single = try device.makeComputePipelineState(function: fn)
        }
        if let fn = lib8x8.makeFunction(name: "ternary_matvec_tiled_8x8") {
            pipelineTiled8x8 = try device.makeComputePipelineState(function: fn)
        }
        if let fn = libTiled.makeFunction(name: "ternary_matvec_tiled") {
            pipelineTiledCooperative = try device.makeComputePipelineState(function: fn)
        }
        
        print("Benchmark initialized on \(device.name)")
    }
    
    // =========================================================================
    // BENCHMARK: Single Matvec Latency
    // =========================================================================
    
    func benchmarkSingleMatvec(iterations: Int = 10000) {
        print("\n=== Single Matvec Latency (\(iterations) iterations) ===")
        
        // 4×4 benchmark
        if let pipeline = pipeline4x4 {
            let W = device.makeBuffer(length: 4, options: .storageModeShared)!  // 4 bytes for 4×4
            let x = device.makeBuffer(length: 16, options: .storageModeShared)! // 4 floats
            let y = device.makeBuffer(length: 16, options: .storageModeShared)! // 4 floats
            
            var m: UInt32 = 4
            var n: UInt32 = 4
            
            let start = CFAbsoluteTimeGetCurrent()
            for _ in 0..<iterations {
                guard let cmdBuffer = commandQueue.makeCommandBuffer(),
                      let encoder = cmdBuffer.makeComputeCommandEncoder() else { continue }
                
                encoder.setComputePipelineState(pipeline)
                encoder.setBuffer(W, offset: 0, index: 0)
                encoder.setBuffer(x, offset: 0, index: 1)
                encoder.setBuffer(y, offset: 0, index: 2)
                encoder.setBytes(&m, length: 4, index: 3)
                encoder.setBytes(&n, length: 4, index: 4)
                
                encoder.dispatchThreads(MTLSize(width: 4, height: 1, depth: 1),
                                        threadsPerThreadgroup: MTLSize(width: 4, height: 1, depth: 1))
                encoder.endEncoding()
                cmdBuffer.commit()
                cmdBuffer.waitUntilCompleted()
            }
            let elapsed = CFAbsoluteTimeGetCurrent() - start
            let avgLatency = (elapsed / Double(iterations)) * 1_000_000  // microseconds
            
            print("  4×4 matvec: \(String(format: "%.2f", avgLatency)) µs/op")
        }
        
        // 8×8 benchmark (parallel threads)
        if let pipeline = pipeline8x8 {
            let W = device.makeBuffer(length: 16, options: .storageModeShared)! // 16 bytes for 8×8
            let x = device.makeBuffer(length: 32, options: .storageModeShared)! // 8 floats
            let y = device.makeBuffer(length: 32, options: .storageModeShared)! // 8 floats
            
            let start = CFAbsoluteTimeGetCurrent()
            for _ in 0..<iterations {
                guard let cmdBuffer = commandQueue.makeCommandBuffer(),
                      let encoder = cmdBuffer.makeComputeCommandEncoder() else { continue }
                
                encoder.setComputePipelineState(pipeline)
                encoder.setBuffer(W, offset: 0, index: 0)
                encoder.setBuffer(x, offset: 0, index: 1)
                encoder.setBuffer(y, offset: 0, index: 2)
                
                encoder.dispatchThreads(MTLSize(width: 8, height: 1, depth: 1),
                                        threadsPerThreadgroup: MTLSize(width: 8, height: 1, depth: 1))
                encoder.endEncoding()
                cmdBuffer.commit()
                cmdBuffer.waitUntilCompleted()
            }
            let elapsed = CFAbsoluteTimeGetCurrent() - start
            let avgLatency = (elapsed / Double(iterations)) * 1_000_000
            
            print("  8×8 matvec (8 threads): \(String(format: "%.2f", avgLatency)) µs/op")
        }
        
        // 8×8 benchmark (single thread)
        if let pipeline = pipeline8x8Single {
            let W = device.makeBuffer(length: 16, options: .storageModeShared)!
            let x = device.makeBuffer(length: 32, options: .storageModeShared)!
            let y = device.makeBuffer(length: 32, options: .storageModeShared)!
            
            let start = CFAbsoluteTimeGetCurrent()
            for _ in 0..<iterations {
                guard let cmdBuffer = commandQueue.makeCommandBuffer(),
                      let encoder = cmdBuffer.makeComputeCommandEncoder() else { continue }
                
                encoder.setComputePipelineState(pipeline)
                encoder.setBuffer(W, offset: 0, index: 0)
                encoder.setBuffer(x, offset: 0, index: 1)
                encoder.setBuffer(y, offset: 0, index: 2)
                
                encoder.dispatchThreads(MTLSize(width: 1, height: 1, depth: 1),
                                        threadsPerThreadgroup: MTLSize(width: 1, height: 1, depth: 1))
                encoder.endEncoding()
                cmdBuffer.commit()
                cmdBuffer.waitUntilCompleted()
            }
            let elapsed = CFAbsoluteTimeGetCurrent() - start
            let avgLatency = (elapsed / Double(iterations)) * 1_000_000
            
            print("  8×8 matvec (1 thread): \(String(format: "%.2f", avgLatency)) µs/op")
        }
    }
    
    // =========================================================================
    // BENCHMARK: Large Matrix Throughput
    // =========================================================================
    
    func benchmarkLargeMatrix(M: Int, N: Int, iterations: Int = 100) {
        print("\n=== Large Matrix Throughput (\(M)×\(N), \(iterations) iterations) ===")
        
        // Using 4×4 tiled (via the base kernel)
        if let pipeline = pipeline4x4 {
            let bytesPerRow4 = (N + 3) / 4  // ceil(N/4) bytes per row
            let W = device.makeBuffer(length: M * bytesPerRow4, options: .storageModeShared)!
            let x = device.makeBuffer(length: N * 4, options: .storageModeShared)!
            let y = device.makeBuffer(length: M * 4, options: .storageModeShared)!
            
            var m = UInt32(M)
            var n = UInt32(N)
            
            let start = CFAbsoluteTimeGetCurrent()
            for _ in 0..<iterations {
                guard let cmdBuffer = commandQueue.makeCommandBuffer(),
                      let encoder = cmdBuffer.makeComputeCommandEncoder() else { continue }
                
                encoder.setComputePipelineState(pipeline)
                encoder.setBuffer(W, offset: 0, index: 0)
                encoder.setBuffer(x, offset: 0, index: 1)
                encoder.setBuffer(y, offset: 0, index: 2)
                encoder.setBytes(&m, length: 4, index: 3)
                encoder.setBytes(&n, length: 4, index: 4)
                
                let threadgroupWidth = min(M, pipeline.maxTotalThreadsPerThreadgroup)
                encoder.dispatchThreads(MTLSize(width: M, height: 1, depth: 1),
                                        threadsPerThreadgroup: MTLSize(width: threadgroupWidth, height: 1, depth: 1))
                encoder.endEncoding()
                cmdBuffer.commit()
                cmdBuffer.waitUntilCompleted()
            }
            let elapsed = CFAbsoluteTimeGetCurrent() - start
            let avgLatency = (elapsed / Double(iterations)) * 1000  // milliseconds
            let elementsPerSec = Double(M * N * iterations) / elapsed
            let gops = elementsPerSec / 1e9
            
            print("  4×4 tiled: \(String(format: "%.3f", avgLatency)) ms/op, \(String(format: "%.2f", gops)) Gop/s")
        }
        
        // Using 8×8 tiled (one thread per row, sequential K)
        if let pipeline = pipelineTiled8x8 {
            let blocksPerRow = (N + 7) / 8
            let W = device.makeBuffer(length: M * blocksPerRow * 2, options: .storageModeShared)! // 2 bytes per block
            let x = device.makeBuffer(length: N * 4, options: .storageModeShared)!
            let y = device.makeBuffer(length: M * 4, options: .storageModeShared)!
            
            var m = UInt32(M)
            var n = UInt32(N)
            
            let start = CFAbsoluteTimeGetCurrent()
            for _ in 0..<iterations {
                guard let cmdBuffer = commandQueue.makeCommandBuffer(),
                      let encoder = cmdBuffer.makeComputeCommandEncoder() else { continue }
                
                encoder.setComputePipelineState(pipeline)
                encoder.setBuffer(W, offset: 0, index: 0)
                encoder.setBuffer(x, offset: 0, index: 1)
                encoder.setBuffer(y, offset: 0, index: 2)
                encoder.setBytes(&m, length: 4, index: 3)
                encoder.setBytes(&n, length: 4, index: 4)
                
                let threadgroupWidth = min(M, pipeline.maxTotalThreadsPerThreadgroup)
                encoder.dispatchThreads(MTLSize(width: M, height: 1, depth: 1),
                                        threadsPerThreadgroup: MTLSize(width: threadgroupWidth, height: 1, depth: 1))
                encoder.endEncoding()
                cmdBuffer.commit()
                cmdBuffer.waitUntilCompleted()
            }
            let elapsed = CFAbsoluteTimeGetCurrent() - start
            let avgLatency = (elapsed / Double(iterations)) * 1000
            let elementsPerSec = Double(M * N * iterations) / elapsed
            let gops = elementsPerSec / 1e9
            
            print("  8×8 tiled: \(String(format: "%.3f", avgLatency)) ms/op, \(String(format: "%.2f", gops)) Gop/s")
        }
        
        // Using threadgroup-cooperative tiled kernel (shared mem + simd_sum)
        if let pipeline = pipelineTiledCooperative {
            let bytesPerRow = (N + 3) / 4
            let W = device.makeBuffer(length: M * bytesPerRow, options: .storageModeShared)!
            let x = device.makeBuffer(length: N * 4, options: .storageModeShared)!
            let y = device.makeBuffer(length: M * 4, options: .storageModeShared)!
            
            var m = UInt32(M)
            var n = UInt32(N)
            
            let threadsPerGroup = min(256, pipeline.maxTotalThreadsPerThreadgroup)
            let numSimdgroups = (threadsPerGroup + 31) / 32
            let sharedMemSize = max(N, numSimdgroups) * MemoryLayout<Float>.size
            
            let start = CFAbsoluteTimeGetCurrent()
            for _ in 0..<iterations {
                guard let cmdBuffer = commandQueue.makeCommandBuffer(),
                      let encoder = cmdBuffer.makeComputeCommandEncoder() else { continue }
                
                encoder.setComputePipelineState(pipeline)
                encoder.setBuffer(W, offset: 0, index: 0)
                encoder.setBuffer(x, offset: 0, index: 1)
                encoder.setBuffer(y, offset: 0, index: 2)
                encoder.setBytes(&m, length: 4, index: 3)
                encoder.setBytes(&n, length: 4, index: 4)
                encoder.setThreadgroupMemoryLength(sharedMemSize, index: 0)
                
                encoder.dispatchThreadgroups(
                    MTLSize(width: M, height: 1, depth: 1),
                    threadsPerThreadgroup: MTLSize(width: threadsPerGroup, height: 1, depth: 1))
                encoder.endEncoding()
                cmdBuffer.commit()
                cmdBuffer.waitUntilCompleted()
            }
            let elapsed = CFAbsoluteTimeGetCurrent() - start
            let avgLatency = (elapsed / Double(iterations)) * 1000
            let elementsPerSec = Double(M * N * iterations) / elapsed
            let gops = elementsPerSec / 1e9
            
            print("  Cooperative tiled: \(String(format: "%.3f", avgLatency)) ms/op, \(String(format: "%.2f", gops)) Gop/s")
        }
    }
    
    // =========================================================================
    // BENCHMARK: Memory Bandwidth Utilization
    // =========================================================================
    
    func benchmarkMemoryBandwidth() {
        print("\n=== Memory Bandwidth Analysis ===")
        
        // Calculate theoretical memory requirements
        let sizes = [(64, 64), (256, 256), (1024, 1024), (4096, 4096)]
        
        for (M, N) in sizes {
            // 4×4 tiling
            let bytes4x4_W = M * ((N + 3) / 4)     // Packed weights
            let bytes4x4_x = N * 4                   // Float input
            let bytes4x4_y = M * 4                   // Float output
            let total4x4 = bytes4x4_W + bytes4x4_x + bytes4x4_y
            
            // 8×8 tiling
            let bytes8x8_W = M * ((N + 7) / 8) * 2  // Packed weights (2 bytes per 8 trits)
            let bytes8x8_x = N * 4
            let bytes8x8_y = M * 4
            let total8x8 = bytes8x8_W + bytes8x8_x + bytes8x8_y
            
            print("  \(M)×\(N): 4×4 = \(total4x4) bytes, 8×8 = \(total8x8) bytes")
        }
        
        print("\n  Note: 8×8 has slightly more weight bytes due to padding")
        print("  but fewer memory transactions due to aligned 128-bit loads")
    }
    
    // =========================================================================
    // BENCHMARK: Batch Processing
    // =========================================================================
    
    func benchmarkBatchProcessing(batchSize: Int, iterations: Int = 100) {
        print("\n=== Batch Processing (batch=\(batchSize), \(iterations) iterations) ===")
        
        // Simulate attention heads or parallel inference
        
        // 4×4: batchSize independent matvecs
        if let pipeline = pipeline4x4 {
            let W = device.makeBuffer(length: batchSize * 4, options: .storageModeShared)!
            let x = device.makeBuffer(length: batchSize * 16, options: .storageModeShared)!
            let y = device.makeBuffer(length: batchSize * 16, options: .storageModeShared)!
            
            var m: UInt32 = 4
            var n: UInt32 = 4
            
            let start = CFAbsoluteTimeGetCurrent()
            for _ in 0..<iterations {
                guard let cmdBuffer = commandQueue.makeCommandBuffer(),
                      let encoder = cmdBuffer.makeComputeCommandEncoder() else { continue }
                
                // Launch all batches
                for b in 0..<batchSize {
                    encoder.setComputePipelineState(pipeline)
                    encoder.setBuffer(W, offset: b * 4, index: 0)
                    encoder.setBuffer(x, offset: b * 16, index: 1)
                    encoder.setBuffer(y, offset: b * 16, index: 2)
                    encoder.setBytes(&m, length: 4, index: 3)
                    encoder.setBytes(&n, length: 4, index: 4)
                    
                    encoder.dispatchThreads(MTLSize(width: 4, height: 1, depth: 1),
                                            threadsPerThreadgroup: MTLSize(width: 4, height: 1, depth: 1))
                }
                encoder.endEncoding()
                cmdBuffer.commit()
                cmdBuffer.waitUntilCompleted()
            }
            let elapsed = CFAbsoluteTimeGetCurrent() - start
            let avgLatency = (elapsed / Double(iterations)) * 1000
            
            print("  4×4 × \(batchSize): \(String(format: "%.3f", avgLatency)) ms/batch")
        }
        
        // 8×8: batchSize independent matvecs (more efficient)
        if let pipeline = pipeline8x8 {
            let W = device.makeBuffer(length: batchSize * 16, options: .storageModeShared)!
            let x = device.makeBuffer(length: batchSize * 32, options: .storageModeShared)!
            let y = device.makeBuffer(length: batchSize * 32, options: .storageModeShared)!
            
            let start = CFAbsoluteTimeGetCurrent()
            for _ in 0..<iterations {
                guard let cmdBuffer = commandQueue.makeCommandBuffer(),
                      let encoder = cmdBuffer.makeComputeCommandEncoder() else { continue }
                
                for b in 0..<batchSize {
                    encoder.setComputePipelineState(pipeline)
                    encoder.setBuffer(W, offset: b * 16, index: 0)
                    encoder.setBuffer(x, offset: b * 32, index: 1)
                    encoder.setBuffer(y, offset: b * 32, index: 2)
                    
                    encoder.dispatchThreads(MTLSize(width: 8, height: 1, depth: 1),
                                            threadsPerThreadgroup: MTLSize(width: 8, height: 1, depth: 1))
                }
                encoder.endEncoding()
                cmdBuffer.commit()
                cmdBuffer.waitUntilCompleted()
            }
            let elapsed = CFAbsoluteTimeGetCurrent() - start
            let avgLatency = (elapsed / Double(iterations)) * 1000
            
            print("  8×8 × \(batchSize): \(String(format: "%.3f", avgLatency)) ms/batch")
        }
    }
    
    // =========================================================================
    // RUN ALL
    // =========================================================================
    
    func runAll() {
        print("\n" + String(repeating: "=", count: 60))
        print("  YINSEN 4×4 vs 8×8 BENCHMARK - \(device.name)")
        print(String(repeating: "=", count: 60))
        
        benchmarkSingleMatvec()
        benchmarkLargeMatrix(M: 512, N: 512)
        benchmarkLargeMatrix(M: 1024, N: 1024)
        benchmarkLargeMatrix(M: 4096, N: 4096)
        benchmarkMemoryBandwidth()
        benchmarkBatchProcessing(batchSize: 8)
        benchmarkBatchProcessing(batchSize: 32)
        
        print("\n" + String(repeating: "=", count: 60))
        print("  BENCHMARK COMPLETE")
        print(String(repeating: "=", count: 60))
    }
}

enum BenchmarkError: Error {
    case noDevice
    case noCommandQueue
}
