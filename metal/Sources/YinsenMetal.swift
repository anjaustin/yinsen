/*
 * YINSEN Metal - Swift API
 *
 * High-level Swift bindings for Yinsen Metal kernels.
 * Provides safe, ergonomic access to ternary compute operations.
 */

import Metal
import Foundation

// =============================================================================
// YINSEN METAL CONTEXT
// =============================================================================

public class YinsenMetal {
    public let device: MTLDevice
    public let commandQueue: MTLCommandQueue
    public let library: MTLLibrary
    
    // Cached pipeline states
    private var pipelines: [String: MTLComputePipelineState] = [:]
    
    public init() throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw YinsenMetalError.noDevice
        }
        self.device = device
        
        guard let queue = device.makeCommandQueue() else {
            throw YinsenMetalError.noCommandQueue
        }
        self.commandQueue = queue
        
        // Load kernel source from bundle or compile from string
        let kernelSource = YinsenMetal.loadKernelSource()
        self.library = try device.makeLibrary(source: kernelSource, options: nil)
    }
    
    private static func loadKernelSource() -> String {
        // Embedded kernel source for standalone compilation
        return """
        #include <metal_stdlib>
        using namespace metal;
        
        // Trit sign extraction
        inline int trit_sign(uint8_t encoding) {
            int lsb = encoding & 1;
            int msb = (encoding >> 1) & 1;
            return lsb * (1 - 2 * msb);
        }
        
        inline int trit_unpack(uint8_t packed, int pos) {
            uint8_t encoding = (packed >> (pos * 2)) & 0x3;
            return trit_sign(encoding);
        }
        
        inline float ternary_dot4(uint8_t packed, float4 x) {
            float sum = 0.0f;
            for (int i = 0; i < 4; i++) {
                int sign = trit_unpack(packed, i);
                sum += float(sign) * x[i];
            }
            return sum;
        }
        
        kernel void ternary_matvec(
            device const uint8_t* W [[buffer(0)]],
            device const float* x [[buffer(1)]],
            device float* y [[buffer(2)]],
            constant uint& M [[buffer(3)]],
            constant uint& N [[buffer(4)]],
            uint row [[thread_position_in_grid]]
        ) {
            if (row >= M) return;
            
            uint bytes_per_row = (N + 3) / 4;
            device const uint8_t* w_row = W + row * bytes_per_row;
            
            float sum = 0.0f;
            uint i = 0;
            
            for (; i + 4 <= N; i += 4) {
                float4 xi = float4(x[i], x[i+1], x[i+2], x[i+3]);
                sum += ternary_dot4(w_row[i/4], xi);
            }
            
            if (i < N) {
                uint8_t packed = w_row[i/4];
                for (uint j = 0; i + j < N; j++) {
                    int sign = trit_unpack(packed, int(j));
                    sum += float(sign) * x[i + j];
                }
            }
            
            y[row] = sum;
        }
        
        kernel void ternary_matvec_bias(
            device const uint8_t* W [[buffer(0)]],
            device const float* x [[buffer(1)]],
            device const float* bias [[buffer(2)]],
            device float* y [[buffer(3)]],
            constant uint& M [[buffer(4)]],
            constant uint& N [[buffer(5)]],
            uint row [[thread_position_in_grid]]
        ) {
            if (row >= M) return;
            
            uint bytes_per_row = (N + 3) / 4;
            device const uint8_t* w_row = W + row * bytes_per_row;
            
            float sum = 0.0f;
            uint i = 0;
            
            for (; i + 4 <= N; i += 4) {
                float4 xi = float4(x[i], x[i+1], x[i+2], x[i+3]);
                sum += ternary_dot4(w_row[i/4], xi);
            }
            
            if (i < N) {
                uint8_t packed = w_row[i/4];
                for (uint j = 0; i + j < N; j++) {
                    int sign = trit_unpack(packed, int(j));
                    sum += float(sign) * x[i + j];
                }
            }
            
            y[row] = sum + bias[row];
        }
        
        kernel void gelu(
            device const float* input [[buffer(0)]],
            device float* output [[buffer(1)]],
            constant uint& n [[buffer(2)]],
            uint idx [[thread_position_in_grid]]
        ) {
            if (idx >= n) return;
            float x = input[idx];
            float sigmoid_val = 1.0f / (1.0f + exp(-1.702f * x));
            output[idx] = x * sigmoid_val;
        }
        
        kernel void softmax_rows(
            device const float* input [[buffer(0)]],
            device float* output [[buffer(1)]],
            constant uint& num_rows [[buffer(2)]],
            constant uint& row_size [[buffer(3)]],
            threadgroup float* shared [[threadgroup(0)]],
            uint row [[threadgroup_position_in_grid]],
            uint tid [[thread_index_in_threadgroup]],
            uint tg_size [[threads_per_threadgroup]]
        ) {
            if (row >= num_rows) return;
            
            device const float* row_in = input + row * row_size;
            device float* row_out = output + row * row_size;
            
            float local_max = -INFINITY;
            for (uint i = tid; i < row_size; i += tg_size) {
                local_max = max(local_max, row_in[i]);
            }
            shared[tid] = local_max;
            threadgroup_barrier(mem_flags::mem_threadgroup);
            
            for (uint stride = tg_size / 2; stride > 0; stride /= 2) {
                if (tid < stride) {
                    shared[tid] = max(shared[tid], shared[tid + stride]);
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }
            float max_val = shared[0];
            threadgroup_barrier(mem_flags::mem_threadgroup);
            
            float local_sum = 0.0f;
            for (uint i = tid; i < row_size; i += tg_size) {
                float exp_val = exp(row_in[i] - max_val);
                row_out[i] = exp_val;
                local_sum += exp_val;
            }
            shared[tid] = local_sum;
            threadgroup_barrier(mem_flags::mem_threadgroup);
            
            for (uint stride = tg_size / 2; stride > 0; stride /= 2) {
                if (tid < stride) {
                    shared[tid] += shared[tid + stride];
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }
            float sum_val = shared[0];
            threadgroup_barrier(mem_flags::mem_threadgroup);
            
            for (uint i = tid; i < row_size; i += tg_size) {
                row_out[i] /= sum_val;
            }
        }
        """
    }
    
    // =========================================================================
    // PIPELINE MANAGEMENT
    // =========================================================================
    
    public func getPipeline(name: String) throws -> MTLComputePipelineState {
        if let cached = pipelines[name] {
            return cached
        }
        
        guard let function = library.makeFunction(name: name) else {
            throw YinsenMetalError.kernelNotFound(name)
        }
        
        let pipeline = try device.makeComputePipelineState(function: function)
        pipelines[name] = pipeline
        return pipeline
    }
    
    // =========================================================================
    // HIGH-LEVEL OPERATIONS
    // =========================================================================
    
    /// Ternary matrix-vector multiply: y = W @ x
    public func ternaryMatvec(
        weights: MTLBuffer,  // Packed ternary [M x ceil(N/4)]
        input: MTLBuffer,    // Float [N]
        output: MTLBuffer,   // Float [M]
        M: Int,
        N: Int
    ) throws {
        let pipeline = try getPipeline(name: "ternary_matvec")
        
        guard let commandBuffer = commandQueue.makeCommandBuffer(),
              let encoder = commandBuffer.makeComputeCommandEncoder() else {
            throw YinsenMetalError.commandBufferFailed
        }
        
        var m = UInt32(M)
        var n = UInt32(N)
        
        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(weights, offset: 0, index: 0)
        encoder.setBuffer(input, offset: 0, index: 1)
        encoder.setBuffer(output, offset: 0, index: 2)
        encoder.setBytes(&m, length: MemoryLayout<UInt32>.size, index: 3)
        encoder.setBytes(&n, length: MemoryLayout<UInt32>.size, index: 4)
        
        let gridSize = MTLSize(width: M, height: 1, depth: 1)
        let threadgroupSize = MTLSize(width: min(M, pipeline.maxTotalThreadsPerThreadgroup), height: 1, depth: 1)
        encoder.dispatchThreads(gridSize, threadsPerThreadgroup: threadgroupSize)
        
        encoder.endEncoding()
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
    }
    
    /// Ternary matrix-vector multiply with bias: y = W @ x + b
    public func ternaryMatvecBias(
        weights: MTLBuffer,
        input: MTLBuffer,
        bias: MTLBuffer,
        output: MTLBuffer,
        M: Int,
        N: Int
    ) throws {
        let pipeline = try getPipeline(name: "ternary_matvec_bias")
        
        guard let commandBuffer = commandQueue.makeCommandBuffer(),
              let encoder = commandBuffer.makeComputeCommandEncoder() else {
            throw YinsenMetalError.commandBufferFailed
        }
        
        var m = UInt32(M)
        var n = UInt32(N)
        
        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(weights, offset: 0, index: 0)
        encoder.setBuffer(input, offset: 0, index: 1)
        encoder.setBuffer(bias, offset: 0, index: 2)
        encoder.setBuffer(output, offset: 0, index: 3)
        encoder.setBytes(&m, length: MemoryLayout<UInt32>.size, index: 4)
        encoder.setBytes(&n, length: MemoryLayout<UInt32>.size, index: 5)
        
        let gridSize = MTLSize(width: M, height: 1, depth: 1)
        let threadgroupSize = MTLSize(width: min(M, pipeline.maxTotalThreadsPerThreadgroup), height: 1, depth: 1)
        encoder.dispatchThreads(gridSize, threadsPerThreadgroup: threadgroupSize)
        
        encoder.endEncoding()
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
    }
    
    /// GELU activation
    public func gelu(input: MTLBuffer, output: MTLBuffer, n: Int) throws {
        let pipeline = try getPipeline(name: "gelu")
        
        guard let commandBuffer = commandQueue.makeCommandBuffer(),
              let encoder = commandBuffer.makeComputeCommandEncoder() else {
            throw YinsenMetalError.commandBufferFailed
        }
        
        var count = UInt32(n)
        
        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(input, offset: 0, index: 0)
        encoder.setBuffer(output, offset: 0, index: 1)
        encoder.setBytes(&count, length: MemoryLayout<UInt32>.size, index: 2)
        
        let gridSize = MTLSize(width: n, height: 1, depth: 1)
        let threadgroupSize = MTLSize(width: min(n, pipeline.maxTotalThreadsPerThreadgroup), height: 1, depth: 1)
        encoder.dispatchThreads(gridSize, threadsPerThreadgroup: threadgroupSize)
        
        encoder.endEncoding()
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
    }
    
    // =========================================================================
    // BUFFER HELPERS
    // =========================================================================
    
    public func makeBuffer<T>(from array: [T]) -> MTLBuffer? {
        return device.makeBuffer(
            bytes: array,
            length: array.count * MemoryLayout<T>.size,
            options: .storageModeShared
        )
    }
    
    public func makeBuffer(length: Int) -> MTLBuffer? {
        return device.makeBuffer(length: length, options: .storageModeShared)
    }
}

// =============================================================================
// ERROR TYPES
// =============================================================================

public enum YinsenMetalError: Error {
    case noDevice
    case noCommandQueue
    case kernelNotFound(String)
    case commandBufferFailed
    case bufferCreationFailed
}
