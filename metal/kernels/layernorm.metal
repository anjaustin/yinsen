/*
 * YINSEN Metal - Layer Normalization
 *
 * LayerNorm and RMSNorm for transformer blocks.
 * Matches include/onnx_shapes.h semantics.
 */

#include <metal_stdlib>
using namespace metal;

// =============================================================================
// LAYER NORMALIZATION
// =============================================================================

// LayerNorm: y = gamma * (x - mean) / sqrt(var + eps) + beta
// Single vector normalization
kernel void layernorm(
    device const float* input [[buffer(0)]],
    device const float* gamma [[buffer(1)]],
    device const float* beta [[buffer(2)]],
    device float* output [[buffer(3)]],
    constant uint& n [[buffer(4)]],
    constant float& eps [[buffer(5)]],
    threadgroup float* shared [[threadgroup(0)]],
    uint tid [[thread_index_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]]
) {
    // Pass 1: Compute mean
    float local_sum = 0.0f;
    for (uint i = tid; i < n; i += tg_size) {
        local_sum += input[i];
    }
    shared[tid] = local_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // Reduction for sum
    for (uint stride = tg_size / 2; stride > 0; stride /= 2) {
        if (tid < stride) {
            shared[tid] += shared[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float mean = shared[0] / float(n);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // Pass 2: Compute variance
    float local_var = 0.0f;
    for (uint i = tid; i < n; i += tg_size) {
        float diff = input[i] - mean;
        local_var += diff * diff;
    }
    shared[tid] = local_var;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    for (uint stride = tg_size / 2; stride > 0; stride /= 2) {
        if (tid < stride) {
            shared[tid] += shared[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float var = shared[0] / float(n);
    float inv_std = rsqrt(var + eps);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // Pass 3: Normalize and scale
    for (uint i = tid; i < n; i += tg_size) {
        float normalized = (input[i] - mean) * inv_std;
        output[i] = gamma[i] * normalized + beta[i];
    }
}

// Row-wise LayerNorm for batched processing
// Each threadgroup handles one row
kernel void layernorm_rows(
    device const float* input [[buffer(0)]],
    device const float* gamma [[buffer(1)]],
    device const float* beta [[buffer(2)]],
    device float* output [[buffer(3)]],
    constant uint& num_rows [[buffer(4)]],
    constant uint& row_size [[buffer(5)]],
    constant float& eps [[buffer(6)]],
    threadgroup float* shared [[threadgroup(0)]],
    uint row [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]]
) {
    if (row >= num_rows) return;
    
    device const float* row_in = input + row * row_size;
    device float* row_out = output + row * row_size;
    
    // Compute mean
    float local_sum = 0.0f;
    for (uint i = tid; i < row_size; i += tg_size) {
        local_sum += row_in[i];
    }
    shared[tid] = local_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    for (uint stride = tg_size / 2; stride > 0; stride /= 2) {
        if (tid < stride) {
            shared[tid] += shared[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float mean = shared[0] / float(row_size);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // Compute variance
    float local_var = 0.0f;
    for (uint i = tid; i < row_size; i += tg_size) {
        float diff = row_in[i] - mean;
        local_var += diff * diff;
    }
    shared[tid] = local_var;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    for (uint stride = tg_size / 2; stride > 0; stride /= 2) {
        if (tid < stride) {
            shared[tid] += shared[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float var = shared[0] / float(row_size);
    float inv_std = rsqrt(var + eps);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // Normalize
    for (uint i = tid; i < row_size; i += tg_size) {
        float normalized = (row_in[i] - mean) * inv_std;
        row_out[i] = gamma[i] * normalized + beta[i];
    }
}

// =============================================================================
// RMS NORMALIZATION (Llama-style)
// =============================================================================

// RMSNorm: y = x / sqrt(mean(x^2) + eps) * gamma
// No mean subtraction, no beta - simpler and often works just as well
kernel void rmsnorm(
    device const float* input [[buffer(0)]],
    device const float* gamma [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant uint& n [[buffer(3)]],
    constant float& eps [[buffer(4)]],
    threadgroup float* shared [[threadgroup(0)]],
    uint tid [[thread_index_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]]
) {
    // Compute mean of squares
    float local_sum_sq = 0.0f;
    for (uint i = tid; i < n; i += tg_size) {
        float val = input[i];
        local_sum_sq += val * val;
    }
    shared[tid] = local_sum_sq;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    for (uint stride = tg_size / 2; stride > 0; stride /= 2) {
        if (tid < stride) {
            shared[tid] += shared[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float rms = rsqrt(shared[0] / float(n) + eps);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // Normalize and scale
    for (uint i = tid; i < n; i += tg_size) {
        output[i] = input[i] * rms * gamma[i];
    }
}

// Row-wise RMSNorm
kernel void rmsnorm_rows(
    device const float* input [[buffer(0)]],
    device const float* gamma [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant uint& num_rows [[buffer(3)]],
    constant uint& row_size [[buffer(4)]],
    constant float& eps [[buffer(5)]],
    threadgroup float* shared [[threadgroup(0)]],
    uint row [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]]
) {
    if (row >= num_rows) return;
    
    device const float* row_in = input + row * row_size;
    device float* row_out = output + row * row_size;
    
    // Compute mean of squares
    float local_sum_sq = 0.0f;
    for (uint i = tid; i < row_size; i += tg_size) {
        float val = row_in[i];
        local_sum_sq += val * val;
    }
    shared[tid] = local_sum_sq;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    for (uint stride = tg_size / 2; stride > 0; stride /= 2) {
        if (tid < stride) {
            shared[tid] += shared[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float rms = rsqrt(shared[0] / float(row_size) + eps);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // Normalize and scale
    for (uint i = tid; i < row_size; i += tg_size) {
        row_out[i] = row_in[i] * rms * gamma[i];
    }
}
