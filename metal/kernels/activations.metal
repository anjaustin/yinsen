/*
 * YINSEN Metal - Activation Functions
 *
 * Standard neural network activations for transformer inference.
 * Matches include/onnx_shapes.h semantics.
 */

#include <metal_stdlib>
using namespace metal;

// =============================================================================
// ELEMENT-WISE ACTIVATIONS
// =============================================================================

kernel void relu(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant uint& n [[buffer(2)]],
    uint idx [[thread_position_in_grid]]
) {
    if (idx >= n) return;
    output[idx] = max(0.0f, input[idx]);
}

kernel void gelu(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant uint& n [[buffer(2)]],
    uint idx [[thread_position_in_grid]]
) {
    if (idx >= n) return;
    
    // GELU approximation: x * sigmoid(1.702 * x)
    float x = input[idx];
    float sigmoid_val = 1.0f / (1.0f + exp(-1.702f * x));
    output[idx] = x * sigmoid_val;
}

kernel void silu(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant uint& n [[buffer(2)]],
    uint idx [[thread_position_in_grid]]
) {
    if (idx >= n) return;
    
    // SiLU (Swish): x * sigmoid(x)
    float x = input[idx];
    float sigmoid_val = 1.0f / (1.0f + exp(-x));
    output[idx] = x * sigmoid_val;
}

kernel void sigmoid(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant uint& n [[buffer(2)]],
    uint idx [[thread_position_in_grid]]
) {
    if (idx >= n) return;
    output[idx] = 1.0f / (1.0f + exp(-input[idx]));
}

kernel void tanh_activation(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant uint& n [[buffer(2)]],
    uint idx [[thread_position_in_grid]]
) {
    if (idx >= n) return;
    output[idx] = tanh(input[idx]);
}

// =============================================================================
// SOFTMAX
// =============================================================================

// Two-pass softmax for numerical stability
// Pass 1: Find max
// Pass 2: Compute exp and sum
// Pass 3: Normalize

// For small vectors (fits in threadgroup), single-kernel softmax
kernel void softmax_small(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant uint& n [[buffer(2)]],
    threadgroup float* shared [[threadgroup(0)]],
    uint tid [[thread_index_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]]
) {
    // This kernel assumes n <= tg_size (typically 256 or 1024)
    // For larger vectors, use multi-pass approach
    
    // Step 1: Load and find local max
    float val = (tid < n) ? input[tid] : -INFINITY;
    shared[tid] = val;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // Parallel reduction for max
    for (uint stride = tg_size / 2; stride > 0; stride /= 2) {
        if (tid < stride && tid + stride < n) {
            shared[tid] = max(shared[tid], shared[tid + stride]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    
    float max_val = shared[0];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // Step 2: Compute exp(x - max) and sum
    float exp_val = (tid < n) ? exp(val - max_val) : 0.0f;
    shared[tid] = exp_val;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // Parallel reduction for sum
    for (uint stride = tg_size / 2; stride > 0; stride /= 2) {
        if (tid < stride) {
            shared[tid] += shared[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    
    float sum_val = shared[0];
    
    // Step 3: Normalize
    if (tid < n) {
        output[tid] = exp_val / sum_val;
    }
}

// Row-wise softmax for batch processing
// Each threadgroup handles one row
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
    
    // Find max (parallel reduction)
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
    
    // Compute exp and sum
    float local_sum = 0.0f;
    for (uint i = tid; i < row_size; i += tg_size) {
        float exp_val = exp(row_in[i] - max_val);
        row_out[i] = exp_val;  // Temporarily store exp values
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
    
    // Normalize
    for (uint i = tid; i < row_size; i += tg_size) {
        row_out[i] /= sum_val;
    }
}

// =============================================================================
// FUSED OPERATIONS
// =============================================================================

// Fused bias + activation (common pattern)
kernel void bias_gelu(
    device const float* input [[buffer(0)]],
    device const float* bias [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant uint& n [[buffer(3)]],
    constant uint& bias_size [[buffer(4)]],  // For broadcasting
    uint idx [[thread_position_in_grid]]
) {
    if (idx >= n) return;
    
    float x = input[idx] + bias[idx % bias_size];
    float sigmoid_val = 1.0f / (1.0f + exp(-1.702f * x));
    output[idx] = x * sigmoid_val;
}

kernel void bias_silu(
    device const float* input [[buffer(0)]],
    device const float* bias [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant uint& n [[buffer(3)]],
    constant uint& bias_size [[buffer(4)]],
    uint idx [[thread_position_in_grid]]
) {
    if (idx >= n) return;
    
    float x = input[idx] + bias[idx % bias_size];
    float sigmoid_val = 1.0f / (1.0f + exp(-x));
    output[idx] = x * sigmoid_val;
}
