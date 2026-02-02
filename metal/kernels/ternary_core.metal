/*
 * YINSEN Metal - Ternary Core Operations
 *
 * Proven-correct ternary compute kernels for Apple Silicon.
 * Matches include/ternary.h semantics exactly.
 *
 * Canonical encoding (2 bits per trit) - matches all backends:
 *   00 = 0  (zero - skip)
 *   01 = +1 (add)
 *   10 = -1 (subtract)
 *   11 = reserved (treated as 0)
 *
 * Verification: All kernels proven via exhaustive testing.
 * See metal/test/ for verification harness.
 */

#include <metal_stdlib>
using namespace metal;

// =============================================================================
// TRIT OPERATIONS
// Guarded to allow concatenation with other .metal files.
// =============================================================================

#ifndef YINSEN_TRIT_PRIMITIVES
#define YINSEN_TRIT_PRIMITIVES

// Extract trit sign from 2-bit canonical encoding
// Returns: +1, -1, or 0
//
// Canonical encoding:
//   00 (0) -> 0
//   01 (1) -> +1
//   10 (2) -> -1
//   11 (3) -> 0 (reserved)
inline int trit_sign(uint8_t encoding) {
    return int(encoding == 1) - int(encoding == 2);
}

// Extract trit at position (0-3) from packed byte
inline int trit_unpack(uint8_t packed, int pos) {
    uint8_t encoding = (packed >> (pos * 2)) & 0x3;
    return trit_sign(encoding);
}

// =============================================================================
// TERNARY DOT PRODUCT - 4 ELEMENTS
// =============================================================================

// Compute dot product of 4 ternary weights with 4 float activations
// packed: 4 trits in 8 bits (2 bits each)
// x: 4 float activations
// Returns: sum of contributions
inline float ternary_dot4(uint8_t packed, float4 x) {
    float sum = 0.0f;
    
    // Trit 0
    int s0 = trit_unpack(packed, 0);
    sum += float(s0) * x.x;
    
    // Trit 1
    int s1 = trit_unpack(packed, 1);
    sum += float(s1) * x.y;
    
    // Trit 2
    int s2 = trit_unpack(packed, 2);
    sum += float(s2) * x.z;
    
    // Trit 3
    int s3 = trit_unpack(packed, 3);
    sum += float(s3) * x.w;
    
    return sum;
}

#endif // YINSEN_TRIT_PRIMITIVES

// Branchless version using FMA
inline float ternary_dot4_branchless(uint8_t packed, float4 x) {
    // Extract all signs at once
    int s0 = trit_unpack(packed, 0);
    int s1 = trit_unpack(packed, 1);
    int s2 = trit_unpack(packed, 2);
    int s3 = trit_unpack(packed, 3);
    
    // Convert to float4 for vectorized multiply-add
    float4 signs = float4(float(s0), float(s1), float(s2), float(s3));
    
    // Dot product via component multiply and sum
    float4 products = signs * x;
    return products.x + products.y + products.z + products.w;
}

// =============================================================================
// TERNARY MATRIX-VECTOR MULTIPLY
// =============================================================================

// y = W @ x
// W: packed ternary [M x N], row-major, 4 trits per byte
// x: input [N]
// y: output [M]
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
    
    // Process 4 elements at a time (one packed byte)
    for (; i + 4 <= N; i += 4) {
        float4 xi = float4(x[i], x[i+1], x[i+2], x[i+3]);
        sum += ternary_dot4(w_row[i/4], xi);
    }
    
    // Handle remainder (1-3 elements)
    if (i < N) {
        uint8_t packed = w_row[i/4];
        for (uint j = 0; i + j < N; j++) {
            int sign = trit_unpack(packed, j);
            sum += float(sign) * x[i + j];
        }
    }
    
    y[row] = sum;
}

// y = W @ x + bias
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
            int sign = trit_unpack(packed, j);
            sum += float(sign) * x[i + j];
        }
    }
    
    y[row] = sum + bias[row];
}

// =============================================================================
// EXHAUSTIVE VERIFICATION KERNEL
// =============================================================================

// Test all 81 weight configurations for a single 4-element dot product
// Used to prove correctness against CPU reference
kernel void verify_dot4_exhaustive(
    device const float4* test_inputs [[buffer(0)]],    // Test activation vectors
    device float* results [[buffer(1)]],               // Output: one result per config
    constant uint& num_inputs [[buffer(2)]],           // Number of test vectors
    uint config_idx [[thread_position_in_grid]]        // 0-80 for all 3^4 configs
) {
    if (config_idx >= 81) return;
    
    // Decode config_idx to 4 trits (-1, 0, +1)
    int t0 = (config_idx % 3) - 1;
    int t1 = ((config_idx / 3) % 3) - 1;
    int t2 = ((config_idx / 9) % 3) - 1;
    int t3 = ((config_idx / 27) % 3) - 1;
    
    // Encode to packed byte using canonical encoding
    auto encode_trit = [](int val) -> uint8_t {
        if (val > 0) return 0x1;  // 01 = +1
        if (val < 0) return 0x2;  // 10 = -1
        return 0x0;               // 00 = 0
    };
    
    uint8_t packed = (encode_trit(t0) << 0) |
                     (encode_trit(t1) << 2) |
                     (encode_trit(t2) << 4) |
                     (encode_trit(t3) << 6);
    
    // Test with first input vector (extend for multiple inputs if needed)
    float4 x = test_inputs[0];
    float result = ternary_dot4(packed, x);
    
    results[config_idx] = result;
}

// Verify 4x4 matvec for a batch of weight configurations
// Each thread handles one weight configuration (of 43M total)
kernel void verify_matvec4x4_batch(
    device const float4* test_input [[buffer(0)]],     // Single test vector [4]
    device float4* results [[buffer(1)]],              // Output [batch_size]
    constant uint& batch_offset [[buffer(2)]],         // Starting config index
    constant uint& batch_size [[buffer(3)]],           // Configs in this batch
    uint local_idx [[thread_position_in_grid]]
) {
    if (local_idx >= batch_size) return;
    
    uint config_idx = batch_offset + local_idx;
    if (config_idx >= 43046721) return;  // 3^16
    
    // Decode config to 16 trits (4x4 matrix)
    uint8_t packed[4];  // 4 bytes = 16 trits
    uint remaining = config_idx;
    
    for (int byte_idx = 0; byte_idx < 4; byte_idx++) {
        uint8_t byte_val = 0;
        for (int trit_idx = 0; trit_idx < 4; trit_idx++) {
            int trit_val = (remaining % 3) - 1;
            remaining /= 3;
            
            uint8_t encoded;
            if (trit_val > 0) encoded = 0x1;       // 01 = +1
            else if (trit_val < 0) encoded = 0x2;  // 10 = -1
            else encoded = 0x0;                     // 00 = 0
            
            byte_val |= (encoded << (trit_idx * 2));
        }
        packed[byte_idx] = byte_val;
    }
    
    // Compute 4x4 matvec manually
    float4 x = test_input[0];
    float4 y;
    
    y.x = ternary_dot4(packed[0], x);
    y.y = ternary_dot4(packed[1], x);
    y.z = ternary_dot4(packed[2], x);
    y.w = ternary_dot4(packed[3], x);
    
    results[local_idx] = y;
}
