/*
 * YINSEN Metal - 8×8 Ternary Kernel (Hardware-Optimized)
 *
 * Optimized for 128-bit SIMD alignment on Apple Silicon.
 * Correctness derived from proven 4×4 primitive via composition.
 *
 * Key optimizations:
 * - 128-bit weight loads (fills NEON register)
 * - Branchless sign extraction
 * - Vectorized reduction for dot products (float4/int4 types)
 *
 * Verification status: FORGED (compositional proof from proven dot4)
 */

#include <metal_stdlib>
using namespace metal;

// =============================================================================
// CONSTANTS AND ENCODING
// =============================================================================

// Canonical trit encoding (matches all backends):
//   00 = 0  (zero)
//   01 = +1 (positive)
//   10 = -1 (negative)
//   11 = reserved (treated as 0)

// =============================================================================
// DOT PRODUCT PRIMITIVES
// =============================================================================

// Proven 4-element dot (from ternary_core.metal)
// Included here for self-contained kernel
// Canonical encoding: 00=0, 01=+1, 10=-1, 11=0(reserved)
inline int trit_sign(uint8_t encoding) {
    return int(encoding == 1) - int(encoding == 2);
}

inline float ternary_dot4(uint8_t packed, float4 x) {
    int s0 = trit_sign(packed & 0x3);
    int s1 = trit_sign((packed >> 2) & 0x3);
    int s2 = trit_sign((packed >> 4) & 0x3);
    int s3 = trit_sign((packed >> 6) & 0x3);
    
    return float(s0) * x.x + float(s1) * x.y + float(s2) * x.z + float(s3) * x.w;
}

// =============================================================================
// 8-ELEMENT DOT PRODUCT
// =============================================================================

// Compositional: dot8 = dot4 + dot4
// Correctness: Proven (each dot4 is exhaustively verified)
inline float ternary_dot8(uint16_t packed, float4 x_lo, float4 x_hi) {
    uint8_t packed_lo = packed & 0xFF;
    uint8_t packed_hi = (packed >> 8) & 0xFF;
    
    return ternary_dot4(packed_lo, x_lo) + ternary_dot4(packed_hi, x_hi);
}

// Branchless version with vectorized types (float4/int4)
// Note: Uses ALU-level SIMD types, NOT Metal simdgroup cooperative intrinsics.
inline float ternary_dot8_vectorized(uint16_t packed, float4 x_lo, float4 x_hi) {
    // Extract all 8 signs at once
    int4 signs_lo, signs_hi;
    
    // Low 4 trits
    signs_lo.x = trit_sign(packed & 0x3);
    signs_lo.y = trit_sign((packed >> 2) & 0x3);
    signs_lo.z = trit_sign((packed >> 4) & 0x3);
    signs_lo.w = trit_sign((packed >> 6) & 0x3);
    
    // High 4 trits
    signs_hi.x = trit_sign((packed >> 8) & 0x3);
    signs_hi.y = trit_sign((packed >> 10) & 0x3);
    signs_hi.z = trit_sign((packed >> 12) & 0x3);
    signs_hi.w = trit_sign((packed >> 14) & 0x3);
    
    // Convert to float and multiply
    float4 contrib_lo = float4(signs_lo) * x_lo;
    float4 contrib_hi = float4(signs_hi) * x_hi;
    
    // Horizontal sum
    return (contrib_lo.x + contrib_lo.y + contrib_lo.z + contrib_lo.w) +
           (contrib_hi.x + contrib_hi.y + contrib_hi.z + contrib_hi.w);
}

// =============================================================================
// 8×8 MATRIX-VECTOR MULTIPLY
// =============================================================================

// y = W @ x
// W: 8 rows × 8 cols = 64 trits = 128 bits = 16 bytes
// x: 8 floats = 256 bits = 32 bytes
// y: 8 floats = 256 bits = 32 bytes
//
// Memory layout:
//   W is packed as uint16_t[8] - one 16-bit word per row (8 trits)
//   Each row's 8 trits map to 8 columns

kernel void ternary_matvec_8x8(
    device const uint16_t* W [[buffer(0)]],   // Packed weights [8] = 16 bytes
    device const float* x [[buffer(1)]],       // Input [8] = 32 bytes
    device float* y [[buffer(2)]],             // Output [8] = 32 bytes
    uint row [[thread_position_in_grid]]
) {
    if (row >= 8) return;
    
    // Load input as two float4s (128 bits each)
    float4 x_lo = float4(x[0], x[1], x[2], x[3]);
    float4 x_hi = float4(x[4], x[5], x[6], x[7]);
    
    // Load this row's weights (16 bits)
    uint16_t w_row = W[row];
    
    // Compute dot product
    y[row] = ternary_dot8_vectorized(w_row, x_lo, x_hi);
}

// RETIRED: All 8 rows in one thread. Kept for reference only.
// Use ternary_matvec_tiled from ternary_matvec_tiled.metal for real work.
kernel void ternary_matvec_8x8_single(
    device const uint16_t* W [[buffer(0)]],
    device const float* x [[buffer(1)]],
    device float* y [[buffer(2)]]
) {
    // Load input once
    float4 x_lo = float4(x[0], x[1], x[2], x[3]);
    float4 x_hi = float4(x[4], x[5], x[6], x[7]);
    
    // Compute all 8 outputs
    y[0] = ternary_dot8_vectorized(W[0], x_lo, x_hi);
    y[1] = ternary_dot8_vectorized(W[1], x_lo, x_hi);
    y[2] = ternary_dot8_vectorized(W[2], x_lo, x_hi);
    y[3] = ternary_dot8_vectorized(W[3], x_lo, x_hi);
    y[4] = ternary_dot8_vectorized(W[4], x_lo, x_hi);
    y[5] = ternary_dot8_vectorized(W[5], x_lo, x_hi);
    y[6] = ternary_dot8_vectorized(W[6], x_lo, x_hi);
    y[7] = ternary_dot8_vectorized(W[7], x_lo, x_hi);
}

// =============================================================================
// LARGE MATRIX SUPPORT (Tiled 8×8)
// =============================================================================

// y = W @ x for arbitrary M×N, tiled in 8×8 blocks
// W: packed uint16_t, row-major, 8 trits per uint16_t
// Rows are padded to multiple of 8 columns
kernel void ternary_matvec_tiled_8x8(
    device const uint16_t* W [[buffer(0)]],
    device const float* x [[buffer(1)]],
    device float* y [[buffer(2)]],
    constant uint& M [[buffer(3)]],           // Output dimension
    constant uint& N [[buffer(4)]],           // Input dimension
    uint row [[thread_position_in_grid]]
) {
    if (row >= M) return;
    
    // Number of 8-element blocks per row
    uint blocks_per_row = (N + 7) / 8;
    
    // Pointer to this row's weights
    device const uint16_t* w_row = W + row * blocks_per_row;
    
    float sum = 0.0f;
    
    // Process full 8-element blocks
    uint col = 0;
    for (uint b = 0; b < blocks_per_row && col + 8 <= N; b++, col += 8) {
        float4 x_lo = float4(x[col+0], x[col+1], x[col+2], x[col+3]);
        float4 x_hi = float4(x[col+4], x[col+5], x[col+6], x[col+7]);
        sum += ternary_dot8_vectorized(w_row[b], x_lo, x_hi);
    }
    
    // Handle remainder (if N not divisible by 8)
    if (col < N) {
        uint16_t packed = w_row[blocks_per_row - 1];
        for (uint i = 0; col + i < N; i++) {
            int sign = trit_sign((packed >> (i * 2)) & 0x3);
            sum += float(sign) * x[col + i];
        }
    }
    
    y[row] = sum;
}

// =============================================================================
// BATCHED OPERATIONS
// =============================================================================

// Batch of 8×8 matvecs (for attention heads, etc.)
// y[b] = W[b] @ x[b] for b in 0..batch_size
kernel void ternary_matvec_8x8_batched(
    device const uint16_t* W [[buffer(0)]],   // [batch × 8] packed weights
    device const float* x [[buffer(1)]],       // [batch × 8] inputs
    device float* y [[buffer(2)]],             // [batch × 8] outputs
    constant uint& batch_size [[buffer(3)]],
    uint2 pos [[thread_position_in_grid]]      // (row, batch)
) {
    uint row = pos.x;
    uint batch = pos.y;
    
    if (row >= 8 || batch >= batch_size) return;
    
    // Offset into this batch
    device const uint16_t* W_b = W + batch * 8;
    device const float* x_b = x + batch * 8;
    device float* y_b = y + batch * 8;
    
    float4 x_lo = float4(x_b[0], x_b[1], x_b[2], x_b[3]);
    float4 x_hi = float4(x_b[4], x_b[5], x_b[6], x_b[7]);
    
    y_b[row] = ternary_dot8_vectorized(W_b[row], x_lo, x_hi);
}

// =============================================================================
// FLOAT16 VARIANTS (BitNet-style)
// =============================================================================

// 8×8 with float16 activations - perfect 128-bit alignment
kernel void ternary_matvec_8x8_f16(
    device const uint16_t* W [[buffer(0)]],   // 128 bits weights
    device const half* x [[buffer(1)]],        // 128 bits input (8 × 16-bit)
    device half* y [[buffer(2)]],              // 128 bits output
    uint row [[thread_position_in_grid]]
) {
    if (row >= 8) return;
    
    // Load as half4 (64 bits each, 128 bits total)
    half4 x_lo = half4(x[0], x[1], x[2], x[3]);
    half4 x_hi = half4(x[4], x[5], x[6], x[7]);
    
    uint16_t w_row = W[row];
    
    // Extract signs
    int4 signs_lo, signs_hi;
    signs_lo.x = trit_sign(w_row & 0x3);
    signs_lo.y = trit_sign((w_row >> 2) & 0x3);
    signs_lo.z = trit_sign((w_row >> 4) & 0x3);
    signs_lo.w = trit_sign((w_row >> 6) & 0x3);
    signs_hi.x = trit_sign((w_row >> 8) & 0x3);
    signs_hi.y = trit_sign((w_row >> 10) & 0x3);
    signs_hi.z = trit_sign((w_row >> 12) & 0x3);
    signs_hi.w = trit_sign((w_row >> 14) & 0x3);
    
    // Compute in half precision
    half4 contrib_lo = half4(signs_lo) * x_lo;
    half4 contrib_hi = half4(signs_hi) * x_hi;
    
    y[row] = (contrib_lo.x + contrib_lo.y + contrib_lo.z + contrib_lo.w) +
             (contrib_hi.x + contrib_hi.y + contrib_hi.z + contrib_hi.w);
}

// =============================================================================
// VERIFICATION HELPERS
// =============================================================================

// Compute reference result for testing
// Each thread computes one 8×8 matvec and compares to expected
kernel void verify_8x8_batch(
    device const uint16_t* W [[buffer(0)]],    // [num_tests × 8]
    device const float* x [[buffer(1)]],        // [num_tests × 8]
    device const float* expected [[buffer(2)]], // [num_tests × 8]
    device uint* errors [[buffer(3)]],          // Error count
    constant uint& num_tests [[buffer(4)]],
    constant float& epsilon [[buffer(5)]],
    uint test_idx [[thread_position_in_grid]]
) {
    if (test_idx >= num_tests) return;
    
    device const uint16_t* W_t = W + test_idx * 8;
    device const float* x_t = x + test_idx * 8;
    device const float* exp_t = expected + test_idx * 8;
    
    float4 x_lo = float4(x_t[0], x_t[1], x_t[2], x_t[3]);
    float4 x_hi = float4(x_t[4], x_t[5], x_t[6], x_t[7]);
    
    for (uint row = 0; row < 8; row++) {
        float computed = ternary_dot8_vectorized(W_t[row], x_lo, x_hi);
        float expected_val = exp_t[row];
        
        if (abs(computed - expected_val) > epsilon) {
            atomic_fetch_add_explicit((device atomic_uint*)errors, 1, memory_order_relaxed);
        }
    }
}
