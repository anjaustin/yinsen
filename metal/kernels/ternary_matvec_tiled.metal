/*
 * YINSEN Metal - Threadgroup-Cooperative Ternary Matvec
 *
 * The first "real" GPU kernel in Yinsen: threadgroup-cooperative K-reduction
 * with shared memory for the input vector and simd_sum for fast partial sums.
 *
 * Architecture:
 *   - One threadgroup per output row
 *   - Up to 256 threads per threadgroup, each handling ceil(N/tg_size) elements
 *   - Input vector loaded into threadgroup shared memory (reused by all threads)
 *   - Two-level reduction: simd_sum within 32-thread simdgroups, then
 *     shared-memory tree reduction across simdgroups
 *
 * Canonical encoding (2 bits per trit) - matches all backends:
 *   00 = 0  (zero - skip)
 *   01 = +1 (add)
 *   10 = -1 (subtract)
 *   11 = reserved (treated as 0)
 *
 * Weight packing: uint8_t, 4 trits per byte (same as ternary_core.metal).
 * Using byte-level packing for compatibility with the CPU packing format
 * and existing test infrastructure.
 */

#include <metal_stdlib>
using namespace metal;

// =============================================================================
// TRIT PRIMITIVES (shared with ternary_core.metal)
// Guarded to allow concatenation with other .metal files.
// =============================================================================

#ifndef YINSEN_TRIT_PRIMITIVES
#define YINSEN_TRIT_PRIMITIVES

inline int trit_sign(uint8_t encoding) {
    return int(encoding == 1) - int(encoding == 2);
}

inline int trit_unpack(uint8_t packed, int pos) {
    uint8_t encoding = (packed >> (pos * 2)) & 0x3;
    return trit_sign(encoding);
}

inline float ternary_dot4(uint8_t packed, float4 x) {
    int s0 = trit_sign(packed & 0x3);
    int s1 = trit_sign((packed >> 2) & 0x3);
    int s2 = trit_sign((packed >> 4) & 0x3);
    int s3 = trit_sign((packed >> 6) & 0x3);
    return float(s0) * x.x + float(s1) * x.y + float(s2) * x.z + float(s3) * x.w;
}

#endif // YINSEN_TRIT_PRIMITIVES

// =============================================================================
// THREADGROUP-COOPERATIVE TERNARY MATVEC
// =============================================================================

/*
 * y = W @ x, with threadgroup cooperation on the K (column) dimension.
 *
 * W: packed ternary [M x N], row-major, 4 trits per byte
 * x: float input [N]
 * y: float output [M]
 *
 * Dispatch: M threadgroups, each with up to 256 threads.
 * Each threadgroup computes one output row.
 *
 * Performance model (4096x4096):
 *   Old kernel: 1 thread reads 1024 bytes of weights + 16384 bytes of x = 17408 bytes
 *               All 4096 threads read x independently = 67MB of x reads
 *   This kernel: 256 threads cooperatively load x into shared mem (16KB),
 *               each reads 4 bytes of weights per iteration = 4KB total weights/thread
 *               Total memory traffic per row: 16KB (x, once) + 4KB (weights) = 20KB
 *               vs old: 17KB per thread = ~4.4MB per row
 */
kernel void ternary_matvec_tiled(
    device const uint8_t* W [[buffer(0)]],      // Packed ternary weights [M x ceil(N/4)]
    device const float* x [[buffer(1)]],         // Input vector [N]
    device float* y [[buffer(2)]],               // Output vector [M]
    constant uint& M [[buffer(3)]],              // Number of rows
    constant uint& N [[buffer(4)]],              // Number of columns
    threadgroup float* shared_x [[threadgroup(0)]],   // Shared input vector [N]
    uint row [[threadgroup_position_in_grid]],         // Which row (threadgroup index)
    uint tid [[thread_index_in_threadgroup]],          // Thread index within threadgroup
    uint tg_size [[threads_per_threadgroup]],          // Threadgroup size
    uint simd_lane [[thread_index_in_simdgroup]],      // Lane within simdgroup
    uint simd_id [[simdgroup_index_in_threadgroup]]    // Which simdgroup
) {
    if (row >= M) return;

    // ---------------------------------------------------------------
    // Step 1: Cooperatively load input vector into shared memory
    // ---------------------------------------------------------------
    for (uint i = tid; i < N; i += tg_size) {
        shared_x[i] = x[i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ---------------------------------------------------------------
    // Step 2: Each thread computes a partial sum over its slice of K
    // ---------------------------------------------------------------
    uint bytes_per_row = (N + 3) / 4;
    device const uint8_t* w_row = W + row * bytes_per_row;

    float partial_sum = 0.0f;

    // Each thread handles columns [tid*4, tid*4 + tg_size*4, tid*4 + 2*tg_size*4, ...]
    // Processing 4 elements (1 byte) per iteration
    for (uint byte_idx = tid; byte_idx < bytes_per_row; byte_idx += tg_size) {
        uint col = byte_idx * 4;
        if (col + 4 <= N) {
            // Full 4-element block
            float4 xi = float4(shared_x[col], shared_x[col+1], shared_x[col+2], shared_x[col+3]);
            partial_sum += ternary_dot4(w_row[byte_idx], xi);
        } else if (col < N) {
            // Remainder (1-3 elements)
            uint8_t packed = w_row[byte_idx];
            for (uint j = 0; col + j < N; j++) {
                int sign = trit_unpack(packed, j);
                partial_sum += float(sign) * shared_x[col + j];
            }
        }
    }

    // ---------------------------------------------------------------
    // Step 3: Reduction - simd_sum within simdgroup, then shared mem
    // ---------------------------------------------------------------

    // Level 1: simd_sum reduces 32 threads to 1 value per simdgroup
    float simd_total = simd_sum(partial_sum);

    // Level 2: First lane of each simdgroup writes to shared memory
    // We reuse the shared_x buffer for reduction. The caller must ensure
    // shared memory is at least max(N, ceil(tg_size/32)) * sizeof(float).
    // Max simdgroups = 256/32 = 8, so 8 floats = 32 bytes minimum.
    uint num_simdgroups = (tg_size + 31) / 32;

    // Barrier to ensure all threads are done reading shared_x before we reuse it
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (simd_lane == 0) {
        shared_x[simd_id] = simd_total;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Level 3: Thread 0 sums across simdgroups and writes result
    if (tid == 0) {
        float total = 0.0f;
        for (uint s = 0; s < num_simdgroups; s++) {
            total += shared_x[s];
        }
        y[row] = total;
    }
}

// =============================================================================
// THREADGROUP-COOPERATIVE TERNARY MATVEC WITH BIAS
// =============================================================================

kernel void ternary_matvec_tiled_bias(
    device const uint8_t* W [[buffer(0)]],
    device const float* x [[buffer(1)]],
    device const float* bias [[buffer(2)]],
    device float* y [[buffer(3)]],
    constant uint& M [[buffer(4)]],
    constant uint& N [[buffer(5)]],
    threadgroup float* shared_x [[threadgroup(0)]],
    uint row [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_id [[simdgroup_index_in_threadgroup]]
) {
    if (row >= M) return;

    // Step 1: Load x into shared memory
    for (uint i = tid; i < N; i += tg_size) {
        shared_x[i] = x[i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Step 2: Partial sums
    uint bytes_per_row = (N + 3) / 4;
    device const uint8_t* w_row = W + row * bytes_per_row;

    float partial_sum = 0.0f;
    for (uint byte_idx = tid; byte_idx < bytes_per_row; byte_idx += tg_size) {
        uint col = byte_idx * 4;
        if (col + 4 <= N) {
            float4 xi = float4(shared_x[col], shared_x[col+1], shared_x[col+2], shared_x[col+3]);
            partial_sum += ternary_dot4(w_row[byte_idx], xi);
        } else if (col < N) {
            uint8_t packed = w_row[byte_idx];
            for (uint j = 0; col + j < N; j++) {
                int sign = trit_unpack(packed, j);
                partial_sum += float(sign) * shared_x[col + j];
            }
        }
    }

    // Step 3: Reduction
    float simd_total = simd_sum(partial_sum);
    uint num_simdgroups = (tg_size + 31) / 32;

    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (simd_lane == 0) {
        shared_x[simd_id] = simd_total;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid == 0) {
        float total = 0.0f;
        for (uint s = 0; s < num_simdgroups; s++) {
            total += shared_x[s];
        }
        y[row] = total + bias[row];
    }
}
