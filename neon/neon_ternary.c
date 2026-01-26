/*
 * neon_ternary.c - NEON TBL-based Ternary MatMul for M4
 *
 * PLAN B: Since Apple blocks ZA tile access, we use NEON intrinsics.
 * 
 * "Inflate & Crush" Strategy:
 *   - Use VLD4 to load activations into 4 streams matching weight bit positions
 *   - Use TBL to expand 2-bit trits to signed int8
 *   - Use SDOT for 4-way multiply-accumulate
 *
 * Weight packing (K-vertical):
 *   Each byte contains 4 weights for the SAME output channel along K:
 *   byte = W[k] | (W[k+1] << 2) | (W[k+2] << 4) | (W[k+3] << 6)
 *   where W[i] is 2-bit trit: 00=0, 01=+1, 10=-1, 11=0
 *
 * Performance target: 300-500+ GFLOP/s on M4
 *
 * Copyright 2026 Trix Research
 */

#include <arm_neon.h>
#include <stdint.h>
#include <stddef.h>
#include <string.h>
#include <stdlib.h>  // for posix_memalign

/*
 * neon_ternary_matvec_ref - Reference implementation
 *
 * Computes: out[n] = sum_k(act[k] * decode(wgt[n, k]))
 */
void neon_ternary_matvec_ref(
    int32_t* __restrict__ out,
    const int8_t* __restrict__ act,
    const uint8_t* __restrict__ wgt,
    int N,  // Output channels
    int K   // Input channels (must be multiple of 4)
) {
    const int K_packed = K / 4;
    
    for (int n = 0; n < N; n++) {
        int32_t acc = 0;
        const uint8_t* w_row = wgt + n * K_packed;
        
        for (int k = 0; k < K; k += 4) {
            uint8_t packed = w_row[k / 4];
            
            for (int i = 0; i < 4; i++) {
                int trit = (packed >> (i * 2)) & 0x3;
                int8_t a = act[k + i];
                
                if (trit == 1) acc += a;       // +1
                else if (trit == 2) acc -= a;  // -1
                // trit == 0 or 3: no-op
            }
        }
        out[n] = acc;
    }
}

/*
 * Decode table for TBL: 2-bit trit index -> signed 8-bit weight
 * Index 0 (00) -> 0
 * Index 1 (01) -> +1
 * Index 2 (10) -> -1 (0xFF as signed)
 * Index 3 (11) -> 0
 */
static const int8_t TRIT_DECODE_TABLE[16] __attribute__((aligned(16))) = {
    0, 1, -1, 0,  // Replicated 4x for TBL
    0, 1, -1, 0,
    0, 1, -1, 0,
    0, 1, -1, 0
};

/*
 * neon_ternary_matvec_sdot - "Inflate & Crush" SDOT kernel
 *
 * The hero instruction is VLD4 which automatically deinterleaves activations
 * into 4 streams that match the packed weight bit positions:
 *   - Stream 0: A[0], A[4], A[8], ...  matches bits 1:0
 *   - Stream 1: A[1], A[5], A[9], ...  matches bits 3:2
 *   - Stream 2: A[2], A[6], A[10], ... matches bits 5:4
 *   - Stream 3: A[3], A[7], A[11], ... matches bits 7:6
 *
 * This eliminates expensive manual deinterleaving!
 *
 * Optimization: Use 4 independent accumulators to break dependency chains
 * and allow instruction-level parallelism.
 */
#if defined(__ARM_FEATURE_DOTPROD)
void neon_ternary_matvec_sdot(
    int32_t* __restrict__ out,
    const int8_t* __restrict__ act,
    const uint8_t* __restrict__ wgt,
    int N,  // Output channels
    int K   // Input channels (must be multiple of 64)
) {
    const int K_packed = K / 4;
    
    // Load decode table
    int8x16_t lut = vld1q_s8(TRIT_DECODE_TABLE);
    uint8x16_t mask_03 = vdupq_n_u8(0x03);
    
    for (int n = 0; n < N; n++) {
        // 4 independent accumulators to break dependency chains
        int32x4_t acc0 = vdupq_n_s32(0);
        int32x4_t acc1 = vdupq_n_s32(0);
        int32x4_t acc2 = vdupq_n_s32(0);
        int32x4_t acc3 = vdupq_n_s32(0);
        
        const int8_t* a_ptr = act;
        const uint8_t* w_ptr = wgt + n * K_packed;
        
        // Process 64 K-steps per iteration
        for (int k = 0; k < K; k += 64) {
            // ===== THE MAGIC: VLD4 deinterleaves for free =====
            int8x16x4_t a_streams = vld4q_s8(a_ptr);
            a_ptr += 64;
            
            // Load 16 packed bytes = 64 weights
            uint8x16_t w_packed = vld1q_u8(w_ptr);
            w_ptr += 16;
            
            // Stream 0: bits 1:0 -> acc0
            uint8x16_t idx0 = vandq_u8(w_packed, mask_03);
            int8x16_t w0 = vqtbl1q_s8(lut, idx0);
            acc0 = vdotq_s32(acc0, w0, a_streams.val[0]);
            
            // Stream 1: bits 3:2 -> acc1
            uint8x16_t idx1 = vandq_u8(vshrq_n_u8(w_packed, 2), mask_03);
            int8x16_t w1 = vqtbl1q_s8(lut, idx1);
            acc1 = vdotq_s32(acc1, w1, a_streams.val[1]);
            
            // Stream 2: bits 5:4 -> acc2
            uint8x16_t idx2 = vandq_u8(vshrq_n_u8(w_packed, 4), mask_03);
            int8x16_t w2 = vqtbl1q_s8(lut, idx2);
            acc2 = vdotq_s32(acc2, w2, a_streams.val[2]);
            
            // Stream 3: bits 7:6 -> acc3
            uint8x16_t idx3 = vshrq_n_u8(w_packed, 6);
            int8x16_t w3 = vqtbl1q_s8(lut, idx3);
            acc3 = vdotq_s32(acc3, w3, a_streams.val[3]);
        }
        
        // Merge accumulators and reduce
        int32x4_t acc = vaddq_s32(vaddq_s32(acc0, acc1), vaddq_s32(acc2, acc3));
        out[n] = vaddvq_s32(acc);
    }
}
#endif

/*
 * neon_ternary_matvec_sdot_4oc - "Quad-Pipe" Broadcast Kernel (legacy)
 */
#if defined(__ARM_FEATURE_DOTPROD)
void neon_ternary_matvec_sdot_4oc(
    int32_t* __restrict__ out,
    const int8_t* __restrict__ act,
    const uint8_t* __restrict__ wgt,
    int N,  // Output channels (must be multiple of 4)
    int K   // Input channels (must be multiple of 64)
) {
    const int K_packed = K / 4;
    int8x16_t lut = vld1q_s8(TRIT_DECODE_TABLE);
    uint8x16_t mask_03 = vdupq_n_u8(0x03);
    
    for (int n = 0; n < N; n += 4) {
        int32x4_t acc0 = vdupq_n_s32(0);
        int32x4_t acc1 = vdupq_n_s32(0);
        int32x4_t acc2 = vdupq_n_s32(0);
        int32x4_t acc3 = vdupq_n_s32(0);
        
        const int8_t* a_ptr = act;
        const uint8_t* w0_ptr = wgt + (n + 0) * K_packed;
        const uint8_t* w1_ptr = wgt + (n + 1) * K_packed;
        const uint8_t* w2_ptr = wgt + (n + 2) * K_packed;
        const uint8_t* w3_ptr = wgt + (n + 3) * K_packed;
        
        for (int k = 0; k < K; k += 64) {
            int8x16x4_t a = vld4q_s8(a_ptr);
            a_ptr += 64;
            
            #define PROCESS_OC(acc, w_ptr) { \
                uint8x16_t w = vld1q_u8(w_ptr); w_ptr += 16; \
                acc = vdotq_s32(acc, vqtbl1q_s8(lut, vandq_u8(w, mask_03)), a.val[0]); \
                acc = vdotq_s32(acc, vqtbl1q_s8(lut, vandq_u8(vshrq_n_u8(w, 2), mask_03)), a.val[1]); \
                acc = vdotq_s32(acc, vqtbl1q_s8(lut, vandq_u8(vshrq_n_u8(w, 4), mask_03)), a.val[2]); \
                acc = vdotq_s32(acc, vqtbl1q_s8(lut, vshrq_n_u8(w, 6)), a.val[3]); \
            }
            
            PROCESS_OC(acc0, w0_ptr);
            PROCESS_OC(acc1, w1_ptr);
            PROCESS_OC(acc2, w2_ptr);
            PROCESS_OC(acc3, w3_ptr);
            #undef PROCESS_OC
        }
        
        out[n + 0] = vaddvq_s32(acc0);
        out[n + 1] = vaddvq_s32(acc1);
        out[n + 2] = vaddvq_s32(acc2);
        out[n + 3] = vaddvq_s32(acc3);
    }
}
#endif

/*
 * neon_ternary_matvec_sdot_8oc - "Block-8" Latency Hiding Kernel
 *
 * Process 8 output channels simultaneously to hide SDOT/TBL latency (~3 cycles).
 * With 8 independent accumulator chains, the out-of-order engine can keep
 * the ALU saturated on every cycle.
 *
 * Uses ~30 vector registers (pushing the 32-register limit).
 */
#if defined(__ARM_FEATURE_DOTPROD)
void neon_ternary_matvec_sdot_8oc(
    int32_t* __restrict__ out,
    const int8_t* __restrict__ act,
    const uint8_t* __restrict__ wgt,
    int N,  // Output channels (must be multiple of 8)
    int K   // Input channels (must be multiple of 64)
) {
    const int K_packed = K / 4;
    
    // Constants (v30, v31 equivalent)
    int8x16_t lut = vld1q_s8(TRIT_DECODE_TABLE);
    uint8x16_t mask_03 = vdupq_n_u8(0x03);
    
    // Process 8 output channels at a time
    for (int n = 0; n < N; n += 8) {
        // 8 Accumulators - keep dependency chains long
        int32x4_t acc0 = vdupq_n_s32(0);
        int32x4_t acc1 = vdupq_n_s32(0);
        int32x4_t acc2 = vdupq_n_s32(0);
        int32x4_t acc3 = vdupq_n_s32(0);
        int32x4_t acc4 = vdupq_n_s32(0);
        int32x4_t acc5 = vdupq_n_s32(0);
        int32x4_t acc6 = vdupq_n_s32(0);
        int32x4_t acc7 = vdupq_n_s32(0);
        
        const int8_t* a_ptr = act;
        
        // Pre-calculate weight row pointers
        const uint8_t* w0_ptr = wgt + (n + 0) * K_packed;
        const uint8_t* w1_ptr = wgt + (n + 1) * K_packed;
        const uint8_t* w2_ptr = wgt + (n + 2) * K_packed;
        const uint8_t* w3_ptr = wgt + (n + 3) * K_packed;
        const uint8_t* w4_ptr = wgt + (n + 4) * K_packed;
        const uint8_t* w5_ptr = wgt + (n + 5) * K_packed;
        const uint8_t* w6_ptr = wgt + (n + 6) * K_packed;
        const uint8_t* w7_ptr = wgt + (n + 7) * K_packed;
        
        // Inner loop: 64 K-steps per iteration
        for (int k = 0; k < K; k += 64) {
            // Load activations (broadcast to all 8 channels)
            int8x16x4_t a = vld4q_s8(a_ptr);
            a_ptr += 64;
            
            // Macro to process one channel
            #define PROCESS_CHANNEL(ACC, W_PTR) { \
                uint8x16_t w = vld1q_u8(W_PTR); W_PTR += 16; \
                ACC = vdotq_s32(ACC, vqtbl1q_s8(lut, vandq_u8(w, mask_03)), a.val[0]); \
                ACC = vdotq_s32(ACC, vqtbl1q_s8(lut, vandq_u8(vshrq_n_u8(w, 2), mask_03)), a.val[1]); \
                ACC = vdotq_s32(ACC, vqtbl1q_s8(lut, vandq_u8(vshrq_n_u8(w, 4), mask_03)), a.val[2]); \
                ACC = vdotq_s32(ACC, vqtbl1q_s8(lut, vshrq_n_u8(w, 6)), a.val[3]); \
            }
            
            // Fire the barrage - compiler will interleave to hide latency
            PROCESS_CHANNEL(acc0, w0_ptr);
            PROCESS_CHANNEL(acc1, w1_ptr);
            PROCESS_CHANNEL(acc2, w2_ptr);
            PROCESS_CHANNEL(acc3, w3_ptr);
            PROCESS_CHANNEL(acc4, w4_ptr);
            PROCESS_CHANNEL(acc5, w5_ptr);
            PROCESS_CHANNEL(acc6, w6_ptr);
            PROCESS_CHANNEL(acc7, w7_ptr);
            
            #undef PROCESS_CHANNEL
        }
        
        // Final reduction and store
        out[n + 0] = vaddvq_s32(acc0);
        out[n + 1] = vaddvq_s32(acc1);
        out[n + 2] = vaddvq_s32(acc2);
        out[n + 3] = vaddvq_s32(acc3);
        out[n + 4] = vaddvq_s32(acc4);
        out[n + 5] = vaddvq_s32(acc5);
        out[n + 6] = vaddvq_s32(acc6);
        out[n + 7] = vaddvq_s32(acc7);
    }
}
#endif

/*
 * neon_ternary_matvec - Main entry point (uses SDOT if available)
 */
void neon_ternary_matvec(
    int32_t* __restrict__ out,
    const int8_t* __restrict__ act,
    const uint8_t* __restrict__ wgt,
    int N,  // Output channels
    int K   // Input channels (must be multiple of 64 for best perf)
) {
#if defined(__ARM_FEATURE_DOTPROD)
    // Use 8-output-channel version for best latency hiding
    if (N % 8 == 0) {
        neon_ternary_matvec_sdot_8oc(out, act, wgt, N, K);
    } else if (N % 4 == 0) {
        neon_ternary_matvec_sdot_4oc(out, act, wgt, N, K);
    } else {
        neon_ternary_matvec_sdot(out, act, wgt, N, K);
    }
#else
    // Fallback to reference for non-SDOT systems
    neon_ternary_matvec_ref(out, act, wgt, N, K);
#endif
}

/*
 * neon_ternary_matmul - Matrix multiplication (batch of vectors)
 *
 * Computes: out[m, n] = sum_k(act[m, k] * decode(wgt[n, k]))
 */
void neon_ternary_matmul(
    int32_t* __restrict__ out,
    const int8_t* __restrict__ act,
    const uint8_t* __restrict__ wgt,
    int M,  // Batch size
    int N,  // Output channels
    int K   // Input channels
) {
    for (int m = 0; m < M; m++) {
        neon_ternary_matvec(
            out + m * N,
            act + m * K,
            wgt,
            N, K
        );
    }
}

/*
 * pack_weights_k_vertical - Pack weights into K-vertical format (Linear layout)
 *
 * Input: weights[N, K] with values -1, 0, +1
 * Output: packed[N, K/4] with 4 trits per byte
 *
 * Layout: Row-major, each row is K/4 bytes
 *   packed[n * K/4 + k/4] = 4 weights for output channel n, input k..k+3
 */
void pack_weights_k_vertical(
    uint8_t* __restrict__ packed,
    const int8_t* __restrict__ weights,
    int N,
    int K
) {
    const int K_packed = K / 4;
    
    for (int n = 0; n < N; n++) {
        for (int k = 0; k < K; k += 4) {
            uint8_t byte = 0;
            for (int i = 0; i < 4; i++) {
                int8_t w = weights[n * K + k + i];
                uint8_t trit;
                if (w == 1) trit = 1;
                else if (w == -1) trit = 2;
                else trit = 0;
                byte |= (trit << (i * 2));
            }
            packed[n * K_packed + k/4] = byte;
        }
    }
}

/*
 * pack_weights_blocked8 - Pack weights into Blocked-8 format for cache efficiency
 *
 * Input: weights[N, K] with values -1, 0, +1
 * Output: packed[N/8, K/4, 8] - blocked by 8 output channels
 *
 * Layout: Groups of 8 output channels are stored together for each K-block
 *   Block(n_block, k_block):
 *     Row0[k_block*64 .. k_block*64+63] (16 bytes)
 *     Row1[k_block*64 .. k_block*64+63] (16 bytes)
 *     ...
 *     Row7[k_block*64 .. k_block*64+63] (16 bytes)
 *   Total: 128 bytes per block, fits in 2 cache lines
 *
 * This ensures all 8 weight row loads in the inner loop hit the same cache lines!
 */
void pack_weights_blocked8(
    uint8_t* __restrict__ packed,
    const int8_t* __restrict__ weights,
    int N,  // Must be multiple of 8
    int K   // Must be multiple of 64
) {
    const int K_packed = K / 4;       // Packed bytes per row
    const int K_blocks = K / 64;      // Number of 64-element K blocks
    const int N_blocks = N / 8;       // Number of 8-row N blocks
    
    // Each block is 8 rows × 16 bytes = 128 bytes
    const int block_stride = 8 * 16;  // 128 bytes per block
    
    for (int nb = 0; nb < N_blocks; nb++) {
        for (int kb = 0; kb < K_blocks; kb++) {
            // Destination: block (nb, kb)
            uint8_t* dst = packed + (nb * K_blocks + kb) * block_stride;
            
            // Pack 8 rows for this K-block
            for (int row = 0; row < 8; row++) {
                int n = nb * 8 + row;
                int k_start = kb * 64;
                
                // Pack 16 bytes (64 weights) for this row
                for (int kk = 0; kk < 64; kk += 4) {
                    int k = k_start + kk;
                    uint8_t byte = 0;
                    for (int i = 0; i < 4; i++) {
                        int8_t w = weights[n * K + k + i];
                        uint8_t trit;
                        if (w == 1) trit = 1;
                        else if (w == -1) trit = 2;
                        else trit = 0;
                        byte |= (trit << (i * 2));
                    }
                    dst[row * 16 + kk/4] = byte;
                }
            }
        }
    }
}

/*
 * neon_ternary_matvec_blocked8 - Kernel optimized for Blocked-8 weight layout
 *
 * This kernel achieves maximum cache efficiency by reading 8 consecutive
 * weight rows from the same 128-byte block (2 cache lines).
 * 
 * Optimizations:
 * - Blocked-8 layout for cache efficiency
 * - Software prefetching (2 blocks ahead)
 */
#if defined(__ARM_FEATURE_DOTPROD)
void neon_ternary_matvec_blocked8(
    int32_t* __restrict__ out,
    const int8_t* __restrict__ act,
    const uint8_t* __restrict__ wgt,  // Blocked-8 format!
    int N,  // Output channels (must be multiple of 8)
    int K   // Input channels (must be multiple of 64)
) {
    const int K_blocks = K / 64;
    const int block_stride = 8 * 16;  // 128 bytes per block
    
    // Constants
    int8x16_t lut = vld1q_s8(TRIT_DECODE_TABLE);
    uint8x16_t mask_03 = vdupq_n_u8(0x03);
    
    // Process 8 output channels at a time
    for (int n = 0; n < N; n += 8) {
        int nb = n / 8;
        
        // 8 Accumulators
        int32x4_t acc0 = vdupq_n_s32(0);
        int32x4_t acc1 = vdupq_n_s32(0);
        int32x4_t acc2 = vdupq_n_s32(0);
        int32x4_t acc3 = vdupq_n_s32(0);
        int32x4_t acc4 = vdupq_n_s32(0);
        int32x4_t acc5 = vdupq_n_s32(0);
        int32x4_t acc6 = vdupq_n_s32(0);
        int32x4_t acc7 = vdupq_n_s32(0);
        
        const int8_t* a_ptr = act;
        const uint8_t* w_base = wgt + nb * K_blocks * block_stride;
        
        // Inner loop over K-blocks
        for (int kb = 0; kb < K_blocks; kb++) {
            // Prefetch next blocks (2 ahead)
            if (kb + 2 < K_blocks) {
                __builtin_prefetch(a_ptr + 128, 0, 3);
                __builtin_prefetch(w_base + (kb + 2) * block_stride, 0, 3);
                __builtin_prefetch(w_base + (kb + 2) * block_stride + 64, 0, 3);
            }
            
            // Load activations (64 elements, deinterleaved)
            int8x16x4_t a = vld4q_s8(a_ptr);
            a_ptr += 64;
            
            // Weight block base pointer - all 8 rows are contiguous!
            const uint8_t* w_block = w_base + kb * block_stride;
            
            // Process all 8 channels
            #define PROCESS_ROW(ACC, ROW) { \
                uint8x16_t w = vld1q_u8(w_block + ROW * 16); \
                ACC = vdotq_s32(ACC, vqtbl1q_s8(lut, vandq_u8(w, mask_03)), a.val[0]); \
                ACC = vdotq_s32(ACC, vqtbl1q_s8(lut, vandq_u8(vshrq_n_u8(w, 2), mask_03)), a.val[1]); \
                ACC = vdotq_s32(ACC, vqtbl1q_s8(lut, vandq_u8(vshrq_n_u8(w, 4), mask_03)), a.val[2]); \
                ACC = vdotq_s32(ACC, vqtbl1q_s8(lut, vshrq_n_u8(w, 6)), a.val[3]); \
            }
            
            PROCESS_ROW(acc0, 0);
            PROCESS_ROW(acc1, 1);
            PROCESS_ROW(acc2, 2);
            PROCESS_ROW(acc3, 3);
            PROCESS_ROW(acc4, 4);
            PROCESS_ROW(acc5, 5);
            PROCESS_ROW(acc6, 6);
            PROCESS_ROW(acc7, 7);
            
            #undef PROCESS_ROW
        }
        
        // Final reduction and store
        out[n + 0] = vaddvq_s32(acc0);
        out[n + 1] = vaddvq_s32(acc1);
        out[n + 2] = vaddvq_s32(acc2);
        out[n + 3] = vaddvq_s32(acc3);
        out[n + 4] = vaddvq_s32(acc4);
        out[n + 5] = vaddvq_s32(acc5);
        out[n + 6] = vaddvq_s32(acc6);
        out[n + 7] = vaddvq_s32(acc7);
    }
}
#endif

/* ========================================================================
 * INT8 DIRECT KERNELS - Zero TBL Overhead
 * ========================================================================
 * 
 * These kernels use pre-expanded Int8 weights (-1, 0, +1 stored as bytes).
 * Trade 4x memory for ~5x speed by eliminating all TBL unpacking.
 *
 * Memory comparison for 7B model:
 *   2-bit packed: 1.75 GB
 *   Int8 direct:  7.0 GB  (fits easily on 16GB+ Macs)
 *
 * Target: 600-900 GOP/s (bandwidth bound)
 */

/*
 * pack_weights_int8_rowmajor - Simple row-major Int8 layout
 *
 * Input: weights[N, K] with values -1, 0, +1
 * Output: packed[N, K] - same layout, just int8
 */
void pack_weights_int8_rowmajor(
    int8_t* __restrict__ packed,
    const int8_t* __restrict__ weights,
    int N,
    int K
) {
    memcpy(packed, weights, N * K);
}

/*
 * pack_weights_int8_blocked8 - Pack Int8 weights into Blocked-8 format
 *
 * Layout: Groups of 8 output channels × 16 K-elements stored together
 *   Block(n_block, k_block):
 *     Row0[k..k+15] (16 bytes), Row1[k..k+15] (16 bytes), ..., Row7[k..k+15] (16 bytes)
 *   Total: 128 bytes per block = 2 cache lines
 */
void pack_weights_int8_blocked8(
    int8_t* __restrict__ packed,
    const int8_t* __restrict__ weights,
    int N,  // Must be multiple of 8
    int K   // Must be multiple of 16
) {
    const int K_blocks = K / 16;
    const int N_blocks = N / 8;
    const int block_stride = 8 * 16;  // 128 bytes per block
    
    for (int nb = 0; nb < N_blocks; nb++) {
        for (int kb = 0; kb < K_blocks; kb++) {
            int8_t* dst = packed + (nb * K_blocks + kb) * block_stride;
            
            for (int row = 0; row < 8; row++) {
                int n = nb * 8 + row;
                int k_start = kb * 16;
                
                // Copy 16 weights for this row
                for (int kk = 0; kk < 16; kk++) {
                    dst[row * 16 + kk] = weights[n * K + k_start + kk];
                }
            }
        }
    }
}

/*
 * neon_int8_matvec_8oc - Zero-overhead Int8 SDOT kernel
 *
 * No TBL, no bit manipulation - pure SDOT throughput.
 * Weights in simple row-major layout [N][K].
 * Uses VLD4 to deinterleave activations, matches with row-wise weight loads.
 *
 * Target: 600-900 GOP/s (memory bandwidth limited)
 */
#if defined(__ARM_FEATURE_DOTPROD)
void neon_int8_matvec_8oc(
    int32_t* __restrict__ out,
    const int8_t* __restrict__ act,
    const int8_t* __restrict__ wgt,  // Int8 row-major [N][K]
    int N,  // Output channels (must be multiple of 8)
    int K   // Input channels (must be multiple of 16)
) {
    // Process 8 output channels at a time
    for (int n = 0; n < N; n += 8) {
        // 8 Accumulators
        int32x4_t acc0 = vdupq_n_s32(0);
        int32x4_t acc1 = vdupq_n_s32(0);
        int32x4_t acc2 = vdupq_n_s32(0);
        int32x4_t acc3 = vdupq_n_s32(0);
        int32x4_t acc4 = vdupq_n_s32(0);
        int32x4_t acc5 = vdupq_n_s32(0);
        int32x4_t acc6 = vdupq_n_s32(0);
        int32x4_t acc7 = vdupq_n_s32(0);
        
        const int8_t* a_ptr = act;
        
        // Row pointers for 8 output channels
        const int8_t* w0_ptr = wgt + (n + 0) * K;
        const int8_t* w1_ptr = wgt + (n + 1) * K;
        const int8_t* w2_ptr = wgt + (n + 2) * K;
        const int8_t* w3_ptr = wgt + (n + 3) * K;
        const int8_t* w4_ptr = wgt + (n + 4) * K;
        const int8_t* w5_ptr = wgt + (n + 5) * K;
        const int8_t* w6_ptr = wgt + (n + 6) * K;
        const int8_t* w7_ptr = wgt + (n + 7) * K;
        
        // Process 16 K-elements per iteration
        for (int k = 0; k < K; k += 16) {
            // Prefetch ahead
            if (k + 32 < K) {
                __builtin_prefetch(a_ptr + 32, 0, 3);
                __builtin_prefetch(w0_ptr + 32, 0, 3);
                __builtin_prefetch(w4_ptr + 32, 0, 3);
            }
            
            // Load 16 activations
            int8x16_t a = vld1q_s8(a_ptr);
            a_ptr += 16;
            
            // Load 16 weights for each output channel and SDOT
            // SDOT: acc[i] += dot(w[4i:4i+4], a[4i:4i+4]) for i=0..3
            
            int8x16_t w0 = vld1q_s8(w0_ptr); w0_ptr += 16;
            int8x16_t w1 = vld1q_s8(w1_ptr); w1_ptr += 16;
            int8x16_t w2 = vld1q_s8(w2_ptr); w2_ptr += 16;
            int8x16_t w3 = vld1q_s8(w3_ptr); w3_ptr += 16;
            int8x16_t w4 = vld1q_s8(w4_ptr); w4_ptr += 16;
            int8x16_t w5 = vld1q_s8(w5_ptr); w5_ptr += 16;
            int8x16_t w6 = vld1q_s8(w6_ptr); w6_ptr += 16;
            int8x16_t w7 = vld1q_s8(w7_ptr); w7_ptr += 16;
            
            // SDOT: 4 dot products of 4 elements each = 16 elements total
            acc0 = vdotq_s32(acc0, w0, a);
            acc1 = vdotq_s32(acc1, w1, a);
            acc2 = vdotq_s32(acc2, w2, a);
            acc3 = vdotq_s32(acc3, w3, a);
            acc4 = vdotq_s32(acc4, w4, a);
            acc5 = vdotq_s32(acc5, w5, a);
            acc6 = vdotq_s32(acc6, w6, a);
            acc7 = vdotq_s32(acc7, w7, a);
        }
        
        // Final horizontal reduction and store
        out[n + 0] = vaddvq_s32(acc0);
        out[n + 1] = vaddvq_s32(acc1);
        out[n + 2] = vaddvq_s32(acc2);
        out[n + 3] = vaddvq_s32(acc3);
        out[n + 4] = vaddvq_s32(acc4);
        out[n + 5] = vaddvq_s32(acc5);
        out[n + 6] = vaddvq_s32(acc6);
        out[n + 7] = vaddvq_s32(acc7);
    }
}

/*
 * neon_int8_matvec_blocked8 - Int8 SDOT kernel with Blocked-8 layout
 *
 * Cache-optimized: 8 weight rows packed together per K-block.
 * All 8 loads hit the same 2 cache lines (128 bytes).
 *
 * Expected: 400-700 GOP/s (bandwidth limited at ~100 GB/s)
 */
void neon_int8_matvec_blocked8(
    int32_t* __restrict__ out,
    const int8_t* __restrict__ act,
    const int8_t* __restrict__ wgt,  // Blocked-8 format
    int N,  // Output channels (must be multiple of 8)
    int K   // Input channels (must be multiple of 16)
) {
    const int K_blocks = K / 16;
    const int block_stride = 8 * 16;  // 128 bytes per block
    
    // Process 8 output channels at a time
    for (int n = 0; n < N; n += 8) {
        int nb = n / 8;
        
        // 8 Accumulators
        int32x4_t acc0 = vdupq_n_s32(0);
        int32x4_t acc1 = vdupq_n_s32(0);
        int32x4_t acc2 = vdupq_n_s32(0);
        int32x4_t acc3 = vdupq_n_s32(0);
        int32x4_t acc4 = vdupq_n_s32(0);
        int32x4_t acc5 = vdupq_n_s32(0);
        int32x4_t acc6 = vdupq_n_s32(0);
        int32x4_t acc7 = vdupq_n_s32(0);
        
        const int8_t* a_ptr = act;
        const int8_t* w_base = wgt + nb * K_blocks * block_stride;
        
        // Inner loop over K-blocks
        for (int kb = 0; kb < K_blocks; kb++) {
            // Prefetch 2 blocks ahead
            if (kb + 2 < K_blocks) {
                __builtin_prefetch(a_ptr + 32, 0, 3);
                __builtin_prefetch(w_base + (kb + 2) * block_stride, 0, 3);
                __builtin_prefetch(w_base + (kb + 2) * block_stride + 64, 0, 3);
            }
            
            // Load 16 activations
            int8x16_t a = vld1q_s8(a_ptr);
            a_ptr += 16;
            
            // Weight block - all 8 rows contiguous!
            const int8_t* w_block = w_base + kb * block_stride;
            
            // Load and SDOT for all 8 channels
            int8x16_t w0 = vld1q_s8(w_block + 0 * 16);
            int8x16_t w1 = vld1q_s8(w_block + 1 * 16);
            int8x16_t w2 = vld1q_s8(w_block + 2 * 16);
            int8x16_t w3 = vld1q_s8(w_block + 3 * 16);
            int8x16_t w4 = vld1q_s8(w_block + 4 * 16);
            int8x16_t w5 = vld1q_s8(w_block + 5 * 16);
            int8x16_t w6 = vld1q_s8(w_block + 6 * 16);
            int8x16_t w7 = vld1q_s8(w_block + 7 * 16);
            
            acc0 = vdotq_s32(acc0, w0, a);
            acc1 = vdotq_s32(acc1, w1, a);
            acc2 = vdotq_s32(acc2, w2, a);
            acc3 = vdotq_s32(acc3, w3, a);
            acc4 = vdotq_s32(acc4, w4, a);
            acc5 = vdotq_s32(acc5, w5, a);
            acc6 = vdotq_s32(acc6, w6, a);
            acc7 = vdotq_s32(acc7, w7, a);
        }
        
        // Final reduction and store
        out[n + 0] = vaddvq_s32(acc0);
        out[n + 1] = vaddvq_s32(acc1);
        out[n + 2] = vaddvq_s32(acc2);
        out[n + 3] = vaddvq_s32(acc3);
        out[n + 4] = vaddvq_s32(acc4);
        out[n + 5] = vaddvq_s32(acc5);
        out[n + 6] = vaddvq_s32(acc6);
        out[n + 7] = vaddvq_s32(acc7);
    }
}

/*
 * pack_weights_int8_blocked8_k32 - Int8 Blocked-8 with 32-element K blocks
 *
 * Larger blocks = less loop overhead, better prefetching.
 * Block size: 8 rows × 32 cols = 256 bytes = 4 cache lines
 */
void pack_weights_int8_blocked8_k32(
    int8_t* __restrict__ packed,
    const int8_t* __restrict__ weights,
    int N,  // Must be multiple of 8
    int K   // Must be multiple of 32
) {
    const int K_blocks = K / 32;
    const int N_blocks = N / 8;
    const int block_stride = 8 * 32;  // 256 bytes per block
    
    for (int nb = 0; nb < N_blocks; nb++) {
        for (int kb = 0; kb < K_blocks; kb++) {
            int8_t* dst = packed + (nb * K_blocks + kb) * block_stride;
            
            for (int row = 0; row < 8; row++) {
                int n = nb * 8 + row;
                int k_start = kb * 32;
                
                for (int kk = 0; kk < 32; kk++) {
                    dst[row * 32 + kk] = weights[n * K + k_start + kk];
                }
            }
        }
    }
}

/*
 * neon_int8_matvec_blocked8_k32 - Int8 SDOT with 32-element K-unroll
 *
 * Process 32 K-elements per iteration (2 × vld1q_s8 per row).
 * Halves loop iterations = less branch overhead.
 */
void neon_int8_matvec_blocked8_k32(
    int32_t* __restrict__ out,
    const int8_t* __restrict__ act,
    const int8_t* __restrict__ wgt,  // Blocked-8-K32 format
    int N,  // Output channels (must be multiple of 8)
    int K   // Input channels (must be multiple of 32)
) {
    const int K_blocks = K / 32;
    const int block_stride = 8 * 32;  // 256 bytes per block
    
    for (int n = 0; n < N; n += 8) {
        int nb = n / 8;
        
        // 8 accumulators (will accumulate 2 SDOTs per iteration)
        int32x4_t acc0 = vdupq_n_s32(0);
        int32x4_t acc1 = vdupq_n_s32(0);
        int32x4_t acc2 = vdupq_n_s32(0);
        int32x4_t acc3 = vdupq_n_s32(0);
        int32x4_t acc4 = vdupq_n_s32(0);
        int32x4_t acc5 = vdupq_n_s32(0);
        int32x4_t acc6 = vdupq_n_s32(0);
        int32x4_t acc7 = vdupq_n_s32(0);
        
        const int8_t* a_ptr = act;
        const int8_t* w_base = wgt + nb * K_blocks * block_stride;
        
        for (int kb = 0; kb < K_blocks; kb++) {
            // Prefetch 2 blocks ahead
            if (kb + 2 < K_blocks) {
                __builtin_prefetch(a_ptr + 64, 0, 3);
                __builtin_prefetch(w_base + (kb + 2) * block_stride, 0, 3);
                __builtin_prefetch(w_base + (kb + 2) * block_stride + 128, 0, 3);
            }
            
            // Load 32 activations (2 × 16)
            int8x16_t a0 = vld1q_s8(a_ptr);
            int8x16_t a1 = vld1q_s8(a_ptr + 16);
            a_ptr += 32;
            
            const int8_t* w_block = w_base + kb * block_stride;
            
            // Process all 8 rows, 32 elements each
            #define PROCESS_ROW_K32(ACC, ROW) { \
                int8x16_t w0 = vld1q_s8(w_block + ROW * 32); \
                int8x16_t w1 = vld1q_s8(w_block + ROW * 32 + 16); \
                ACC = vdotq_s32(ACC, w0, a0); \
                ACC = vdotq_s32(ACC, w1, a1); \
            }
            
            PROCESS_ROW_K32(acc0, 0);
            PROCESS_ROW_K32(acc1, 1);
            PROCESS_ROW_K32(acc2, 2);
            PROCESS_ROW_K32(acc3, 3);
            PROCESS_ROW_K32(acc4, 4);
            PROCESS_ROW_K32(acc5, 5);
            PROCESS_ROW_K32(acc6, 6);
            PROCESS_ROW_K32(acc7, 7);
            
            #undef PROCESS_ROW_K32
        }
        out[n + 0] = vaddvq_s32(acc0);
        out[n + 1] = vaddvq_s32(acc1);
        out[n + 2] = vaddvq_s32(acc2);
        out[n + 3] = vaddvq_s32(acc3);
        out[n + 4] = vaddvq_s32(acc4);
        out[n + 5] = vaddvq_s32(acc5);
        out[n + 6] = vaddvq_s32(acc6);
        out[n + 7] = vaddvq_s32(acc7);
    }
}

/*
 * pack_weights_int8_blocked8_k64 - Int8 Blocked-8 with 64-element K blocks
 *
 * Block size: 8 rows × 64 cols = 512 bytes = 8 cache lines
 * Matches the 2-bit kernel's 64-element stride for direct comparison.
 */
void pack_weights_int8_blocked8_k64(
    int8_t* __restrict__ packed,
    const int8_t* __restrict__ weights,
    int N,  // Must be multiple of 8
    int K   // Must be multiple of 64
) {
    const int K_blocks = K / 64;
    const int N_blocks = N / 8;
    const int block_stride = 8 * 64;  // 512 bytes per block
    
    for (int nb = 0; nb < N_blocks; nb++) {
        for (int kb = 0; kb < K_blocks; kb++) {
            int8_t* dst = packed + (nb * K_blocks + kb) * block_stride;
            
            for (int row = 0; row < 8; row++) {
                int n = nb * 8 + row;
                int k_start = kb * 64;
                
                for (int kk = 0; kk < 64; kk++) {
                    dst[row * 64 + kk] = weights[n * K + k_start + kk];
                }
            }
        }
    }
}

/*
 * neon_int8_matvec_blocked8_k64 - Int8 SDOT with 64-element K-unroll
 *
 * Process 64 K-elements per iteration (4 × vld1q_s8 per row).
 * Maximum unroll - trades register pressure for minimal loop overhead.
 */
void neon_int8_matvec_blocked8_k64(
    int32_t* __restrict__ out,
    const int8_t* __restrict__ act,
    const int8_t* __restrict__ wgt,  // Blocked-8-K64 format
    int N,  // Output channels (must be multiple of 8)
    int K   // Input channels (must be multiple of 64)
) {
    const int K_blocks = K / 64;
    const int block_stride = 8 * 64;  // 512 bytes per block
    
    for (int n = 0; n < N; n += 8) {
        int nb = n / 8;
        
        // 8 accumulators
        int32x4_t acc0 = vdupq_n_s32(0);
        int32x4_t acc1 = vdupq_n_s32(0);
        int32x4_t acc2 = vdupq_n_s32(0);
        int32x4_t acc3 = vdupq_n_s32(0);
        int32x4_t acc4 = vdupq_n_s32(0);
        int32x4_t acc5 = vdupq_n_s32(0);
        int32x4_t acc6 = vdupq_n_s32(0);
        int32x4_t acc7 = vdupq_n_s32(0);
        
        const int8_t* a_ptr = act;
        const int8_t* w_base = wgt + nb * K_blocks * block_stride;
        
        for (int kb = 0; kb < K_blocks; kb++) {
            // Prefetch 2 blocks ahead
            if (kb + 2 < K_blocks) {
                __builtin_prefetch(a_ptr + 128, 0, 3);
                __builtin_prefetch(w_base + (kb + 2) * block_stride, 0, 3);
                __builtin_prefetch(w_base + (kb + 2) * block_stride + 256, 0, 3);
            }
            
            // Load 64 activations (4 × 16)
            int8x16_t a0 = vld1q_s8(a_ptr);
            int8x16_t a1 = vld1q_s8(a_ptr + 16);
            int8x16_t a2 = vld1q_s8(a_ptr + 32);
            int8x16_t a3 = vld1q_s8(a_ptr + 48);
            a_ptr += 64;
            
            const int8_t* w_block = w_base + kb * block_stride;
            
            // Process all 8 rows, 64 elements each
            #define PROCESS_ROW_K64(ACC, ROW) { \
                int8x16_t w0 = vld1q_s8(w_block + ROW * 64); \
                int8x16_t w1 = vld1q_s8(w_block + ROW * 64 + 16); \
                int8x16_t w2 = vld1q_s8(w_block + ROW * 64 + 32); \
                int8x16_t w3 = vld1q_s8(w_block + ROW * 64 + 48); \
                ACC = vdotq_s32(ACC, w0, a0); \
                ACC = vdotq_s32(ACC, w1, a1); \
                ACC = vdotq_s32(ACC, w2, a2); \
                ACC = vdotq_s32(ACC, w3, a3); \
            }
            
            PROCESS_ROW_K64(acc0, 0);
            PROCESS_ROW_K64(acc1, 1);
            PROCESS_ROW_K64(acc2, 2);
            PROCESS_ROW_K64(acc3, 3);
            PROCESS_ROW_K64(acc4, 4);
            PROCESS_ROW_K64(acc5, 5);
            PROCESS_ROW_K64(acc6, 6);
            PROCESS_ROW_K64(acc7, 7);
            
            #undef PROCESS_ROW_K64
        }
        out[n + 0] = vaddvq_s32(acc0);
        out[n + 1] = vaddvq_s32(acc1);
        out[n + 2] = vaddvq_s32(acc2);
        out[n + 3] = vaddvq_s32(acc3);
        out[n + 4] = vaddvq_s32(acc4);
        out[n + 5] = vaddvq_s32(acc5);
        out[n + 6] = vaddvq_s32(acc6);
        out[n + 7] = vaddvq_s32(acc7);
    }
}
#endif

/* ========================================================================
 * I8MM "MICRO-TENSOR ENGINE" KERNELS
 * ========================================================================
 *
 * Uses SMMLA (Signed Matrix Multiply Accumulate) from ARMv8.6 I8MM extension.
 * SMMLA computes a 2x2 matrix product in a single instruction:
 *   C[2x2] += A[2x8] * B[8x2]
 *
 * This is 2x denser than SDOT (which does 1x4 dot product).
 *
 * Strategy: "Twin-Pipe" - process output channels in pairs.
 *   - Duplicate activation vector to fill 2x8 matrix (fake batch=2)
 *   - Interleave weight pairs into 8x2 matrix
 *   - One SMMLA computes 2 output channels × 8 K-steps
 *
 * Target: 300-400 GOP/s
 */

#if defined(__ARM_FEATURE_MATMUL_INT8)

/*
 * pack_weights_i8mm_paired - Column-major layout for I8MM SMMLA
 *
 * SMMLA expects B matrix as 8×2 in COLUMN-MAJOR order:
 *   Bytes 0-7  = Column 0 (8 elements for output channel N)
 *   Bytes 8-15 = Column 1 (8 elements for output channel N+1)
 *
 * Input: weights[N, K] with values -1, 0, +1
 * Output: packed[N/2, K/8, 16] - 8×2 tiles in column-major
 *
 * For each pair (n, n+1) and each 8-element K block:
 *   [W[n,k+0..7], W[n+1,k+0..7]]
 */
void pack_weights_i8mm_paired(
    int8_t* __restrict__ packed,
    const int8_t* __restrict__ weights,
    int N,  // Must be multiple of 2
    int K   // Must be multiple of 8
) {
    const int K_blocks = K / 8;
    
    for (int n = 0; n < N; n += 2) {
        for (int kb = 0; kb < K_blocks; kb++) {
            int8_t* dst = packed + (n / 2) * K * 2 + kb * 16;
            const int8_t* src0 = weights + n * K + kb * 8;
            const int8_t* src1 = weights + (n + 1) * K + kb * 8;
            
            // Column 0: 8 weights for channel n
            for (int i = 0; i < 8; i++) {
                dst[i] = src0[i];
            }
            // Column 1: 8 weights for channel n+1
            for (int i = 0; i < 8; i++) {
                dst[8 + i] = src1[i];
            }
        }
    }
}

/*
 * neon_i8mm_matvec_2oc - Basic I8MM kernel (2 output channels per iteration)
 *
 * Uses vmmlaq_s32 (SMMLA) to compute 2 outputs × 8 K-steps per instruction.
 *
 * SMMLA semantics: C[2×2] += A[2×8] × B[8×2]
 *   A layout: bytes 0-7 = row 0, bytes 8-15 = row 1 (row-major)
 *   B layout: bytes 0-7 = col 0, bytes 8-15 = col 1 (column-major!)
 *   C layout: [C00, C01, C10, C11]
 *
 * For batch=1 inference, we duplicate the activation vector to both rows.
 * Result: C[0,0] and C[1,0] both contain dot(act, weights_col0) = channel n
 *         C[0,1] and C[1,1] both contain dot(act, weights_col1) = channel n+1
 */
void neon_i8mm_matvec_2oc(
    int32_t* __restrict__ out,
    const int8_t* __restrict__ act,
    const int8_t* __restrict__ wgt,  // Column-major tiles [N/2, K/8, 16]
    int N,  // Output channels (must be multiple of 2)
    int K   // Input channels (must be multiple of 16)
) {
    for (int n = 0; n < N; n += 2) {
        // Accumulator: 2x2 matrix, we extract [0] and [1] for our 2 channels
        int32x4_t acc = vdupq_n_s32(0);
        
        const int8_t* a_ptr = act;
        const int8_t* w_ptr = wgt + (n / 2) * K * 2;  // Column-major tiles
        
        for (int k = 0; k < K; k += 16) {
            // Load 16 activations
            int8x16_t a_raw = vld1q_s8(a_ptr);
            a_ptr += 16;
            
            // Duplicate to create 2x8 matrix (fake batch=2)
            // For SMMLA: row 0 = A[0..7], row 1 = same A[0..7]
            int8x16_t a_lo = vreinterpretq_s8_s64(
                vdupq_laneq_s64(vreinterpretq_s64_s8(a_raw), 0));
            int8x16_t a_hi = vreinterpretq_s8_s64(
                vdupq_laneq_s64(vreinterpretq_s64_s8(a_raw), 1));
            
            // Load column-major weight tiles
            // w_0: col0 = W[n, k:k+8], col1 = W[n+1, k:k+8]
            // w_1: col0 = W[n, k+8:k+16], col1 = W[n+1, k+8:k+16]
            int8x16_t w_0 = vld1q_s8(w_ptr);
            int8x16_t w_1 = vld1q_s8(w_ptr + 16);
            w_ptr += 32;
            
            // SMMLA: acc[2x2] += a[2x8] * w[8x2]
            acc = vmmlaq_s32(acc, a_lo, w_0);
            acc = vmmlaq_s32(acc, a_hi, w_1);
        }
        
        // C[0] = dot(act, weights_n), C[1] = dot(act, weights_n+1)
        out[n + 0] = vgetq_lane_s32(acc, 0);
        out[n + 1] = vgetq_lane_s32(acc, 1);
    }
}

/*
 * neon_i8mm_matvec_8oc - I8MM kernel with 8 output channel blocking
 *
 * Process 4 pairs (8 channels) in parallel for latency hiding.
 * Uses 4 accumulator registers for 4 SMMLA streams.
 */
void neon_i8mm_matvec_8oc(
    int32_t* __restrict__ out,
    const int8_t* __restrict__ act,
    const int8_t* __restrict__ wgt,  // Column-major tiles [N/2, K/8, 16]
    int N,  // Output channels (must be multiple of 8)
    int K   // Input channels (must be multiple of 16)
) {
    for (int n = 0; n < N; n += 8) {
        // 4 accumulators for 4 pairs (8 output channels)
        int32x4_t acc0 = vdupq_n_s32(0);
        int32x4_t acc1 = vdupq_n_s32(0);
        int32x4_t acc2 = vdupq_n_s32(0);
        int32x4_t acc3 = vdupq_n_s32(0);
        
        const int8_t* a_ptr = act;
        // Weight pointers for 4 pairs (each pair processes 2 channels)
        const int8_t* w0_ptr = wgt + ((n + 0) / 2) * K * 2;
        const int8_t* w1_ptr = wgt + ((n + 2) / 2) * K * 2;
        const int8_t* w2_ptr = wgt + ((n + 4) / 2) * K * 2;
        const int8_t* w3_ptr = wgt + ((n + 6) / 2) * K * 2;
        
        for (int k = 0; k < K; k += 16) {
            // Prefetch
            if (k + 32 < K) {
                __builtin_prefetch(a_ptr + 32, 0, 3);
                __builtin_prefetch(w0_ptr + 64, 0, 3);
                __builtin_prefetch(w2_ptr + 64, 0, 3);
            }
            
            // Load and duplicate activations
            int8x16_t a_raw = vld1q_s8(a_ptr);
            a_ptr += 16;
            
            int8x16_t a_lo = vreinterpretq_s8_s64(
                vdupq_laneq_s64(vreinterpretq_s64_s8(a_raw), 0));
            int8x16_t a_hi = vreinterpretq_s8_s64(
                vdupq_laneq_s64(vreinterpretq_s64_s8(a_raw), 1));
            
            // Process pair 0 (channels n, n+1)
            {
                int8x16_t w0 = vld1q_s8(w0_ptr);
                int8x16_t w1 = vld1q_s8(w0_ptr + 16);
                w0_ptr += 32;
                acc0 = vmmlaq_s32(acc0, a_lo, w0);
                acc0 = vmmlaq_s32(acc0, a_hi, w1);
            }
            
            // Process pair 1 (channels n+2, n+3)
            {
                int8x16_t w0 = vld1q_s8(w1_ptr);
                int8x16_t w1 = vld1q_s8(w1_ptr + 16);
                w1_ptr += 32;
                acc1 = vmmlaq_s32(acc1, a_lo, w0);
                acc1 = vmmlaq_s32(acc1, a_hi, w1);
            }
            
            // Process pair 2 (channels n+4, n+5)
            {
                int8x16_t w0 = vld1q_s8(w2_ptr);
                int8x16_t w1 = vld1q_s8(w2_ptr + 16);
                w2_ptr += 32;
                acc2 = vmmlaq_s32(acc2, a_lo, w0);
                acc2 = vmmlaq_s32(acc2, a_hi, w1);
            }
            
            // Process pair 3 (channels n+6, n+7)
            {
                int8x16_t w0 = vld1q_s8(w3_ptr);
                int8x16_t w1 = vld1q_s8(w3_ptr + 16);
                w3_ptr += 32;
                acc3 = vmmlaq_s32(acc3, a_lo, w0);
                acc3 = vmmlaq_s32(acc3, a_hi, w1);
            }
        }
        
        // Extract results
        out[n + 0] = vgetq_lane_s32(acc0, 0);
        out[n + 1] = vgetq_lane_s32(acc0, 1);
        out[n + 2] = vgetq_lane_s32(acc1, 0);
        out[n + 3] = vgetq_lane_s32(acc1, 1);
        out[n + 4] = vgetq_lane_s32(acc2, 0);
        out[n + 5] = vgetq_lane_s32(acc2, 1);
        out[n + 6] = vgetq_lane_s32(acc3, 0);
        out[n + 7] = vgetq_lane_s32(acc3, 1);
    }
}

/*
 * pack_weights_i8mm_blocked8 - Blocked I8MM layout for cache efficiency
 *
 * Combines column-major tiles with 8-channel blocking.
 * For each N-block (8 channels = 4 pairs) and K-block (16 elements = 2 SMMLA ops):
 *   Pack 4 pairs × 2 tiles = 8 tiles = 128 bytes
 *
 * Block layout: [pair0_tile0, pair0_tile1, pair1_tile0, pair1_tile1, ...]
 * Each tile is 16 bytes (col0: 8 bytes, col1: 8 bytes)
 */
void pack_weights_i8mm_blocked8(
    int8_t* __restrict__ packed,
    const int8_t* __restrict__ weights,
    int N,  // Must be multiple of 8
    int K   // Must be multiple of 16
) {
    const int K_blocks = K / 16;  // 16 K-steps per block (2 SMMLA ops)
    const int N_blocks = N / 8;
    // Block: 4 pairs × 2 tiles × 16 bytes = 128 bytes
    const int block_stride = 4 * 2 * 16;
    
    for (int nb = 0; nb < N_blocks; nb++) {
        for (int kb = 0; kb < K_blocks; kb++) {
            int8_t* dst = packed + (nb * K_blocks + kb) * block_stride;
            
            // Pack 4 pairs (8 channels)
            for (int pair = 0; pair < 4; pair++) {
                int n0 = nb * 8 + pair * 2;
                int n1 = n0 + 1;
                int k_start = kb * 16;
                
                // Tile 0: K[0..7] for this pair
                for (int i = 0; i < 8; i++) {
                    dst[pair * 32 + i] = weights[n0 * K + k_start + i];
                    dst[pair * 32 + 8 + i] = weights[n1 * K + k_start + i];
                }
                // Tile 1: K[8..15] for this pair
                for (int i = 0; i < 8; i++) {
                    dst[pair * 32 + 16 + i] = weights[n0 * K + k_start + 8 + i];
                    dst[pair * 32 + 24 + i] = weights[n1 * K + k_start + 8 + i];
                }
            }
        }
    }
}

/*
 * neon_i8mm_matvec_blocked8 - Cache-optimized I8MM kernel
 *
 * Uses blocked weight layout for sequential memory access.
 * All 4 pairs (8 channels) read from same 128-byte block.
 */
void neon_i8mm_matvec_blocked8(
    int32_t* __restrict__ out,
    const int8_t* __restrict__ act,
    const int8_t* __restrict__ wgt,  // Blocked-8 I8MM format
    int N,  // Output channels (must be multiple of 8)
    int K   // Input channels (must be multiple of 16)
) {
    const int K_blocks = K / 16;
    const int block_stride = 4 * 2 * 16;  // 128 bytes per block
    
    for (int n = 0; n < N; n += 8) {
        int nb = n / 8;
        
        int32x4_t acc0 = vdupq_n_s32(0);
        int32x4_t acc1 = vdupq_n_s32(0);
        int32x4_t acc2 = vdupq_n_s32(0);
        int32x4_t acc3 = vdupq_n_s32(0);
        
        const int8_t* a_ptr = act;
        const int8_t* w_base = wgt + nb * K_blocks * block_stride;
        
        for (int kb = 0; kb < K_blocks; kb++) {
            // Prefetch
            if (kb + 2 < K_blocks) {
                __builtin_prefetch(a_ptr + 32, 0, 3);
                __builtin_prefetch(w_base + (kb + 2) * block_stride, 0, 3);
            }
            
            // Load 16 activations
            int8x16_t a_raw = vld1q_s8(a_ptr);
            a_ptr += 16;
            
            int8x16_t a_lo = vreinterpretq_s8_s64(
                vdupq_laneq_s64(vreinterpretq_s64_s8(a_raw), 0));
            int8x16_t a_hi = vreinterpretq_s8_s64(
                vdupq_laneq_s64(vreinterpretq_s64_s8(a_raw), 1));
            
            const int8_t* w_block = w_base + kb * block_stride;
            
            // Each pair: 32 bytes = 2 tiles (tile0 for K[0..7], tile1 for K[8..15])
            #define PROCESS_PAIR(ACC, PAIR) { \
                int8x16_t w0 = vld1q_s8(w_block + PAIR * 32); \
                int8x16_t w1 = vld1q_s8(w_block + PAIR * 32 + 16); \
                ACC = vmmlaq_s32(ACC, a_lo, w0); \
                ACC = vmmlaq_s32(ACC, a_hi, w1); \
            }
            
            PROCESS_PAIR(acc0, 0);
            PROCESS_PAIR(acc1, 1);
            PROCESS_PAIR(acc2, 2);
            PROCESS_PAIR(acc3, 3);
            
            #undef PROCESS_PAIR
        }
        
        out[n + 0] = vgetq_lane_s32(acc0, 0);
        out[n + 1] = vgetq_lane_s32(acc0, 1);
        out[n + 2] = vgetq_lane_s32(acc1, 0);
        out[n + 3] = vgetq_lane_s32(acc1, 1);
        out[n + 4] = vgetq_lane_s32(acc2, 0);
        out[n + 5] = vgetq_lane_s32(acc2, 1);
        out[n + 6] = vgetq_lane_s32(acc3, 0);
        out[n + 7] = vgetq_lane_s32(acc3, 1);
    }
}

/*
 * pack_weights_i8mm_blocked16 - Blocked I8MM layout for 16 output channels
 *
 * 8 pairs × 16 K-steps = 256 bytes per block
 */
void pack_weights_i8mm_blocked16(
    int8_t* __restrict__ packed,
    const int8_t* __restrict__ weights,
    int N,  // Must be multiple of 16
    int K   // Must be multiple of 16
) {
    const int K_blocks = K / 16;
    const int N_blocks = N / 16;
    const int block_stride = 8 * 2 * 16;  // 256 bytes per block
    
    for (int nb = 0; nb < N_blocks; nb++) {
        for (int kb = 0; kb < K_blocks; kb++) {
            int8_t* dst = packed + (nb * K_blocks + kb) * block_stride;
            
            for (int pair = 0; pair < 8; pair++) {
                int n0 = nb * 16 + pair * 2;
                int n1 = n0 + 1;
                int k_start = kb * 16;
                
                // Tile 0: K[0..7]
                for (int i = 0; i < 8; i++) {
                    dst[pair * 32 + i] = weights[n0 * K + k_start + i];
                    dst[pair * 32 + 8 + i] = weights[n1 * K + k_start + i];
                }
                // Tile 1: K[8..15]
                for (int i = 0; i < 8; i++) {
                    dst[pair * 32 + 16 + i] = weights[n0 * K + k_start + 8 + i];
                    dst[pair * 32 + 24 + i] = weights[n1 * K + k_start + 8 + i];
                }
            }
        }
    }
}

/*
 * neon_i8mm_matvec_blocked16 - 16 output channel I8MM kernel
 *
 * Process 8 pairs (16 channels) per N-iteration for maximum parallelism.
 */
void neon_i8mm_matvec_blocked16(
    int32_t* __restrict__ out,
    const int8_t* __restrict__ act,
    const int8_t* __restrict__ wgt,
    int N,  // Must be multiple of 16
    int K   // Must be multiple of 16
) {
    const int K_blocks = K / 16;
    const int block_stride = 8 * 2 * 16;  // 256 bytes
    
    for (int n = 0; n < N; n += 16) {
        int nb = n / 16;
        
        // 8 accumulators for 8 pairs
        int32x4_t acc0 = vdupq_n_s32(0);
        int32x4_t acc1 = vdupq_n_s32(0);
        int32x4_t acc2 = vdupq_n_s32(0);
        int32x4_t acc3 = vdupq_n_s32(0);
        int32x4_t acc4 = vdupq_n_s32(0);
        int32x4_t acc5 = vdupq_n_s32(0);
        int32x4_t acc6 = vdupq_n_s32(0);
        int32x4_t acc7 = vdupq_n_s32(0);
        
        const int8_t* a_ptr = act;
        const int8_t* w_base = wgt + nb * K_blocks * block_stride;
        
        for (int kb = 0; kb < K_blocks; kb++) {
            if (kb + 2 < K_blocks) {
                __builtin_prefetch(a_ptr + 32, 0, 3);
                __builtin_prefetch(w_base + (kb + 2) * block_stride, 0, 3);
                __builtin_prefetch(w_base + (kb + 2) * block_stride + 128, 0, 3);
            }
            
            int8x16_t a_raw = vld1q_s8(a_ptr);
            a_ptr += 16;
            
            int8x16_t a_lo = vreinterpretq_s8_s64(
                vdupq_laneq_s64(vreinterpretq_s64_s8(a_raw), 0));
            int8x16_t a_hi = vreinterpretq_s8_s64(
                vdupq_laneq_s64(vreinterpretq_s64_s8(a_raw), 1));
            
            const int8_t* w_block = w_base + kb * block_stride;
            
            #define PROCESS_PAIR16(ACC, PAIR) { \
                int8x16_t w0 = vld1q_s8(w_block + PAIR * 32); \
                int8x16_t w1 = vld1q_s8(w_block + PAIR * 32 + 16); \
                ACC = vmmlaq_s32(ACC, a_lo, w0); \
                ACC = vmmlaq_s32(ACC, a_hi, w1); \
            }
            
            PROCESS_PAIR16(acc0, 0);
            PROCESS_PAIR16(acc1, 1);
            PROCESS_PAIR16(acc2, 2);
            PROCESS_PAIR16(acc3, 3);
            PROCESS_PAIR16(acc4, 4);
            PROCESS_PAIR16(acc5, 5);
            PROCESS_PAIR16(acc6, 6);
            PROCESS_PAIR16(acc7, 7);
            
            #undef PROCESS_PAIR16
        }
        
        out[n + 0] = vgetq_lane_s32(acc0, 0);
        out[n + 1] = vgetq_lane_s32(acc0, 1);
        out[n + 2] = vgetq_lane_s32(acc1, 0);
        out[n + 3] = vgetq_lane_s32(acc1, 1);
        out[n + 4] = vgetq_lane_s32(acc2, 0);
        out[n + 5] = vgetq_lane_s32(acc2, 1);
        out[n + 6] = vgetq_lane_s32(acc3, 0);
        out[n + 7] = vgetq_lane_s32(acc3, 1);
        out[n + 8] = vgetq_lane_s32(acc4, 0);
        out[n + 9] = vgetq_lane_s32(acc4, 1);
        out[n + 10] = vgetq_lane_s32(acc5, 0);
        out[n + 11] = vgetq_lane_s32(acc5, 1);
        out[n + 12] = vgetq_lane_s32(acc6, 0);
        out[n + 13] = vgetq_lane_s32(acc6, 1);
        out[n + 14] = vgetq_lane_s32(acc7, 0);
        out[n + 15] = vgetq_lane_s32(acc7, 1);
    }
}

#endif  // __ARM_FEATURE_MATMUL_INT8

/* ========================================================================
 * GHOST-STREAM KERNELS - Non-Temporal Weight Loading
 * ========================================================================
 *
 * The L2 cache is the critical battlefield for MatVec:
 *   - Activations: "Hot" data, accessed N times, must stay in cache
 *   - Weights: "Cold" data, accessed once, should NOT pollute cache
 *
 * Standard vld1q loads weights into L2, evicting hot activations.
 * LDNP (Load Pair Non-temporal) bypasses cache allocation:
 *   DRAM -> Registers -> ALU -> Trash (no L2 residency)
 *
 * Combined with Block-12 unrolling to maximize activation reuse:
 *   - 12 output channels processed per activation load
 *   - 12 accumulators (v0-v11)
 *   - 12 weight temps (v16-v27)
 *   - 4 activation streams or 1 activation vector (v12-v15)
 *   - Total: ~28 registers (within 32 limit)
 *
 * Target: 200+ GOP/s
 */

#if defined(__ARM_FEATURE_DOTPROD)

/*
 * pack_weights_ghost12 - Weight layout for Ghost-Stream Block-12 kernel
 *
 * For each N-block (12 channels) and K-block (16 elements):
 *   Store 12 consecutive weight vectors [Ch0..Ch11] for K[0..15]
 *   Block size: 12 × 16 = 192 bytes
 *
 * This layout ensures LDNP can load pairs of weight vectors efficiently.
 */
void pack_weights_ghost12(
    int8_t* __restrict__ packed,
    const int8_t* __restrict__ weights,
    int N,  // Must be multiple of 12
    int K   // Must be multiple of 16
) {
    const int K_blocks = K / 16;
    const int N_blocks = N / 12;
    const int block_stride = 12 * 16;  // 192 bytes per block
    
    for (int nb = 0; nb < N_blocks; nb++) {
        for (int kb = 0; kb < K_blocks; kb++) {
            int8_t* dst = packed + (nb * K_blocks + kb) * block_stride;
            
            // Pack 12 channels, 16 K-elements each
            for (int ch = 0; ch < 12; ch++) {
                int n = nb * 12 + ch;
                int k_start = kb * 16;
                
                for (int kk = 0; kk < 16; kk++) {
                    dst[ch * 16 + kk] = weights[n * K + k_start + kk];
                }
            }
        }
    }
}

/*
 * neon_ghost12_matvec - Ghost-Stream kernel with LDNP + Block-12
 *
 * Uses non-temporal loads (LDNP) for weights to avoid L2 cache pollution.
 * Processes 12 output channels per iteration for maximum activation reuse.
 */
void neon_ghost12_matvec(
    int32_t* __restrict__ out,
    const int8_t* __restrict__ act,
    const int8_t* __restrict__ wgt,  // Ghost-12 layout
    int N,  // Must be multiple of 12
    int K   // Must be multiple of 16
) {
    const int K_blocks = K / 16;
    const int block_stride = 12 * 16;  // 192 bytes
    
    for (int n = 0; n < N; n += 12) {
        int nb = n / 12;
        
        // 12 accumulators in registers v0-v11
        int32x4_t acc0, acc1, acc2, acc3, acc4, acc5;
        int32x4_t acc6, acc7, acc8, acc9, acc10, acc11;
        
        // Zero accumulators via inline asm for precise register control
        __asm__ volatile (
            "movi v0.4s, #0 \n\t"
            "movi v1.4s, #0 \n\t"
            "movi v2.4s, #0 \n\t"
            "movi v3.4s, #0 \n\t"
            "movi v4.4s, #0 \n\t"
            "movi v5.4s, #0 \n\t"
            "movi v6.4s, #0 \n\t"
            "movi v7.4s, #0 \n\t"
            "movi v8.4s, #0 \n\t"
            "movi v9.4s, #0 \n\t"
            "movi v10.4s, #0 \n\t"
            "movi v11.4s, #0 \n\t"
            : "=w"(acc0), "=w"(acc1), "=w"(acc2), "=w"(acc3),
              "=w"(acc4), "=w"(acc5), "=w"(acc6), "=w"(acc7),
              "=w"(acc8), "=w"(acc9), "=w"(acc10), "=w"(acc11)
            :
            :
        );
        
        const int8_t* a_ptr = act;
        const int8_t* w_base = wgt + nb * K_blocks * block_stride;
        
        for (int kb = 0; kb < K_blocks; kb++) {
            const int8_t* w_block = w_base + kb * block_stride;
            
            // Load 16 activations (standard load - keep in cache)
            int8x16_t a_vec;
            __asm__ volatile (
                "ldr q12, [%[a]], #16 \n\t"
                : [a] "+r"(a_ptr), "=w"(a_vec)
                :
                : "memory"
            );
            
            // Ghost-load weights with LDNP (non-temporal) and fire SDOT
            // LDNP loads 2 × 16 bytes = 32 bytes (2 weight vectors)
            // LDNP does NOT support post-increment, use explicit offsets
            __asm__ volatile (
                // Load weights for Ch0-Ch1 (non-temporal), SDOT
                "ldnp q16, q17, [%[w]] \n\t"
                "sdot v0.4s, v12.16b, v16.16b \n\t"
                "sdot v1.4s, v12.16b, v17.16b \n\t"
                
                // Ch2-Ch3
                "ldnp q18, q19, [%[w], #32] \n\t"
                "sdot v2.4s, v12.16b, v18.16b \n\t"
                "sdot v3.4s, v12.16b, v19.16b \n\t"
                
                // Ch4-Ch5
                "ldnp q20, q21, [%[w], #64] \n\t"
                "sdot v4.4s, v12.16b, v20.16b \n\t"
                "sdot v5.4s, v12.16b, v21.16b \n\t"
                
                // Ch6-Ch7
                "ldnp q22, q23, [%[w], #96] \n\t"
                "sdot v6.4s, v12.16b, v22.16b \n\t"
                "sdot v7.4s, v12.16b, v23.16b \n\t"
                
                // Ch8-Ch9
                "ldnp q24, q25, [%[w], #128] \n\t"
                "sdot v8.4s, v12.16b, v24.16b \n\t"
                "sdot v9.4s, v12.16b, v25.16b \n\t"
                
                // Ch10-Ch11
                "ldnp q26, q27, [%[w], #160] \n\t"
                "sdot v10.4s, v12.16b, v26.16b \n\t"
                "sdot v11.4s, v12.16b, v27.16b \n\t"
                
                : "+w"(acc0), "+w"(acc1), "+w"(acc2), "+w"(acc3),
                  "+w"(acc4), "+w"(acc5), "+w"(acc6), "+w"(acc7),
                  "+w"(acc8), "+w"(acc9), "+w"(acc10), "+w"(acc11)
                : [w] "r"(w_block), "w"(a_vec)
                : "v16", "v17", "v18", "v19", "v20", "v21",
                  "v22", "v23", "v24", "v25", "v26", "v27", "memory"
            );
        }
        
        // Horizontal reduction and store
        out[n + 0] = vaddvq_s32(acc0);
        out[n + 1] = vaddvq_s32(acc1);
        out[n + 2] = vaddvq_s32(acc2);
        out[n + 3] = vaddvq_s32(acc3);
        out[n + 4] = vaddvq_s32(acc4);
        out[n + 5] = vaddvq_s32(acc5);
        out[n + 6] = vaddvq_s32(acc6);
        out[n + 7] = vaddvq_s32(acc7);
        out[n + 8] = vaddvq_s32(acc8);
        out[n + 9] = vaddvq_s32(acc9);
        out[n + 10] = vaddvq_s32(acc10);
        out[n + 11] = vaddvq_s32(acc11);
    }
}

/*
 * neon_ghost12_matvec_v2 - Simplified version without complex asm constraints
 *
 * Uses LDNP via inline asm but lets compiler handle register allocation
 * for accumulators.
 */
void neon_ghost12_matvec_v2(
    int32_t* __restrict__ out,
    const int8_t* __restrict__ act,
    const int8_t* __restrict__ wgt,  // Ghost-12 layout
    int N,  // Must be multiple of 12
    int K   // Must be multiple of 16
) {
    const int K_blocks = K / 16;
    const int block_stride = 12 * 16;
    
    for (int n = 0; n < N; n += 12) {
        int nb = n / 12;
        
        // Accumulators (compiler will allocate registers)
        int32x4_t acc0 = vdupq_n_s32(0);
        int32x4_t acc1 = vdupq_n_s32(0);
        int32x4_t acc2 = vdupq_n_s32(0);
        int32x4_t acc3 = vdupq_n_s32(0);
        int32x4_t acc4 = vdupq_n_s32(0);
        int32x4_t acc5 = vdupq_n_s32(0);
        int32x4_t acc6 = vdupq_n_s32(0);
        int32x4_t acc7 = vdupq_n_s32(0);
        int32x4_t acc8 = vdupq_n_s32(0);
        int32x4_t acc9 = vdupq_n_s32(0);
        int32x4_t acc10 = vdupq_n_s32(0);
        int32x4_t acc11 = vdupq_n_s32(0);
        
        const int8_t* a_ptr = act;
        const int8_t* w_base = wgt + nb * K_blocks * block_stride;
        
        for (int kb = 0; kb < K_blocks; kb++) {
            const int8_t* w_ptr = w_base + kb * block_stride;
            
            // Standard activation load (keep in cache)
            int8x16_t a = vld1q_s8(a_ptr);
            a_ptr += 16;
            
            // Non-temporal weight loads using inline asm
            // LDNP does NOT support post-increment, use explicit offsets
            int8x16_t w0, w1, w2, w3, w4, w5, w6, w7, w8, w9, w10, w11;
            
            __asm__ volatile (
                "ldnp %q[w0], %q[w1], [%[ptr]] \n\t"
                "ldnp %q[w2], %q[w3], [%[ptr], #32] \n\t"
                "ldnp %q[w4], %q[w5], [%[ptr], #64] \n\t"
                "ldnp %q[w6], %q[w7], [%[ptr], #96] \n\t"
                "ldnp %q[w8], %q[w9], [%[ptr], #128] \n\t"
                "ldnp %q[w10], %q[w11], [%[ptr], #160] \n\t"
                : [w0] "=w"(w0), [w1] "=w"(w1), [w2] "=w"(w2), [w3] "=w"(w3),
                  [w4] "=w"(w4), [w5] "=w"(w5), [w6] "=w"(w6), [w7] "=w"(w7),
                  [w8] "=w"(w8), [w9] "=w"(w9), [w10] "=w"(w10), [w11] "=w"(w11)
                : [ptr] "r"(w_ptr)
                : "memory"
            );
            
            // SDOT for all 12 channels
            acc0 = vdotq_s32(acc0, w0, a);
            acc1 = vdotq_s32(acc1, w1, a);
            acc2 = vdotq_s32(acc2, w2, a);
            acc3 = vdotq_s32(acc3, w3, a);
            acc4 = vdotq_s32(acc4, w4, a);
            acc5 = vdotq_s32(acc5, w5, a);
            acc6 = vdotq_s32(acc6, w6, a);
            acc7 = vdotq_s32(acc7, w7, a);
            acc8 = vdotq_s32(acc8, w8, a);
            acc9 = vdotq_s32(acc9, w9, a);
            acc10 = vdotq_s32(acc10, w10, a);
            acc11 = vdotq_s32(acc11, w11, a);
        }
        
        out[n + 0] = vaddvq_s32(acc0);
        out[n + 1] = vaddvq_s32(acc1);
        out[n + 2] = vaddvq_s32(acc2);
        out[n + 3] = vaddvq_s32(acc3);
        out[n + 4] = vaddvq_s32(acc4);
        out[n + 5] = vaddvq_s32(acc5);
        out[n + 6] = vaddvq_s32(acc6);
        out[n + 7] = vaddvq_s32(acc7);
        out[n + 8] = vaddvq_s32(acc8);
        out[n + 9] = vaddvq_s32(acc9);
        out[n + 10] = vaddvq_s32(acc10);
        out[n + 11] = vaddvq_s32(acc11);
    }
}

/*
 * neon_ghost12_matvec_ldr - Same as v2 but with standard LDR (temporal) loads
 *
 * This is a control to compare LDNP vs LDR performance.
 */
void neon_ghost12_matvec_ldr(
    int32_t* __restrict__ out,
    const int8_t* __restrict__ act,
    const int8_t* __restrict__ wgt,  // Ghost-12 layout
    int N,  // Must be multiple of 12
    int K   // Must be multiple of 16
) {
    const int K_blocks = K / 16;
    const int block_stride = 12 * 16;
    
    for (int n = 0; n < N; n += 12) {
        int nb = n / 12;
        
        int32x4_t acc0 = vdupq_n_s32(0);
        int32x4_t acc1 = vdupq_n_s32(0);
        int32x4_t acc2 = vdupq_n_s32(0);
        int32x4_t acc3 = vdupq_n_s32(0);
        int32x4_t acc4 = vdupq_n_s32(0);
        int32x4_t acc5 = vdupq_n_s32(0);
        int32x4_t acc6 = vdupq_n_s32(0);
        int32x4_t acc7 = vdupq_n_s32(0);
        int32x4_t acc8 = vdupq_n_s32(0);
        int32x4_t acc9 = vdupq_n_s32(0);
        int32x4_t acc10 = vdupq_n_s32(0);
        int32x4_t acc11 = vdupq_n_s32(0);
        
        const int8_t* a_ptr = act;
        const int8_t* w_base = wgt + nb * K_blocks * block_stride;
        
        for (int kb = 0; kb < K_blocks; kb++) {
            const int8_t* w_ptr = w_base + kb * block_stride;
            
            int8x16_t a = vld1q_s8(a_ptr);
            a_ptr += 16;
            
            // Standard temporal loads
            int8x16_t w0 = vld1q_s8(w_ptr);
            int8x16_t w1 = vld1q_s8(w_ptr + 16);
            int8x16_t w2 = vld1q_s8(w_ptr + 32);
            int8x16_t w3 = vld1q_s8(w_ptr + 48);
            int8x16_t w4 = vld1q_s8(w_ptr + 64);
            int8x16_t w5 = vld1q_s8(w_ptr + 80);
            int8x16_t w6 = vld1q_s8(w_ptr + 96);
            int8x16_t w7 = vld1q_s8(w_ptr + 112);
            int8x16_t w8 = vld1q_s8(w_ptr + 128);
            int8x16_t w9 = vld1q_s8(w_ptr + 144);
            int8x16_t w10 = vld1q_s8(w_ptr + 160);
            int8x16_t w11 = vld1q_s8(w_ptr + 176);
            
            acc0 = vdotq_s32(acc0, w0, a);
            acc1 = vdotq_s32(acc1, w1, a);
            acc2 = vdotq_s32(acc2, w2, a);
            acc3 = vdotq_s32(acc3, w3, a);
            acc4 = vdotq_s32(acc4, w4, a);
            acc5 = vdotq_s32(acc5, w5, a);
            acc6 = vdotq_s32(acc6, w6, a);
            acc7 = vdotq_s32(acc7, w7, a);
            acc8 = vdotq_s32(acc8, w8, a);
            acc9 = vdotq_s32(acc9, w9, a);
            acc10 = vdotq_s32(acc10, w10, a);
            acc11 = vdotq_s32(acc11, w11, a);
        }
        
        out[n + 0] = vaddvq_s32(acc0);
        out[n + 1] = vaddvq_s32(acc1);
        out[n + 2] = vaddvq_s32(acc2);
        out[n + 3] = vaddvq_s32(acc3);
        out[n + 4] = vaddvq_s32(acc4);
        out[n + 5] = vaddvq_s32(acc5);
        out[n + 6] = vaddvq_s32(acc6);
        out[n + 7] = vaddvq_s32(acc7);
        out[n + 8] = vaddvq_s32(acc8);
        out[n + 9] = vaddvq_s32(acc9);
        out[n + 10] = vaddvq_s32(acc10);
        out[n + 11] = vaddvq_s32(acc11);
    }
}

/*
 * pack_weights_block16 - Weight layout for Block-16 kernel
 *
 * 16 output channels × 16 K-elements = 256 bytes per block
 */
void pack_weights_block16(
    int8_t* __restrict__ packed,
    const int8_t* __restrict__ weights,
    int N,  // Must be multiple of 16
    int K   // Must be multiple of 16
) {
    const int K_blocks = K / 16;
    const int N_blocks = N / 16;
    const int block_stride = 16 * 16;  // 256 bytes
    
    for (int nb = 0; nb < N_blocks; nb++) {
        for (int kb = 0; kb < K_blocks; kb++) {
            int8_t* dst = packed + (nb * K_blocks + kb) * block_stride;
            
            for (int ch = 0; ch < 16; ch++) {
                int n = nb * 16 + ch;
                int k_start = kb * 16;
                
                for (int kk = 0; kk < 16; kk++) {
                    dst[ch * 16 + kk] = weights[n * K + k_start + kk];
                }
            }
        }
    }
}

/*
 * neon_block16_matvec - Block-16 kernel maximizing register usage
 *
 * Uses all 32 registers:
 *   - 16 accumulators (v0-v15)
 *   - 1 activation (v16)
 *   - 15 temps for weights (loaded in batches)
 */
void neon_block16_matvec(
    int32_t* __restrict__ out,
    const int8_t* __restrict__ act,
    const int8_t* __restrict__ wgt,  // Block-16 layout
    int N,  // Must be multiple of 16
    int K   // Must be multiple of 16
) {
    const int K_blocks = K / 16;
    const int block_stride = 16 * 16;
    
    for (int n = 0; n < N; n += 16) {
        int nb = n / 16;
        
        // 16 accumulators - max practical limit
        int32x4_t acc0 = vdupq_n_s32(0);
        int32x4_t acc1 = vdupq_n_s32(0);
        int32x4_t acc2 = vdupq_n_s32(0);
        int32x4_t acc3 = vdupq_n_s32(0);
        int32x4_t acc4 = vdupq_n_s32(0);
        int32x4_t acc5 = vdupq_n_s32(0);
        int32x4_t acc6 = vdupq_n_s32(0);
        int32x4_t acc7 = vdupq_n_s32(0);
        int32x4_t acc8 = vdupq_n_s32(0);
        int32x4_t acc9 = vdupq_n_s32(0);
        int32x4_t acc10 = vdupq_n_s32(0);
        int32x4_t acc11 = vdupq_n_s32(0);
        int32x4_t acc12 = vdupq_n_s32(0);
        int32x4_t acc13 = vdupq_n_s32(0);
        int32x4_t acc14 = vdupq_n_s32(0);
        int32x4_t acc15 = vdupq_n_s32(0);
        
        const int8_t* a_ptr = act;
        const int8_t* w_base = wgt + nb * K_blocks * block_stride;
        
        for (int kb = 0; kb < K_blocks; kb++) {
            const int8_t* w_ptr = w_base + kb * block_stride;
            
            // Prefetch next block
            if (kb + 1 < K_blocks) {
                __builtin_prefetch(a_ptr + 16, 0, 3);
                __builtin_prefetch(w_base + (kb + 1) * block_stride, 0, 3);
                __builtin_prefetch(w_base + (kb + 1) * block_stride + 128, 0, 3);
            }
            
            int8x16_t a = vld1q_s8(a_ptr);
            a_ptr += 16;
            
            // Load all 16 weight vectors
            int8x16_t w0 = vld1q_s8(w_ptr);
            int8x16_t w1 = vld1q_s8(w_ptr + 16);
            int8x16_t w2 = vld1q_s8(w_ptr + 32);
            int8x16_t w3 = vld1q_s8(w_ptr + 48);
            int8x16_t w4 = vld1q_s8(w_ptr + 64);
            int8x16_t w5 = vld1q_s8(w_ptr + 80);
            int8x16_t w6 = vld1q_s8(w_ptr + 96);
            int8x16_t w7 = vld1q_s8(w_ptr + 112);
            int8x16_t w8 = vld1q_s8(w_ptr + 128);
            int8x16_t w9 = vld1q_s8(w_ptr + 144);
            int8x16_t w10 = vld1q_s8(w_ptr + 160);
            int8x16_t w11 = vld1q_s8(w_ptr + 176);
            int8x16_t w12 = vld1q_s8(w_ptr + 192);
            int8x16_t w13 = vld1q_s8(w_ptr + 208);
            int8x16_t w14 = vld1q_s8(w_ptr + 224);
            int8x16_t w15 = vld1q_s8(w_ptr + 240);
            
            // SDOT all 16 channels
            acc0 = vdotq_s32(acc0, w0, a);
            acc1 = vdotq_s32(acc1, w1, a);
            acc2 = vdotq_s32(acc2, w2, a);
            acc3 = vdotq_s32(acc3, w3, a);
            acc4 = vdotq_s32(acc4, w4, a);
            acc5 = vdotq_s32(acc5, w5, a);
            acc6 = vdotq_s32(acc6, w6, a);
            acc7 = vdotq_s32(acc7, w7, a);
            acc8 = vdotq_s32(acc8, w8, a);
            acc9 = vdotq_s32(acc9, w9, a);
            acc10 = vdotq_s32(acc10, w10, a);
            acc11 = vdotq_s32(acc11, w11, a);
            acc12 = vdotq_s32(acc12, w12, a);
            acc13 = vdotq_s32(acc13, w13, a);
            acc14 = vdotq_s32(acc14, w14, a);
            acc15 = vdotq_s32(acc15, w15, a);
        }
        
        out[n + 0] = vaddvq_s32(acc0);
        out[n + 1] = vaddvq_s32(acc1);
        out[n + 2] = vaddvq_s32(acc2);
        out[n + 3] = vaddvq_s32(acc3);
        out[n + 4] = vaddvq_s32(acc4);
        out[n + 5] = vaddvq_s32(acc5);
        out[n + 6] = vaddvq_s32(acc6);
        out[n + 7] = vaddvq_s32(acc7);
        out[n + 8] = vaddvq_s32(acc8);
        out[n + 9] = vaddvq_s32(acc9);
        out[n + 10] = vaddvq_s32(acc10);
        out[n + 11] = vaddvq_s32(acc11);
        out[n + 12] = vaddvq_s32(acc12);
        out[n + 13] = vaddvq_s32(acc13);
        out[n + 14] = vaddvq_s32(acc14);
        out[n + 15] = vaddvq_s32(acc15);
    }
}

/*
 * Aligned memory allocation helper for huge pages (16KB alignment)
 */
void* alloc_aligned_16k(size_t size) {
    void* ptr = NULL;
    if (posix_memalign(&ptr, 16384, size) != 0) {
        return NULL;
    }
    return ptr;
}

#endif  // __ARM_FEATURE_DOTPROD

