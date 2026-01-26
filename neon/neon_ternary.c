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
 * pack_weights_k_vertical - Pack weights into K-vertical format
 *
 * Input: weights[N, K] with values -1, 0, +1
 * Output: packed[N, K/4] with 4 trits per byte
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
