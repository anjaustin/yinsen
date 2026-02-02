/*
 * TERNARY_DOT_CHIP — Frozen Ternary Dot Product Primitive
 *
 * y = sum(x[i] where w[i]=+1) - sum(x[i] where w[i]=-1)
 *
 * No multiplication. Just conditional add/subtract.
 * This is the core instruction that makes ternary neural networks
 * possible on hardware without a multiplier.
 *
 * Encoding (2 bits per trit):
 *   00 = 0  (skip)
 *   01 = +1 (add)
 *   10 = -1 (subtract)
 *   11 = reserved (skip)
 *
 * Three variants:
 *   FLOAT:  Float activations, 2-bit weights → float accumulator
 *   INT8:   Int8 activations, 2-bit weights → int32 accumulator
 *   INT16:  Int16 activations, 2-bit weights → int32 accumulator
 *
 * Created by: Tripp + Claude
 * Date: January 31, 2026
 */

#ifndef TRIX_TERNARY_DOT_CHIP_H
#define TRIX_TERNARY_DOT_CHIP_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * TERNARY_DOT_CHIP — Float activations, 2-bit weights
 *
 * The workhorse. Used in CfC ternary cell for gate and candidate.
 */
static inline float TERNARY_DOT_CHIP(
    const uint8_t* w_packed,
    const float* x,
    int n
) {
    float sum = 0.0f;

    /* Process 4 trits per byte */
    int full_bytes = n / 4;
    int remainder = n & 3;

    for (int b = 0; b < full_bytes; b++) {
        uint8_t packed = w_packed[b];
        const float* xp = x + b * 4;

        /* Unrolled: extract each 2-bit trit and accumulate */
        uint8_t t0 = packed & 0x03;
        uint8_t t1 = (packed >> 2) & 0x03;
        uint8_t t2 = (packed >> 4) & 0x03;
        uint8_t t3 = (packed >> 6) & 0x03;

        if (t0 == 1) sum += xp[0]; else if (t0 == 2) sum -= xp[0];
        if (t1 == 1) sum += xp[1]; else if (t1 == 2) sum -= xp[1];
        if (t2 == 1) sum += xp[2]; else if (t2 == 2) sum -= xp[2];
        if (t3 == 1) sum += xp[3]; else if (t3 == 2) sum -= xp[3];
    }

    /* Handle remainder */
    if (remainder > 0) {
        uint8_t packed = w_packed[full_bytes];
        const float* xp = x + full_bytes * 4;
        for (int i = 0; i < remainder; i++) {
            uint8_t t = (packed >> (i * 2)) & 0x03;
            if (t == 1) sum += xp[i]; else if (t == 2) sum -= xp[i];
        }
    }

    return sum;
}

/**
 * TERNARY_DOT_INT8_CHIP — Int8 activations, 2-bit weights
 *
 * Pure integer path. No float at all.
 * This is what NEON SDOT accelerates.
 */
static inline int32_t TERNARY_DOT_INT8_CHIP(
    const uint8_t* w_packed,
    const int8_t* x,
    int n
) {
    int32_t sum = 0;

    int full_bytes = n / 4;
    int remainder = n & 3;

    for (int b = 0; b < full_bytes; b++) {
        uint8_t packed = w_packed[b];
        const int8_t* xp = x + b * 4;

        uint8_t t0 = packed & 0x03;
        uint8_t t1 = (packed >> 2) & 0x03;
        uint8_t t2 = (packed >> 4) & 0x03;
        uint8_t t3 = (packed >> 6) & 0x03;

        if (t0 == 1) sum += xp[0]; else if (t0 == 2) sum -= xp[0];
        if (t1 == 1) sum += xp[1]; else if (t1 == 2) sum -= xp[1];
        if (t2 == 1) sum += xp[2]; else if (t2 == 2) sum -= xp[2];
        if (t3 == 1) sum += xp[3]; else if (t3 == 2) sum -= xp[3];
    }

    if (remainder > 0) {
        uint8_t packed = w_packed[full_bytes];
        const int8_t* xp = x + full_bytes * 4;
        for (int i = 0; i < remainder; i++) {
            uint8_t t = (packed >> (i * 2)) & 0x03;
            if (t == 1) sum += xp[i]; else if (t == 2) sum -= xp[i];
        }
    }

    return sum;
}

/**
 * TERNARY_MATVEC_CHIP — Ternary matrix-vector: y[M] = W[M,N] @ x[N]
 */
static inline void TERNARY_MATVEC_CHIP(
    const uint8_t* W_packed,
    const float* x,
    float* y,
    int M, int N
) {
    int bytes_per_row = (N + 3) / 4;
    for (int i = 0; i < M; i++) {
        y[i] = TERNARY_DOT_CHIP(W_packed + i * bytes_per_row, x, N);
    }
}

/**
 * TERNARY_MATVEC_BIAS_CHIP — y[M] = W[M,N] @ x[N] + bias[M]
 */
static inline void TERNARY_MATVEC_BIAS_CHIP(
    const uint8_t* W_packed,
    const float* x,
    const float* bias,
    float* y,
    int M, int N
) {
    int bytes_per_row = (N + 3) / 4;
    for (int i = 0; i < M; i++) {
        y[i] = TERNARY_DOT_CHIP(W_packed + i * bytes_per_row, x, N) + bias[i];
    }
}

#ifdef __cplusplus
}
#endif

#endif /* TRIX_TERNARY_DOT_CHIP_H */
