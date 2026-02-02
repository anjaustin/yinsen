/*
 * ACTIVATION_Q15 — Fixed-Point Nonlinear Primitives (Q15)
 *
 * Integer-only sigmoid and tanh via 256-entry LUT + linear interpolation.
 * Zero floating-point operations. Designed for RV32IMAC (no FPU).
 *
 * Q15 format: int16_t where 1.0 = 32767 (0x7FFF), -1.0 = -32768 (0x8000).
 * Multiply: (int32_t)a * b >> 15 (uses hardware MUL from M extension).
 *
 * Input range: [-8.0, +8.0] mapped to Q12.3 fixed-point (int16_t).
 * Q12.3: 1.0 = 8, so +-8.0 = +-64. Full range fits in int16_t.
 * BUT: for pre-activation values, we need more range and precision.
 * We use Q4.11: 1.0 = 2048, range [-16.0, +16.0). Adequate for
 * pre-activations (bias + sparse dot) which typically land in [-8, +8].
 *
 * Activation outputs:
 *   sigmoid: [0, 1] in Q15 → [0, 32767]
 *   tanh:    [-1, 1] in Q15 → [-32768, 32767]
 *
 * LUT domain: input in Q4.11 format, covering [-8.0, +8.0].
 *   Index = (input_q11 + 8*2048) * 255 / (16*2048)
 *         = (input_q11 + 16384) * 255 / 32768
 * We precompute this as a shift+multiply to avoid division.
 *
 * Tables: 256 entries * 2 bytes = 512 bytes per function, 1024 bytes total.
 * Half the size of the float LUT. Stays hot in L1.
 *
 * Created: February 2026
 * Part of the Yinsen Sentinel fixed-point compute stack.
 */

#ifndef TRIX_ACTIVATION_Q15_H
#define TRIX_ACTIVATION_Q15_H

#include <stdint.h>
#include <string.h>
#include <math.h>  /* For Q15_LUT_INIT only (one-time init, not hot path) */

#ifdef __cplusplus
extern "C" {
#endif

/* ═══════════════════════════════════════════════════════════════════════════
 * Q15 ARITHMETIC PRIMITIVES
 * ═══════════════════════════════════════════════════════════════════════════ */

/* Q15: 1.0 = 32767, -1.0 = -32768 */
#define Q15_ONE      ((int16_t)32767)
#define Q15_NEG_ONE  ((int16_t)-32768)
#define Q15_HALF     ((int16_t)16384)
#define Q15_ZERO     ((int16_t)0)

/* Q15 multiply: result = (a * b) >> 15, with rounding */
static inline int16_t q15_mul(int16_t a, int16_t b) {
    int32_t product = (int32_t)a * (int32_t)b;
    /* Round to nearest: add 0.5 ULP before shift */
    return (int16_t)((product + (1 << 14)) >> 15);
}

/* Q15 multiply returning int32_t (for accumulation before final shift) */
static inline int32_t q15_mul32(int16_t a, int16_t b) {
    return (int32_t)a * (int32_t)b;
}

/* Saturating add */
static inline int16_t q15_sat_add(int16_t a, int16_t b) {
    int32_t sum = (int32_t)a + (int32_t)b;
    if (sum > 32767) return 32767;
    if (sum < -32768) return -32768;
    return (int16_t)sum;
}

/* Saturating subtract */
static inline int16_t q15_sat_sub(int16_t a, int16_t b) {
    int32_t diff = (int32_t)a - (int32_t)b;
    if (diff > 32767) return 32767;
    if (diff < -32768) return -32768;
    return (int16_t)diff;
}

/* ═══════════════════════════════════════════════════════════════════════════
 * Q4.11 FORMAT — Pre-activation values
 *
 * Used for: bias + sparse dot product results, before activation.
 * 1.0 = 2048 (1 << 11). Range: [-16.0, +16.0). Precision: ~0.00049.
 * Adequate for pre-activations which typically land in [-8, +8].
 * ═══════════════════════════════════════════════════════════════════════════ */

#define Q11_SHIFT  11
#define Q11_ONE    (1 << Q11_SHIFT)  /* 2048 */

/* Convert float to Q4.11 (for init/test only — NOT in hot path) */
static inline int16_t float_to_q11(float x) {
    int32_t v = (int32_t)(x * (float)Q11_ONE + (x >= 0 ? 0.5f : -0.5f));
    if (v > 32767) return 32767;
    if (v < -32768) return -32768;
    return (int16_t)v;
}

/* Convert float to Q15 (for init/test only — NOT in hot path) */
static inline int16_t float_to_q15(float x) {
    int32_t v = (int32_t)(x * 32768.0f + (x >= 0 ? 0.5f : -0.5f));
    if (v > 32767) return 32767;
    if (v < -32768) return -32768;
    return (int16_t)v;
}

/* Convert Q15 to float (for test/debug only — NOT in hot path) */
static inline float q15_to_float(int16_t x) {
    return (float)x / 32768.0f;
}

/* Convert Q4.11 to float (for test/debug only — NOT in hot path) */
static inline float q11_to_float(int16_t x) {
    return (float)x / (float)Q11_ONE;
}

/* ═══════════════════════════════════════════════════════════════════════════
 * Q15 LUT TABLES — 256 entries, 512 bytes each
 *
 * Domain: input in Q4.11 covering [-8.0, +8.0]
 * Sigmoid output: Q15 [0, 32767] representing [0.0, 1.0)
 * Tanh output: Q15 [-32768, 32767] representing [-1.0, 1.0)
 *
 * Index computation (integer-only):
 *   x_shifted = input_q11 + 8*2048 = input_q11 + 16384
 *   x_shifted is in [0, 32768] for inputs in [-8, +8]
 *   idx_full = x_shifted * 255 / 32768
 *   We use: idx_full = (x_shifted * 255 + 16384) >> 15
 *   idx = idx_full >> 8  (integer part)
 *   frac = idx_full & 0xFF (fractional part, 0-255, for lerp)
 *
 * Actually simpler: since range is exactly 16.0 in Q11 = 32768,
 * and we want 256 bins, each bin is 32768/256 = 128 Q11 units.
 *   idx_raw = x_shifted (0..32768)
 *   idx = idx_raw >> 7  (divide by 128 → 0..256)
 *   frac = (idx_raw >> 0) & 0x7F  (lower 7 bits → 0..127)
 *
 * Lerp: result = table[idx] + (frac * (table[idx+1] - table[idx])) >> 7
 * ═══════════════════════════════════════════════════════════════════════════ */

#define Q15_LUT_SIZE    256
#define Q15_LUT_XMIN_Q11  (-8 * Q11_ONE)   /* -16384 in Q4.11 */
#define Q15_LUT_XMAX_Q11  ( 8 * Q11_ONE)   /*  16384 in Q4.11 */
#define Q15_LUT_BIN_SHIFT  7               /* 32768 / 256 = 128 = 1<<7 */

/* Tables — static per translation unit (header-only, linker folds duplicates) */
static int16_t _sigmoid_lut_q15[Q15_LUT_SIZE + 1]; /* +1 for lerp guard */
static int16_t _tanh_lut_q15[Q15_LUT_SIZE + 1];
static int _q15_lut_ready = 0;

/**
 * Q15_LUT_INIT — Fill lookup tables. Call once at startup.
 *
 * This is the ONLY function that uses floating-point. It runs once
 * at init time (not in the hot path). On the target MCU, this costs
 * ~50 us of soft-float math at startup. Negligible.
 *
 * Idempotent: safe to call multiple times.
 */
static inline void Q15_LUT_INIT(void) {
    if (_q15_lut_ready) return;
    for (int i = 0; i <= Q15_LUT_SIZE; i++) {
        /* Map index to x value: x = -8.0 + i * 16.0/256 */
        float x = -8.0f + (float)i * (16.0f / 256.0f);

        /* Sigmoid: [0, 1] → Q15 [0, 32767] */
        float s = 1.0f / (1.0f + expf(-x));
        int32_t sq = (int32_t)(s * 32768.0f);
        if (sq > 32767) sq = 32767;
        if (sq < 0) sq = 0;
        _sigmoid_lut_q15[i] = (int16_t)sq;

        /* Tanh: [-1, 1] → Q15 [-32768, 32767] */
        float t = tanhf(x);
        int32_t tq = (int32_t)(t * 32768.0f);
        if (tq > 32767) tq = 32767;
        if (tq < -32768) tq = -32768;
        _tanh_lut_q15[i] = (int16_t)tq;
    }
    _q15_lut_ready = 1;
}

/**
 * Q15_LUT_INIT_CONST — Static initialization without float.
 *
 * Alternative: precompute the tables at build time and embed as const arrays.
 * Use this when even init-time float is unacceptable. Pass the tables as
 * parameters instead of using the static globals.
 *
 * Tables can be generated with:
 *   for i in range(257):
 *       x = -8.0 + i * 16.0/256.0
 *       print(f"  {int(1/(1+math.exp(-x)) * 32768):6d},")
 */

/**
 * SIGMOID_Q15 — Integer-only sigmoid via LUT + lerp
 *
 * Input:  pre-activation in Q4.11 (int16_t, 1.0 = 2048)
 * Output: sigmoid in Q15 (int16_t, 1.0 = 32767)
 *
 * Zero floating-point operations. ~10 integer instructions.
 *
 * Max error vs libm sigmoid: < 2^-7 in Q15 (~0.008)
 * Equivalent float error: ~0.00025
 */
static inline int16_t SIGMOID_Q15(int16_t x_q11) {
    /* Clamp to LUT range */
    if (x_q11 <= Q15_LUT_XMIN_Q11) return _sigmoid_lut_q15[0];
    if (x_q11 >= Q15_LUT_XMAX_Q11) return _sigmoid_lut_q15[Q15_LUT_SIZE];

    /* Shift to unsigned range: [0, 32768] */
    int32_t x_shifted = (int32_t)x_q11 - Q15_LUT_XMIN_Q11;

    /* Split into index (high bits) and fraction (low 7 bits) */
    int idx = (int)(x_shifted >> Q15_LUT_BIN_SHIFT);
    int frac = (int)(x_shifted & ((1 << Q15_LUT_BIN_SHIFT) - 1));

    /* Guard: clamp index (shouldn't be needed after range check, but safe) */
    if (idx >= Q15_LUT_SIZE) return _sigmoid_lut_q15[Q15_LUT_SIZE];

    /* Linear interpolation: y0 + frac * (y1 - y0) >> 7 */
    int32_t y0 = _sigmoid_lut_q15[idx];
    int32_t dy = _sigmoid_lut_q15[idx + 1] - y0;
    return (int16_t)(y0 + ((dy * frac) >> Q15_LUT_BIN_SHIFT));
}

/**
 * TANH_Q15 — Integer-only tanh via LUT + lerp
 *
 * Input:  pre-activation in Q4.11 (int16_t, 1.0 = 2048)
 * Output: tanh in Q15 (int16_t, 1.0 = 32767)
 *
 * Zero floating-point operations. ~10 integer instructions.
 */
static inline int16_t TANH_Q15(int16_t x_q11) {
    /* Clamp to LUT range */
    if (x_q11 <= Q15_LUT_XMIN_Q11) return _tanh_lut_q15[0];
    if (x_q11 >= Q15_LUT_XMAX_Q11) return _tanh_lut_q15[Q15_LUT_SIZE];

    /* Shift to unsigned range: [0, 32768] */
    int32_t x_shifted = (int32_t)x_q11 - Q15_LUT_XMIN_Q11;

    /* Split into index and fraction */
    int idx = (int)(x_shifted >> Q15_LUT_BIN_SHIFT);
    int frac = (int)(x_shifted & ((1 << Q15_LUT_BIN_SHIFT) - 1));

    if (idx >= Q15_LUT_SIZE) return _tanh_lut_q15[Q15_LUT_SIZE];

    /* Lerp */
    int32_t y0 = _tanh_lut_q15[idx];
    int32_t dy = _tanh_lut_q15[idx + 1] - y0;
    return (int16_t)(y0 + ((dy * frac) >> Q15_LUT_BIN_SHIFT));
}

#ifdef __cplusplus
}
#endif

#endif /* TRIX_ACTIVATION_Q15_H */
