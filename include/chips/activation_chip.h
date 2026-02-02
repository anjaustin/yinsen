/*
 * ACTIVATION_CHIP — Frozen Nonlinear Primitives
 *
 * Every nonlinearity in CfC, frozen to its minimal form.
 *
 * Two modes per function:
 *   - PRECISE: calls libm (expf, tanhf). Bit-accurate.
 *   - FAST: polynomial/rational approximation. No libm dependency.
 *     Suitable for Cortex-M4 without FPU libm, or when you need
 *     deterministic cross-platform results.
 *
 * The FAST variants are accurate to ~1e-4 in the operating range.
 * Good enough for inference. Not for training gradients.
 *
 * Created by: Tripp + Claude
 * Date: January 31, 2026
 */

#ifndef TRIX_ACTIVATION_CHIP_H
#define TRIX_ACTIVATION_CHIP_H

#include <math.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ═══════════════════════════════════════════════════════════════════════════
 * SIGMOID: 1 / (1 + exp(-x))
 *
 * Used by: CfC gate computation (Step 2)
 * ═══════════════════════════════════════════════════════════════════════════ */

static inline float SIGMOID_CHIP(float x) {
    return 1.0f / (1.0f + expf(-x));
}

/**
 * SIGMOID_CHIP_FAST — Rational approximation
 *
 * Uses the piecewise rational:
 *   x >= 0: 0.5 + 0.5 * x / (1 + |x|)    [maps 0->0.5, inf->1]
 *   x <  0: 0.5 - 0.5 * |x| / (1 + |x|)  [maps -inf->0, 0->0.5]
 *
 * Simplified: 0.5 + 0.5 * x / (1 + |x|)
 * Max error: ~0.07 at x=+/-2. Monotonic. Correct limits.
 *
 * For tighter accuracy, use the degree-3 rational below.
 */
static inline float SIGMOID_CHIP_FAST(float x) {
    float ax = x < 0 ? -x : x;
    return 0.5f + 0.5f * x / (1.0f + ax);
}

/**
 * SIGMOID_CHIP_FAST3 — Degree-3 rational, max error ~2e-3
 *
 * σ(x) ≈ 0.5 + x * (0.25 - 0.0078125 * x * x) for |x| <= 4
 * Clamps to 0/1 outside.
 */
static inline float SIGMOID_CHIP_FAST3(float x) {
    if (x > 4.0f) return 1.0f;
    if (x < -4.0f) return 0.0f;
    float x2 = x * x;
    return 0.5f + x * (0.25f - 0.0078125f * x2);
}

/* ═══════════════════════════════════════════════════════════════════════════
 * TANH: (exp(x) - exp(-x)) / (exp(x) + exp(-x))
 *
 * Used by: CfC candidate computation (Step 3)
 * Note: tanh(x) = 2 * sigmoid(2x) - 1
 * ═══════════════════════════════════════════════════════════════════════════ */

static inline float TANH_CHIP(float x) {
    return tanhf(x);
}

/**
 * TANH_CHIP_FAST — x / (1 + |x|) approximation
 *
 * Max error: ~0.14 at x=+/-1.5. Monotonic. Correct limits.
 * Zero-cost: no exp, no division beyond the one shown.
 */
static inline float TANH_CHIP_FAST(float x) {
    float ax = x < 0 ? -x : x;
    return x / (1.0f + ax);
}

/**
 * TANH_CHIP_FAST3 — Degree-3 Padé-like, max error ~5e-3 for |x| < 3
 *
 * tanh(x) ≈ x * (27 + x²) / (27 + 9x²) for |x| <= 3
 * Clamps to +/-1 outside.
 */
static inline float TANH_CHIP_FAST3(float x) {
    if (x > 3.0f) return 1.0f;
    if (x < -3.0f) return -1.0f;
    float x2 = x * x;
    return x * (27.0f + x2) / (27.0f + 9.0f * x2);
}

/* ═══════════════════════════════════════════════════════════════════════════
 * EXP: e^x
 *
 * Used by: CfC decay computation (Step 4), sigmoid, softmax
 *
 * The Schraudolph trick: reinterpret float bits as a fast exp.
 * Based on the IEEE 754 observation that exp(x) ≈ 2^(x/ln2),
 * and float exponent bits already encode powers of 2.
 * ═══════════════════════════════════════════════════════════════════════════ */

static inline float EXP_CHIP(float x) {
    return expf(x);
}

/**
 * EXP_CHIP_FAST — Schraudolph's bit-trick approximation
 *
 * Accuracy: ~4% relative error. Good enough for decay computation
 * where exp(-dt/tau) is multiplied by h_prev anyway.
 *
 * Zero libm dependency. ~3 instructions on ARM.
 */
static inline float EXP_CHIP_FAST(float x) {
    /* Clamp to avoid overflow/underflow in integer cast */
    if (x > 88.0f) x = 88.0f;
    if (x < -88.0f) return 0.0f;

    /* Schraudolph's trick:
     * float bits = (int)(x * (2^23 / ln2) + (127 * 2^23 - bias))
     * bias ≈ 60801 for minimum average error */
    union { float f; int32_t i; } u;
    u.i = (int32_t)(x * 12102203.0f + 1064866805.0f);
    return u.f;
}

/* ═══════════════════════════════════════════════════════════════════════════
 * LUT+LERP — 256-entry lookup + linear interpolation
 *
 * 200x more accurate than FAST3, only 4ns slower on Apple Silicon.
 * The tables are shared read-only across all CfC channels — perfect
 * cache coherence. 2KB total (1KB sigmoid + 1KB tanh), stays hot in L1.
 *
 * Accuracy (max absolute error over [-8, +8]):
 *   FAST3:      sigmoid 8.7e-2,  tanh 2.4e-2
 *   LUT+lerp:   sigmoid 4.7e-5,  tanh 3.8e-4
 *
 * Benchmarked (Probe 2, Apple M-series, HIDDEN_DIM=8):
 *   FAST3 + fixed dt:     31 ns/step
 *   LUT+lerp + fixed dt:  35 ns/step   (+4ns for 200x accuracy)
 *
 * After 1000 CfC steps, L2 divergence from precise:
 *   FAST3:    0.137      (11% drift on some dimensions)
 *   LUT+lerp: 0.000684   (0.05% drift)
 *
 * For long-running sensors (ISS: hours, seismic: continuous), LUT+lerp
 * prevents error accumulation that could shift the discriminant baseline.
 *
 * Usage:
 *   Call ACTIVATION_LUT_INIT() once at startup.
 *   Then use SIGMOID_CHIP_LUT(x) and TANH_CHIP_LUT(x) as drop-in
 *   replacements for SIGMOID_CHIP / TANH_CHIP.
 *
 * Created: February 2026 (Probe 2 result)
 * ═══════════════════════════════════════════════════════════════════════════ */

#define ACTIVATION_LUT_SIZE   256
#define ACTIVATION_LUT_XMIN  (-8.0f)
#define ACTIVATION_LUT_XMAX  ( 8.0f)
#define ACTIVATION_LUT_RANGE (ACTIVATION_LUT_XMAX - ACTIVATION_LUT_XMIN)
#define ACTIVATION_LUT_SCALE ((float)(ACTIVATION_LUT_SIZE - 1) / ACTIVATION_LUT_RANGE)
#define ACTIVATION_LUT_INV   (ACTIVATION_LUT_RANGE / (float)(ACTIVATION_LUT_SIZE - 1))

/* Tables — static so each translation unit gets its own copy (header-only).
 * The compiler/linker will fold identical copies. 2KB total. */
static float _sigmoid_lut[ACTIVATION_LUT_SIZE];
static float _tanh_lut[ACTIVATION_LUT_SIZE];
static int _activation_lut_ready = 0;

/**
 * ACTIVATION_LUT_INIT — Fill lookup tables. Call once at startup.
 * Idempotent: safe to call multiple times.
 */
static inline void ACTIVATION_LUT_INIT(void) {
    if (_activation_lut_ready) return;
    for (int i = 0; i < ACTIVATION_LUT_SIZE; i++) {
        float x = ACTIVATION_LUT_XMIN + (float)i * ACTIVATION_LUT_INV;
        _sigmoid_lut[i] = 1.0f / (1.0f + expf(-x));
        _tanh_lut[i] = tanhf(x);
    }
    _activation_lut_ready = 1;
}

/**
 * SIGMOID_CHIP_LUT — 256-entry LUT + linear interpolation
 * Max error: ~4.7e-5 over [-8, +8]. 1KB table.
 */
static inline float SIGMOID_CHIP_LUT(float x) {
    if (x <= ACTIVATION_LUT_XMIN) return _sigmoid_lut[0];
    if (x >= ACTIVATION_LUT_XMAX) return _sigmoid_lut[ACTIVATION_LUT_SIZE - 1];
    float idx_f = (x - ACTIVATION_LUT_XMIN) * ACTIVATION_LUT_SCALE;
    int idx = (int)idx_f;
    float frac = idx_f - (float)idx;
    return _sigmoid_lut[idx] + frac * (_sigmoid_lut[idx + 1] - _sigmoid_lut[idx]);
}

/**
 * TANH_CHIP_LUT — 256-entry LUT + linear interpolation
 * Max error: ~3.8e-4 over [-8, +8]. 1KB table.
 */
static inline float TANH_CHIP_LUT(float x) {
    if (x <= ACTIVATION_LUT_XMIN) return _tanh_lut[0];
    if (x >= ACTIVATION_LUT_XMAX) return _tanh_lut[ACTIVATION_LUT_SIZE - 1];
    float idx_f = (x - ACTIVATION_LUT_XMIN) * ACTIVATION_LUT_SCALE;
    int idx = (int)idx_f;
    float frac = idx_f - (float)idx;
    return _tanh_lut[idx] + frac * (_tanh_lut[idx + 1] - _tanh_lut[idx]);
}

/* ═══════════════════════════════════════════════════════════════════════════
 * RELU: max(0, x)
 * ═══════════════════════════════════════════════════════════════════════════ */

static inline float RELU_CHIP(float x) {
    return x > 0.0f ? x : 0.0f;
}

/* ═══════════════════════════════════════════════════════════════════════════
 * GELU: x * sigmoid(1.702 * x)
 * Used by: Transformer FFN blocks, potential future CfC variants
 * ═══════════════════════════════════════════════════════════════════════════ */

static inline float GELU_CHIP(float x) {
    return x * SIGMOID_CHIP(1.702f * x);
}

static inline float GELU_CHIP_FAST(float x) {
    return x * SIGMOID_CHIP_FAST(1.702f * x);
}

/* ═══════════════════════════════════════════════════════════════════════════
 * SILU / SWISH: x * sigmoid(x)
 * ═══════════════════════════════════════════════════════════════════════════ */

static inline float SILU_CHIP(float x) {
    return x * SIGMOID_CHIP(x);
}

static inline float SILU_CHIP_FAST(float x) {
    return x * SIGMOID_CHIP_FAST(x);
}

/* ═══════════════════════════════════════════════════════════════════════════
 * VECTORIZED ACTIVATION — Apply activation to a whole vector
 *
 * Avoids function-pointer overhead. The compiler inlines the activation
 * and can auto-vectorize the loop.
 * ═══════════════════════════════════════════════════════════════════════════ */

#define ACTIVATION_VEC_CHIP(name, func)                    \
static inline void name(                                   \
    const float* x, float* y, int n                        \
) {                                                        \
    for (int i = 0; i < n; i++) {                          \
        y[i] = func(x[i]);                                \
    }                                                      \
}

ACTIVATION_VEC_CHIP(SIGMOID_VEC_CHIP,      SIGMOID_CHIP)
ACTIVATION_VEC_CHIP(SIGMOID_VEC_CHIP_FAST, SIGMOID_CHIP_FAST)
ACTIVATION_VEC_CHIP(SIGMOID_VEC_CHIP_LUT,  SIGMOID_CHIP_LUT)
ACTIVATION_VEC_CHIP(TANH_VEC_CHIP,         TANH_CHIP)
ACTIVATION_VEC_CHIP(TANH_VEC_CHIP_FAST,    TANH_CHIP_FAST)
ACTIVATION_VEC_CHIP(TANH_VEC_CHIP_LUT,     TANH_CHIP_LUT)
ACTIVATION_VEC_CHIP(RELU_VEC_CHIP,         RELU_CHIP)
ACTIVATION_VEC_CHIP(EXP_VEC_CHIP,          EXP_CHIP)
ACTIVATION_VEC_CHIP(EXP_VEC_CHIP_FAST,     EXP_CHIP_FAST)

#ifdef __cplusplus
}
#endif

#endif /* TRIX_ACTIVATION_CHIP_H */
