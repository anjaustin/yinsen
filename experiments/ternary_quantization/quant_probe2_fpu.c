/*
 * Ternary Quantization Probe 2 — FPU Optimization
 *
 * Probe 1 established: ternary-as-float (Strategy C) runs at 44ns via
 * standard GEMM. The FPU multiplies by 1.0 as fast as by 0.3.
 *
 * This probe asks: what else can we eliminate from the CfC hot path?
 *
 * CfC step breakdown (HIDDEN_DIM=8, CONCAT_DIM=10):
 *   1. Concatenate [x; h] — memcpy, ~2 ns
 *   2. Gate GEMM + sigmoid — W_gate[8×10] @ concat + bias, 8 sigmoid
 *   3. Cand GEMM + tanh — W_cand[8×10] @ concat + bias, 8 tanh
 *   4. Decay — 8 × exp(-dt/tau), OR precomputed (0 ns)
 *   5. Mix — 8 × multiply-add, ~1 ns
 *
 * Strategies tested:
 *   A: Precise (expf, tanhf, sigmoid via expf) — baseline
 *   B: FAST3 (degree-3 rational, no libm) — from activation_chip.h
 *   C: Schraudolph exp + FAST sigmoid/tanh — hybrid
 *   D: LUT+lerp (256-entry lookup + linear interpolation)
 *   E: LUT+cubic (256-entry lookup + cubic Hermite spline)
 *
 * Each strategy tested with:
 *   - Variable dt (CFC_CELL_GENERIC with full decay computation)
 *   - Fixed dt (CFC_CELL_FIXED with precomputed decay)
 *
 * All strategies use ternary-as-float weights (Strategy C from Probe 1).
 *
 * Compile:
 *   cc -O2 -I include -I include/chips experiments/ternary_quantization/quant_probe2_fpu.c -lm -o experiments/ternary_quantization/quant_probe2_fpu
 *
 * Created by: Tripp + Manus
 * Date: February 2026
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <sys/time.h>

#include "onnx_shapes.h"
#include "chips/cfc_cell_chip.h"
#include "chips/activation_chip.h"
#include "chips/decay_chip.h"

#define INPUT_DIM    2
#define HIDDEN_DIM   8
#define CONCAT_DIM   (INPUT_DIM + HIDDEN_DIM)

/* ═══════════════════════════════════════════════════════════════════════════
 * LUT — 256 entries, covers [-8, +8] for sigmoid and tanh.
 *
 * Why 256: fits in 1KB (256 × 4 bytes). Two tables = 2KB. Well within
 * L1 data cache (32KB+). All 8 neurons hit the same table — perfect
 * cache coherence across channels.
 *
 * Why [-8, +8]: sigmoid and tanh are both saturated beyond ±6.
 * Pre-activation values in CfC with ternary weights are bounded by
 * concat_dim (10) since each weight is ±1 or 0. Max possible
 * pre-activation = 10.0, but typical is much smaller.
 * ═══════════════════════════════════════════════════════════════════════════ */

#define LUT_SIZE     256
#define LUT_XMIN    (-8.0f)
#define LUT_XMAX    ( 8.0f)
#define LUT_RANGE   (LUT_XMAX - LUT_XMIN)
#define LUT_SCALE   ((float)(LUT_SIZE - 1) / LUT_RANGE)
#define LUT_INV     (LUT_RANGE / (float)(LUT_SIZE - 1))

static float sigmoid_lut[LUT_SIZE];
static float tanh_lut[LUT_SIZE];

/* Derivatives for cubic Hermite spline */
static float sigmoid_deriv_lut[LUT_SIZE];
static float tanh_deriv_lut[LUT_SIZE];

static void init_luts(void) {
    for (int i = 0; i < LUT_SIZE; i++) {
        float x = LUT_XMIN + (float)i * LUT_INV;
        sigmoid_lut[i] = 1.0f / (1.0f + expf(-x));
        tanh_lut[i] = tanhf(x);

        /* Exact derivatives for cubic spline */
        float s = sigmoid_lut[i];
        sigmoid_deriv_lut[i] = s * (1.0f - s);  /* sigmoid'(x) = sigmoid(x)(1-sigmoid(x)) */
        tanh_deriv_lut[i] = 1.0f - tanh_lut[i] * tanh_lut[i];  /* tanh'(x) = 1 - tanh²(x) */
    }
}

/* LUT + linear interpolation (lerp) */
static inline float sigmoid_lut_lerp(float x) {
    if (x <= LUT_XMIN) return sigmoid_lut[0];
    if (x >= LUT_XMAX) return sigmoid_lut[LUT_SIZE - 1];
    float idx_f = (x - LUT_XMIN) * LUT_SCALE;
    int idx = (int)idx_f;
    float frac = idx_f - (float)idx;
    return sigmoid_lut[idx] + frac * (sigmoid_lut[idx + 1] - sigmoid_lut[idx]);
}

static inline float tanh_lut_lerp(float x) {
    if (x <= LUT_XMIN) return tanh_lut[0];
    if (x >= LUT_XMAX) return tanh_lut[LUT_SIZE - 1];
    float idx_f = (x - LUT_XMIN) * LUT_SCALE;
    int idx = (int)idx_f;
    float frac = idx_f - (float)idx;
    return tanh_lut[idx] + frac * (tanh_lut[idx + 1] - tanh_lut[idx]);
}

/* LUT + cubic Hermite spline
 *
 * Uses exact derivatives at knot points for C1 continuity.
 * h00(t) = 2t³ - 3t² + 1,  h10(t) = t³ - 2t² + t
 * h01(t) = -2t³ + 3t²,     h11(t) = t³ - t²
 *
 * p(t) = h00*y0 + h10*dx*m0 + h01*y1 + h11*dx*m1
 * where dx = grid spacing, m0/m1 = derivatives at endpoints
 */
static inline float sigmoid_lut_cubic(float x) {
    if (x <= LUT_XMIN) return sigmoid_lut[0];
    if (x >= LUT_XMAX) return sigmoid_lut[LUT_SIZE - 1];
    float idx_f = (x - LUT_XMIN) * LUT_SCALE;
    int idx = (int)idx_f;
    if (idx >= LUT_SIZE - 1) idx = LUT_SIZE - 2;
    float t = idx_f - (float)idx;

    float y0 = sigmoid_lut[idx];
    float y1 = sigmoid_lut[idx + 1];
    float m0 = sigmoid_deriv_lut[idx] * LUT_INV;     /* scale derivative by grid spacing */
    float m1 = sigmoid_deriv_lut[idx + 1] * LUT_INV;

    float t2 = t * t;
    float t3 = t2 * t;

    return (2*t3 - 3*t2 + 1)*y0 + (t3 - 2*t2 + t)*m0
         + (-2*t3 + 3*t2)*y1 + (t3 - t2)*m1;
}

static inline float tanh_lut_cubic(float x) {
    if (x <= LUT_XMIN) return tanh_lut[0];
    if (x >= LUT_XMAX) return tanh_lut[LUT_SIZE - 1];
    float idx_f = (x - LUT_XMIN) * LUT_SCALE;
    int idx = (int)idx_f;
    if (idx >= LUT_SIZE - 1) idx = LUT_SIZE - 2;
    float t = idx_f - (float)idx;

    float y0 = tanh_lut[idx];
    float y1 = tanh_lut[idx + 1];
    float m0 = tanh_deriv_lut[idx] * LUT_INV;
    float m1 = tanh_deriv_lut[idx + 1] * LUT_INV;

    float t2 = t * t;
    float t3 = t2 * t;

    return (2*t3 - 3*t2 + 1)*y0 + (t3 - 2*t2 + t)*m0
         + (-2*t3 + 3*t2)*y1 + (t3 - t2)*m1;
}

/* ═══════════════════════════════════════════════════════════════════════════
 * CfC Cell Variants — same computation, different activation paths
 *
 * All use ternary-as-float weights via standard GEMM (yinsen_gemm).
 * ═══════════════════════════════════════════════════════════════════════════ */

/* Variant: FAST3 activations, variable dt */
static inline void cfc_cell_fast3(
    const float *x, const float *h_prev, float dt,
    const float *W_gate, const float *b_gate,
    const float *W_cand, const float *b_cand,
    const float *tau, int input_dim, int hidden_dim,
    float *h_new
) {
    const int cd = input_dim + hidden_dim;
    float concat[cd], gpre[hidden_dim], gate[hidden_dim];
    float cpre[hidden_dim], cand[hidden_dim];

    memcpy(concat, x, input_dim * sizeof(float));
    memcpy(concat + input_dim, h_prev, hidden_dim * sizeof(float));

    yinsen_gemm(concat, W_gate, b_gate, gpre, 1, hidden_dim, cd, 1.0f, 1.0f);
    for (int i = 0; i < hidden_dim; i++) gate[i] = SIGMOID_CHIP_FAST3(gpre[i]);

    yinsen_gemm(concat, W_cand, b_cand, cpre, 1, hidden_dim, cd, 1.0f, 1.0f);
    for (int i = 0; i < hidden_dim; i++) cand[i] = TANH_CHIP_FAST3(cpre[i]);

    for (int i = 0; i < hidden_dim; i++) {
        float decay = expf(-dt / tau[i]);
        h_new[i] = (1.0f - gate[i]) * h_prev[i] * decay + gate[i] * cand[i];
    }
}

/* Variant: FAST3 activations, FIXED dt (precomputed decay) */
static inline void cfc_cell_fast3_fixed(
    const float *x, const float *h_prev,
    const float *W_gate, const float *b_gate,
    const float *W_cand, const float *b_cand,
    const float *decay_pre, int input_dim, int hidden_dim,
    float *h_new
) {
    const int cd = input_dim + hidden_dim;
    float concat[cd], gpre[hidden_dim], gate[hidden_dim];
    float cpre[hidden_dim], cand[hidden_dim];

    memcpy(concat, x, input_dim * sizeof(float));
    memcpy(concat + input_dim, h_prev, hidden_dim * sizeof(float));

    yinsen_gemm(concat, W_gate, b_gate, gpre, 1, hidden_dim, cd, 1.0f, 1.0f);
    for (int i = 0; i < hidden_dim; i++) gate[i] = SIGMOID_CHIP_FAST3(gpre[i]);

    yinsen_gemm(concat, W_cand, b_cand, cpre, 1, hidden_dim, cd, 1.0f, 1.0f);
    for (int i = 0; i < hidden_dim; i++) cand[i] = TANH_CHIP_FAST3(cpre[i]);

    for (int i = 0; i < hidden_dim; i++)
        h_new[i] = (1.0f - gate[i]) * h_prev[i] * decay_pre[i] + gate[i] * cand[i];
}

/* Variant: LUT+lerp, fixed dt */
static inline void cfc_cell_lut_lerp_fixed(
    const float *x, const float *h_prev,
    const float *W_gate, const float *b_gate,
    const float *W_cand, const float *b_cand,
    const float *decay_pre, int input_dim, int hidden_dim,
    float *h_new
) {
    const int cd = input_dim + hidden_dim;
    float concat[cd], gpre[hidden_dim], gate[hidden_dim];
    float cpre[hidden_dim], cand[hidden_dim];

    memcpy(concat, x, input_dim * sizeof(float));
    memcpy(concat + input_dim, h_prev, hidden_dim * sizeof(float));

    yinsen_gemm(concat, W_gate, b_gate, gpre, 1, hidden_dim, cd, 1.0f, 1.0f);
    for (int i = 0; i < hidden_dim; i++) gate[i] = sigmoid_lut_lerp(gpre[i]);

    yinsen_gemm(concat, W_cand, b_cand, cpre, 1, hidden_dim, cd, 1.0f, 1.0f);
    for (int i = 0; i < hidden_dim; i++) cand[i] = tanh_lut_lerp(cpre[i]);

    for (int i = 0; i < hidden_dim; i++)
        h_new[i] = (1.0f - gate[i]) * h_prev[i] * decay_pre[i] + gate[i] * cand[i];
}

/* Variant: LUT+cubic, fixed dt */
static inline void cfc_cell_lut_cubic_fixed(
    const float *x, const float *h_prev,
    const float *W_gate, const float *b_gate,
    const float *W_cand, const float *b_cand,
    const float *decay_pre, int input_dim, int hidden_dim,
    float *h_new
) {
    const int cd = input_dim + hidden_dim;
    float concat[cd], gpre[hidden_dim], gate[hidden_dim];
    float cpre[hidden_dim], cand[hidden_dim];

    memcpy(concat, x, input_dim * sizeof(float));
    memcpy(concat + input_dim, h_prev, hidden_dim * sizeof(float));

    yinsen_gemm(concat, W_gate, b_gate, gpre, 1, hidden_dim, cd, 1.0f, 1.0f);
    for (int i = 0; i < hidden_dim; i++) gate[i] = sigmoid_lut_cubic(gpre[i]);

    yinsen_gemm(concat, W_cand, b_cand, cpre, 1, hidden_dim, cd, 1.0f, 1.0f);
    for (int i = 0; i < hidden_dim; i++) cand[i] = tanh_lut_cubic(cpre[i]);

    for (int i = 0; i < hidden_dim; i++)
        h_new[i] = (1.0f - gate[i]) * h_prev[i] * decay_pre[i] + gate[i] * cand[i];
}

/* ═══════════════════════════════════════════════════════════════════════════
 * Ternary-as-float weights (t=0.10 from Probe 1 — best threshold)
 * ═══════════════════════════════════════════════════════════════════════════ */

/* Quantized at threshold=0.10:
 * 0.1 → +1, 0.0 stays 0, -0.1 → -1 ... wait, |-0.1| = 0.1 which is NOT > 0.1
 * So at t=0.10, weights with |w|=0.1 map to 0. Only |w|>0.1 survive.
 * That means 0.2→1, 0.3→1, ..., 0.8→1, -0.1→0, 0.1→0, 0.0→0 */

static void quantize_to_float(const float *src, float *dst, int n, float thresh) {
    for (int i = 0; i < n; i++) {
        if (src[i] > thresh) dst[i] = 1.0f;
        else if (src[i] < -thresh) dst[i] = -1.0f;
        else dst[i] = 0.0f;
    }
}

/* Original float weights */
static const float W_gate_orig[HIDDEN_DIM * CONCAT_DIM] = {
     0.1f,  0.8f,   0.1f, -0.1f,  0.0f,  0.0f,  0.0f,  0.0f,  0.0f,  0.0f,
     0.7f,  0.2f,  -0.1f,  0.1f,  0.0f,  0.0f,  0.0f,  0.0f,  0.0f,  0.0f,
     0.4f,  0.5f,   0.0f,  0.0f,  0.1f, -0.1f,  0.0f,  0.0f,  0.0f,  0.0f,
     0.1f,  0.6f,   0.0f,  0.0f,  0.0f,  0.0f,  0.1f, -0.1f,  0.0f,  0.0f,
     0.6f,  0.1f,   0.0f,  0.0f,  0.0f,  0.0f,  0.0f,  0.0f,  0.1f, -0.1f,
     0.2f,  0.3f,   0.2f,  0.2f,  0.1f,  0.0f,  0.0f,  0.0f,  0.0f,  0.0f,
     0.3f,  0.3f,   0.1f,  0.0f,  0.0f,  0.1f,  0.0f,  0.0f,  0.0f,  0.0f,
     0.2f,  0.7f,  -0.1f,  0.0f,  0.0f,  0.0f,  0.0f,  0.1f,  0.0f,  0.0f,
};

static const float b_gate[HIDDEN_DIM] = {
    -0.5f, -0.3f, -0.4f, -0.5f, -0.3f, -0.4f, -0.3f, -0.5f
};

static const float W_cand_orig[HIDDEN_DIM * CONCAT_DIM] = {
     0.5f,  0.3f,   0.1f,  0.0f, -0.1f,  0.0f,  0.0f,  0.0f,  0.0f,  0.0f,
     0.3f,  0.5f,   0.0f,  0.1f,  0.0f, -0.1f,  0.0f,  0.0f,  0.0f,  0.0f,
     0.4f,  0.4f,  -0.1f,  0.0f,  0.0f,  0.0f,  0.1f,  0.0f,  0.0f,  0.0f,
     0.2f,  0.6f,   0.0f, -0.1f,  0.0f,  0.0f,  0.0f,  0.1f,  0.0f,  0.0f,
     0.6f,  0.2f,   0.0f,  0.0f, -0.1f,  0.0f,  0.0f,  0.0f,  0.1f,  0.0f,
     0.3f,  0.4f,   0.1f,  0.1f,  0.0f,  0.0f, -0.1f,  0.0f,  0.0f,  0.0f,
     0.4f,  0.3f,   0.0f,  0.0f,  0.1f,  0.1f,  0.0f, -0.1f,  0.0f,  0.0f,
     0.3f,  0.5f,   0.0f,  0.0f,  0.0f,  0.0f,  0.1f,  0.0f, -0.1f,  0.0f,
};

static const float b_cand[HIDDEN_DIM] = {0};

static const float tau[HIDDEN_DIM] = {
    5.0f, 15.0f, 45.0f, 120.0f, 10.0f, 30.0f, 90.0f, 600.0f
};

/* ═══════════════════════════════════════════════════════════════════════════
 * Accuracy measurement
 * ═══════════════════════════════════════════════════════════════════════════ */

static float max_abs_err_sigmoid(float (*approx)(float)) {
    float max_err = 0;
    for (int i = 0; i < 10000; i++) {
        float x = -8.0f + 16.0f * i / 9999.0f;
        float exact = 1.0f / (1.0f + expf(-x));
        float approx_val = approx(x);
        float err = fabsf(exact - approx_val);
        if (err > max_err) max_err = err;
    }
    return max_err;
}

static float max_abs_err_tanh(float (*approx)(float)) {
    float max_err = 0;
    for (int i = 0; i < 10000; i++) {
        float x = -8.0f + 16.0f * i / 9999.0f;
        float exact = tanhf(x);
        float approx_val = approx(x);
        float err = fabsf(exact - approx_val);
        if (err > max_err) max_err = err;
    }
    return max_err;
}

/* ═══════════════════════════════════════════════════════════════════════════
 * MAIN
 * ═══════════════════════════════════════════════════════════════════════════ */

int main(void) {
    printf("═══════════════════════════════════════════════════════════════\n");
    printf("  TERNARY QUANTIZATION PROBE 2 — FPU Optimization\n");
    printf("═══════════════════════════════════════════════════════════════\n\n");

    init_luts();

    /* Quantize weights at t=0.10 */
    float W_gate_t[HIDDEN_DIM * CONCAT_DIM];
    float W_cand_t[HIDDEN_DIM * CONCAT_DIM];
    quantize_to_float(W_gate_orig, W_gate_t, HIDDEN_DIM * CONCAT_DIM, 0.10f);
    quantize_to_float(W_cand_orig, W_cand_t, HIDDEN_DIM * CONCAT_DIM, 0.10f);

    /* Precompute decay for fixed-dt path */
    float decay_pre[HIDDEN_DIM];
    float dt = 10.0f;
    for (int i = 0; i < HIDDEN_DIM; i++)
        decay_pre[i] = expf(-dt / tau[i]);

    /* ── 1. Accuracy of each activation path ── */
    printf("1. ACTIVATION ACCURACY (max absolute error over [-8, +8])\n\n");

    printf("   %-20s  %12s  %12s\n", "Method", "Sigmoid Err", "Tanh Err");
    printf("   %-20s  %12s  %12s\n", "--------------------", "------------", "------------");
    printf("   %-20s  %12s  %12s\n", "Precise (libm)", "0 (exact)", "0 (exact)");
    printf("   %-20s  %12.2e  %12.2e\n", "FAST (rational)",
           max_abs_err_sigmoid(SIGMOID_CHIP_FAST),
           max_abs_err_tanh(TANH_CHIP_FAST));
    printf("   %-20s  %12.2e  %12.2e\n", "FAST3 (degree-3)",
           max_abs_err_sigmoid(SIGMOID_CHIP_FAST3),
           max_abs_err_tanh(TANH_CHIP_FAST3));
    printf("   %-20s  %12.2e  %12.2e\n", "LUT+lerp (256)",
           max_abs_err_sigmoid(sigmoid_lut_lerp),
           max_abs_err_tanh(tanh_lut_lerp));
    printf("   %-20s  %12.2e  %12.2e\n", "LUT+cubic (256)",
           max_abs_err_sigmoid(sigmoid_lut_cubic),
           max_abs_err_tanh(tanh_lut_cubic));

    printf("\n   LUT memory: sigmoid %zu bytes + tanh %zu bytes = %zu bytes\n",
           sizeof(sigmoid_lut), sizeof(tanh_lut),
           sizeof(sigmoid_lut) + sizeof(tanh_lut));
    printf("   LUT+cubic adds: %zu bytes (derivatives) = %zu total\n",
           sizeof(sigmoid_deriv_lut) + sizeof(tanh_deriv_lut),
           sizeof(sigmoid_lut) + sizeof(tanh_lut) +
           sizeof(sigmoid_deriv_lut) + sizeof(tanh_deriv_lut));

    /* ── 2. Isolated activation benchmarks ── */
    printf("\n2. ISOLATED ACTIVATION BENCHMARKS (100K calls)\n\n");

    {
        int M = 100000;
        float vals[HIDDEN_DIM];
        /* Pre-fill with typical pre-activation values */
        for (int i = 0; i < HIDDEN_DIM; i++)
            vals[i] = -2.0f + 4.0f * i / (HIDDEN_DIM - 1);

        volatile float sink = 0;
        struct timeval t0, t1;

        /* Sigmoid variants */
        #define BENCH_ACTIVATION(label, expr) do { \
            gettimeofday(&t0, NULL); \
            for (int k = 0; k < M; k++) { \
                for (int i = 0; i < HIDDEN_DIM; i++) sink += (expr); \
            } \
            gettimeofday(&t1, NULL); \
            double us = (double)(t1.tv_sec - t0.tv_sec) * 1e6 \
                      + (double)(t1.tv_usec - t0.tv_usec); \
            printf("   %-25s  %6.1f ns/call (%d × %d)\n", \
                   label, (us / M / HIDDEN_DIM) * 1000.0, M, HIDDEN_DIM); \
        } while(0)

        printf("   Sigmoid:\n");
        BENCH_ACTIVATION("Precise (expf)", SIGMOID_CHIP(vals[i]));
        BENCH_ACTIVATION("FAST (rational)", SIGMOID_CHIP_FAST(vals[i]));
        BENCH_ACTIVATION("FAST3 (degree-3)", SIGMOID_CHIP_FAST3(vals[i]));
        BENCH_ACTIVATION("LUT+lerp", sigmoid_lut_lerp(vals[i]));
        BENCH_ACTIVATION("LUT+cubic", sigmoid_lut_cubic(vals[i]));

        printf("\n   Tanh:\n");
        BENCH_ACTIVATION("Precise (tanhf)", TANH_CHIP(vals[i]));
        BENCH_ACTIVATION("FAST (rational)", TANH_CHIP_FAST(vals[i]));
        BENCH_ACTIVATION("FAST3 (degree-3)", TANH_CHIP_FAST3(vals[i]));
        BENCH_ACTIVATION("LUT+lerp", tanh_lut_lerp(vals[i]));
        BENCH_ACTIVATION("LUT+cubic", tanh_lut_cubic(vals[i]));

        printf("\n   Decay (exp):\n");
        BENCH_ACTIVATION("Precise (expf)", expf(-dt / tau[i % HIDDEN_DIM]));
        BENCH_ACTIVATION("Schraudolph", EXP_CHIP_FAST(-dt / tau[i % HIDDEN_DIM]));
        BENCH_ACTIVATION("Precomputed", decay_pre[i]);  /* just a load */

        (void)sink;
    }

    /* ── 3. Full CfC step benchmarks ── */
    printf("\n3. FULL CfC STEP — Ternary-as-float weights, all combos\n");
    printf("   (Each row: 100K iterations of one CfC step, HIDDEN_DIM=%d)\n\n", HIDDEN_DIM);

    {
        int M = 100000;
        float h[HIDDEN_DIM] = {0};
        float inp[INPUT_DIM] = {0.5f, 1.0f};
        float h_new[HIDDEN_DIM];
        struct timeval t0, t1;

        #define BENCH_CFC(label, call) do { \
            memset(h, 0, sizeof(h)); \
            gettimeofday(&t0, NULL); \
            for (int k = 0; k < M; k++) { \
                call; \
                memcpy(h, h_new, sizeof(h)); \
            } \
            gettimeofday(&t1, NULL); \
            double us = (double)(t1.tv_sec - t0.tv_sec) * 1e6 \
                      + (double)(t1.tv_usec - t0.tv_usec); \
            double ns = (us / M) * 1000.0; \
            printf("   %-45s  %5.0f ns/step\n", label, ns); \
        } while(0)

        printf("   %-45s  %s\n", "Configuration", "Time");
        printf("   %-45s  %s\n", "---------------------------------------------", "---------");

        /* Variable dt (includes decay exp) */
        BENCH_CFC("Precise + variable dt (baseline)",
            CFC_CELL_GENERIC(inp, h, dt, W_gate_t, b_gate, W_cand_t, b_cand,
                             tau, 0, INPUT_DIM, HIDDEN_DIM, h_new));

        BENCH_CFC("FAST3 + variable dt",
            cfc_cell_fast3(inp, h, dt, W_gate_t, b_gate, W_cand_t, b_cand,
                           tau, INPUT_DIM, HIDDEN_DIM, h_new));

        printf("   ---\n");

        /* Fixed dt (precomputed decay — eliminates all exp from hot path) */
        BENCH_CFC("Precise + FIXED dt (precomputed decay)",
            CFC_CELL_FIXED(inp, h, W_gate_t, b_gate, W_cand_t, b_cand,
                           decay_pre, INPUT_DIM, HIDDEN_DIM, h_new));

        BENCH_CFC("FAST3 + FIXED dt",
            cfc_cell_fast3_fixed(inp, h, W_gate_t, b_gate, W_cand_t, b_cand,
                                 decay_pre, INPUT_DIM, HIDDEN_DIM, h_new));

        BENCH_CFC("LUT+lerp + FIXED dt",
            cfc_cell_lut_lerp_fixed(inp, h, W_gate_t, b_gate, W_cand_t, b_cand,
                                    decay_pre, INPUT_DIM, HIDDEN_DIM, h_new));

        BENCH_CFC("LUT+cubic + FIXED dt",
            cfc_cell_lut_cubic_fixed(inp, h, W_gate_t, b_gate, W_cand_t, b_cand,
                                     decay_pre, INPUT_DIM, HIDDEN_DIM, h_new));
    }

    /* ── 4. Cache coherence analysis ── */
    printf("\n4. CACHE COHERENCE ANALYSIS\n\n");
    {
        size_t w_bytes = 2 * HIDDEN_DIM * CONCAT_DIM * sizeof(float); /* ternary-as-float */
        size_t b_bytes = 2 * HIDDEN_DIM * sizeof(float);
        size_t tau_bytes = HIDDEN_DIM * sizeof(float);
        size_t decay_bytes = HIDDEN_DIM * sizeof(float);
        size_t lut_bytes = sizeof(sigmoid_lut) + sizeof(tanh_lut);
        size_t lut_cubic_bytes = lut_bytes + sizeof(sigmoid_deriv_lut) + sizeof(tanh_deriv_lut);
        size_t h_bytes = HIDDEN_DIM * sizeof(float);

        printf("   Read-only shared data (hot across all channels):\n");
        printf("     W_gate + W_cand (ternary-as-float): %4zu bytes\n", w_bytes);
        printf("     b_gate + b_cand:                    %4zu bytes\n", b_bytes);
        printf("     tau:                                %4zu bytes\n", tau_bytes);
        printf("     decay_precomputed:                  %4zu bytes\n", decay_bytes);
        printf("     sigmoid LUT + tanh LUT:             %4zu bytes\n", lut_bytes);
        printf("     + cubic derivative LUTs:            %4zu bytes\n", lut_cubic_bytes);
        printf("     Total shared (lerp):                %4zu bytes\n",
               w_bytes + b_bytes + tau_bytes + decay_bytes + lut_bytes);
        printf("     Total shared (cubic):               %4zu bytes\n",
               w_bytes + b_bytes + tau_bytes + decay_bytes + lut_cubic_bytes);
        printf("\n");
        printf("   Per-channel mutable data:\n");
        printf("     h_state:                            %4zu bytes\n", h_bytes);
        printf("     (discriminant, calibration etc. not in CfC hot path)\n");
        printf("\n");

        int n_channels = 8;
        size_t total_shared_lerp = w_bytes + b_bytes + decay_bytes + lut_bytes;
        size_t total_per_ch = h_bytes + CONCAT_DIM * sizeof(float)  /* concat */
                            + 4 * HIDDEN_DIM * sizeof(float);  /* gate_pre, gate, cand_pre, cand */
        size_t total_hot = total_shared_lerp + n_channels * total_per_ch;

        printf("   Hot working set (%d channels):\n", n_channels);
        printf("     Shared: %zu bytes\n", total_shared_lerp);
        printf("     Per-channel stack: %zu bytes × %d = %zu bytes\n",
               total_per_ch, n_channels, n_channels * total_per_ch);
        printf("     Total: %zu bytes (%.1f%% of 32KB L1D)\n",
               total_hot, 100.0f * total_hot / 32768.0f);
        printf("\n");
        printf("   Cache coherence win: all %d channels read the same weights\n", n_channels);
        printf("   and the same LUT. Only h_state is per-channel. The LUTs\n");
        printf("   stay hot after the first channel's step.\n");
    }

    /* ── 5. Detection quality sanity check ── */
    printf("\n5. DETECTION QUALITY — Hidden state divergence from precise\n");
    printf("   (Run 1000 steps, measure L2 distance from precise path)\n\n");
    {
        float h_precise[HIDDEN_DIM] = {0};
        float h_fast3[HIDDEN_DIM] = {0};
        float h_lerp[HIDDEN_DIM] = {0};
        float h_cubic[HIDDEN_DIM] = {0};
        float h_new[HIDDEN_DIM];

        unsigned int seed = 42;
        int N = 1000;

        for (int step = 0; step < N; step++) {
            /* Synthetic input */
            seed = seed * 1103515245u + 12345u;
            float v = (float)(seed & 0xFFFF) / 65536.0f - 0.5f;
            float inp[INPUT_DIM] = {v, 1.0f};

            /* Precise */
            CFC_CELL_FIXED(inp, h_precise, W_gate_t, b_gate, W_cand_t, b_cand,
                           decay_pre, INPUT_DIM, HIDDEN_DIM, h_new);
            memcpy(h_precise, h_new, sizeof(h_new));

            /* FAST3 */
            cfc_cell_fast3_fixed(inp, h_fast3, W_gate_t, b_gate, W_cand_t, b_cand,
                                 decay_pre, INPUT_DIM, HIDDEN_DIM, h_new);
            memcpy(h_fast3, h_new, sizeof(h_new));

            /* LUT+lerp */
            cfc_cell_lut_lerp_fixed(inp, h_lerp, W_gate_t, b_gate, W_cand_t, b_cand,
                                    decay_pre, INPUT_DIM, HIDDEN_DIM, h_new);
            memcpy(h_lerp, h_new, sizeof(h_new));

            /* LUT+cubic */
            cfc_cell_lut_cubic_fixed(inp, h_cubic, W_gate_t, b_gate, W_cand_t, b_cand,
                                     decay_pre, INPUT_DIM, HIDDEN_DIM, h_new);
            memcpy(h_cubic, h_new, sizeof(h_new));
        }

        /* L2 distance from precise */
        float d_fast3 = 0, d_lerp = 0, d_cubic = 0;
        for (int i = 0; i < HIDDEN_DIM; i++) {
            float df = h_fast3[i] - h_precise[i];
            float dl = h_lerp[i] - h_precise[i];
            float dc = h_cubic[i] - h_precise[i];
            d_fast3 += df * df;
            d_lerp += dl * dl;
            d_cubic += dc * dc;
        }
        d_fast3 = sqrtf(d_fast3);
        d_lerp = sqrtf(d_lerp);
        d_cubic = sqrtf(d_cubic);

        printf("   After %d steps (fixed dt, ternary-as-float weights):\n", N);
        printf("   %-20s  L2 from precise\n", "Method");
        printf("   %-20s  %-15s\n", "--------------------", "---------------");
        printf("   %-20s  %.6e\n", "FAST3 (degree-3)", d_fast3);
        printf("   %-20s  %.6e\n", "LUT+lerp (256)", d_lerp);
        printf("   %-20s  %.6e\n", "LUT+cubic (256)", d_cubic);

        /* Also show the hidden states */
        printf("\n   Final hidden states:\n");
        printf("   %-10s", "Dim");
        printf("  %-10s", "Precise");
        printf("  %-10s", "FAST3");
        printf("  %-10s", "LUT+lerp");
        printf("  %-10s\n", "LUT+cubic");
        for (int i = 0; i < HIDDEN_DIM; i++) {
            printf("   h[%d]     %10.6f  %10.6f  %10.6f  %10.6f\n",
                   i, h_precise[i], h_fast3[i], h_lerp[i], h_cubic[i]);
        }
    }

    printf("\n═══════════════════════════════════════════════════════════════\n");
    printf("  SUMMARY\n");
    printf("═══════════════════════════════════════════════════════════════\n\n");
    printf("  The optimal CfC configuration for FPU platforms:\n");
    printf("    Weights:     ternary-as-float (+1, 0, -1 stored as 2-bit packed)\n");
    printf("    GEMM:        standard float multiply-accumulate (FPU speed)\n");
    printf("    Activations: [winner from benchmarks above]\n");
    printf("    Decay:       precomputed for fixed-rate sensors\n");
    printf("    LUT:         shared read-only, hot in L1 across all channels\n");
    printf("\n");

    return 0;
}
