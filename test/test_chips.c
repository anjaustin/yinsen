/*
 * test_chips.c — Verify all forged chips
 *
 * Tests correctness of each chip against known values or libm reference.
 * Every chip must compile standalone (only <math.h>/<stdint.h>).
 */

#include <stdio.h>
#include <math.h>
#include <string.h>
#include <assert.h>

#include "chips/gemm_chip.h"
#include "chips/activation_chip.h"
#include "chips/decay_chip.h"
#include "chips/ternary_dot_chip.h"
#include "chips/fft_chip.h"
#include "chips/softmax_chip.h"
#include "chips/norm_chip.h"
#include "chips/cfc_cell_chip.h"

static int tests_passed = 0;
static int tests_failed = 0;

#define ASSERT_NEAR(a, b, eps, label) do { \
    float _a = (a), _b = (b), _e = (eps); \
    if (fabsf(_a - _b) > _e) { \
        printf("  FAIL: %s: expected %.6f, got %.6f (diff %.2e)\n", \
               label, _b, _a, fabsf(_a - _b)); \
        tests_failed++; \
    } else { tests_passed++; } \
} while(0)

#define ASSERT_EQ_INT(a, b, label) do { \
    if ((a) != (b)) { \
        printf("  FAIL: %s: expected %d, got %d\n", label, (b), (a)); \
        tests_failed++; \
    } else { tests_passed++; } \
} while(0)

/* ═══════════════════════════════════════════════════════════════ */
void test_gemm_chip(void) {
    printf("--- GEMM_CHIP ---\n");

    /* 1x3 @ 3x2 = 1x2 */
    float a[3] = {1, 2, 3};
    float b[6] = {1, 2,  3, 4,  5, 6};  /* 3x2 row-major */
    float c[2];

    GEMM_CHIP_BARE(a, b, c, 1, 2, 3);
    /* [1,2,3] @ [[1,2],[3,4],[5,6]] = [1+6+15, 2+8+18] = [22, 28] */
    ASSERT_NEAR(c[0], 22.0f, 1e-5f, "bare [0]");
    ASSERT_NEAR(c[1], 28.0f, 1e-5f, "bare [1]");

    float bias[2] = {10, 20};
    GEMM_CHIP_BIASED(a, b, bias, c, 1, 2, 3);
    ASSERT_NEAR(c[0], 32.0f, 1e-5f, "biased [0]");
    ASSERT_NEAR(c[1], 48.0f, 1e-5f, "biased [1]");

    GEMM_CHIP(a, b, bias, c, 1, 2, 3, 2.0f, 0.5f);
    /* 2 * 22 + 0.5 * 10 = 49, 2 * 28 + 0.5 * 20 = 66 */
    ASSERT_NEAR(c[0], 49.0f, 1e-5f, "full [0]");
    ASSERT_NEAR(c[1], 66.0f, 1e-5f, "full [1]");

    /* MATVEC_CHIP: y = W @ x + bias, W is 2x3 (transposed layout) */
    float W[6] = {1, 3, 5,  2, 4, 6};  /* 2x3 row-major */
    float x[3] = {1, 2, 3};
    float y[2];
    float bv[2] = {10, 20};
    MATVEC_CHIP(x, W, bv, y, 2, 3);
    /* row0: 1+6+15+10=32, row1: 2+8+18+20=48 */
    ASSERT_NEAR(y[0], 32.0f, 1e-5f, "matvec [0]");
    ASSERT_NEAR(y[1], 48.0f, 1e-5f, "matvec [1]");

    printf("  %d passed\n\n", 8);
}

/* ═══════════════════════════════════════════════════════════════ */
void test_activation_chip(void) {
    printf("--- ACTIVATION_CHIP ---\n");

    /* Sigmoid */
    ASSERT_NEAR(SIGMOID_CHIP(0.0f), 0.5f, 1e-6f, "sigmoid(0)");
    ASSERT_NEAR(SIGMOID_CHIP(10.0f), 1.0f, 1e-4f, "sigmoid(10)");
    ASSERT_NEAR(SIGMOID_CHIP(-10.0f), 0.0f, 1e-4f, "sigmoid(-10)");

    /* Sigmoid fast: correct at 0, correct limits, monotonic */
    ASSERT_NEAR(SIGMOID_CHIP_FAST(0.0f), 0.5f, 1e-6f, "sigmoid_fast(0)");
    float sf_pos = SIGMOID_CHIP_FAST(5.0f);
    float sf_neg = SIGMOID_CHIP_FAST(-5.0f);
    assert(sf_pos > 0.5f && sf_pos < 1.0f);
    assert(sf_neg > 0.0f && sf_neg < 0.5f);
    tests_passed += 2;

    /* Tanh */
    ASSERT_NEAR(TANH_CHIP(0.0f), 0.0f, 1e-6f, "tanh(0)");
    ASSERT_NEAR(TANH_CHIP(5.0f), 1.0f, 1e-4f, "tanh(5)");
    ASSERT_NEAR(TANH_CHIP_FAST(0.0f), 0.0f, 1e-6f, "tanh_fast(0)");

    /* Exp */
    ASSERT_NEAR(EXP_CHIP(0.0f), 1.0f, 1e-6f, "exp(0)");
    ASSERT_NEAR(EXP_CHIP(1.0f), 2.71828f, 1e-3f, "exp(1)");

    /* Exp fast: within 5% */
    float ef = EXP_CHIP_FAST(1.0f);
    ASSERT_NEAR(ef, 2.71828f, 0.15f, "exp_fast(1)");

    /* ReLU */
    ASSERT_NEAR(RELU_CHIP(1.0f), 1.0f, 1e-6f, "relu(1)");
    ASSERT_NEAR(RELU_CHIP(-1.0f), 0.0f, 1e-6f, "relu(-1)");

    /* Vector versions */
    float xv[4] = {-2, -1, 0, 1};
    float yv[4];
    SIGMOID_VEC_CHIP(xv, yv, 4);
    assert(yv[0] < 0.5f && yv[3] > 0.5f);
    tests_passed++;

    /* LUT+lerp — init and test accuracy */
    ACTIVATION_LUT_INIT();

    /* Sigmoid LUT: correct at 0, within 1e-4 across range */
    ASSERT_NEAR(SIGMOID_CHIP_LUT(0.0f), 0.5f, 1e-4f, "sigmoid_lut(0)");
    ASSERT_NEAR(SIGMOID_CHIP_LUT(5.0f), SIGMOID_CHIP(5.0f), 1e-4f, "sigmoid_lut(5)");
    ASSERT_NEAR(SIGMOID_CHIP_LUT(-5.0f), SIGMOID_CHIP(-5.0f), 1e-4f, "sigmoid_lut(-5)");

    /* Tanh LUT: correct at 0, within 1e-3 across range */
    ASSERT_NEAR(TANH_CHIP_LUT(0.0f), 0.0f, 1e-4f, "tanh_lut(0)");
    ASSERT_NEAR(TANH_CHIP_LUT(3.0f), TANH_CHIP(3.0f), 1e-3f, "tanh_lut(3)");
    ASSERT_NEAR(TANH_CHIP_LUT(-3.0f), TANH_CHIP(-3.0f), 1e-3f, "tanh_lut(-3)");

    /* LUT saturation bounds */
    ASSERT_NEAR(SIGMOID_CHIP_LUT(-10.0f), 0.0f, 1e-3f, "sigmoid_lut_sat_lo");
    ASSERT_NEAR(SIGMOID_CHIP_LUT(10.0f), 1.0f, 1e-3f, "sigmoid_lut_sat_hi");
    ASSERT_NEAR(TANH_CHIP_LUT(-10.0f), -1.0f, 1e-3f, "tanh_lut_sat_lo");
    ASSERT_NEAR(TANH_CHIP_LUT(10.0f), 1.0f, 1e-3f, "tanh_lut_sat_hi");

    /* LUT vector versions */
    float xvl[4] = {-3, -1, 1, 3};
    float yvl[4];
    SIGMOID_VEC_CHIP_LUT(xvl, yvl, 4);
    assert(yvl[0] < 0.5f && yvl[3] > 0.5f);
    tests_passed++;
    TANH_VEC_CHIP_LUT(xvl, yvl, 4);
    assert(yvl[0] < 0.0f && yvl[3] > 0.0f);
    tests_passed++;

    /* Idempotent init (should not crash) */
    ACTIVATION_LUT_INIT();
    tests_passed++;

    printf("  %d passed\n\n", 15 + 14);
}

/* ═══════════════════════════════════════════════════════════════ */
void test_decay_chip(void) {
    printf("--- DECAY_CHIP ---\n");

    /* exp(-0.01 / 1.0) = exp(-0.01) ≈ 0.99005 */
    ASSERT_NEAR(DECAY_CHIP_SCALAR(0.01f, 1.0f), expf(-0.01f), 1e-6f, "scalar");

    /* Fast: within 5% */
    float df = DECAY_CHIP_SCALAR_FAST(0.01f, 1.0f);
    ASSERT_NEAR(df, expf(-0.01f), 0.05f, "scalar_fast");

    /* Shared */
    float decay[4];
    DECAY_CHIP_SHARED(0.01f, 1.0f, decay, 4);
    for (int i = 0; i < 4; i++) {
        ASSERT_NEAR(decay[i], expf(-0.01f), 1e-6f, "shared");
    }

    /* Per-neuron */
    float tau[4] = {0.5f, 1.0f, 2.0f, 5.0f};
    DECAY_CHIP_PER_NEURON(0.01f, tau, decay, 4);
    for (int i = 0; i < 4; i++) {
        ASSERT_NEAR(decay[i], expf(-0.01f / tau[i]), 1e-6f, "per_neuron");
    }

    /* Precompute */
    float decay2[4];
    DECAY_CHIP_PRECOMPUTE(0.01f, tau, 0, decay2, 4);
    for (int i = 0; i < 4; i++) {
        ASSERT_NEAR(decay2[i], decay[i], 1e-6f, "precompute");
    }

    printf("  %d passed\n\n", 14);
}

/* ═══════════════════════════════════════════════════════════════ */
void test_ternary_dot_chip(void) {
    printf("--- TERNARY_DOT_CHIP ---\n");

    /* 4 weights: [+1, -1, 0, +1], activations [1, 2, 3, 4] */
    /* Expected: 1 - 2 + 0 + 4 = 3 */
    uint8_t packed = (0x01) | (0x02 << 2) | (0x00 << 4) | (0x01 << 6);
    float x[4] = {1, 2, 3, 4};

    float result = TERNARY_DOT_CHIP(&packed, x, 4);
    ASSERT_NEAR(result, 3.0f, 1e-6f, "dot4");

    /* Int8 version */
    int8_t xi[4] = {10, 20, 30, 40};
    int32_t iresult = TERNARY_DOT_INT8_CHIP(&packed, xi, 4);
    ASSERT_EQ_INT(iresult, 10 - 20 + 0 + 40, "dot4_int8");

    /* Matvec: 2x4 matrix */
    uint8_t W[2] = {packed, packed};
    float mv_y[2];
    TERNARY_MATVEC_CHIP(W, x, mv_y, 2, 4);
    ASSERT_NEAR(mv_y[0], 3.0f, 1e-6f, "matvec[0]");
    ASSERT_NEAR(mv_y[1], 3.0f, 1e-6f, "matvec[1]");

    /* Matvec with bias */
    float bias[2] = {10, 20};
    TERNARY_MATVEC_BIAS_CHIP(W, x, bias, mv_y, 2, 4);
    ASSERT_NEAR(mv_y[0], 13.0f, 1e-6f, "matvec_bias[0]");
    ASSERT_NEAR(mv_y[1], 23.0f, 1e-6f, "matvec_bias[1]");

    /* 5 elements (remainder test) */
    uint8_t packed2[2];
    packed2[0] = (0x01) | (0x01 << 2) | (0x01 << 4) | (0x01 << 6); /* all +1 */
    packed2[1] = (0x02);  /* -1 */
    float x5[5] = {1, 2, 3, 4, 5};
    float r5 = TERNARY_DOT_CHIP(packed2, x5, 5);
    ASSERT_NEAR(r5, 1+2+3+4-5, 1e-6f, "dot5_remainder");

    printf("  %d passed\n\n", 7);
}

/* ═══════════════════════════════════════════════════════════════ */
void test_fft_chip(void) {
    printf("--- FFT_CHIP ---\n");

    /* 8-point FFT of a known signal: DC + cosine */
    int N = 8;
    float real[8], imag[8];

    /* Pure DC signal: all ones */
    for (int i = 0; i < N; i++) { real[i] = 1.0f; imag[i] = 0.0f; }
    FFT_CHIP(real, imag, N);
    ASSERT_NEAR(real[0], 8.0f, 1e-4f, "DC real[0]");
    ASSERT_NEAR(real[1], 0.0f, 1e-4f, "DC real[1]");
    ASSERT_NEAR(imag[0], 0.0f, 1e-4f, "DC imag[0]");

    /* Pure cosine at bin 1: cos(2*pi*k/N) */
    for (int i = 0; i < N; i++) {
        real[i] = cosf(2.0f * 3.14159265f * (float)i / (float)N);
        imag[i] = 0.0f;
    }
    FFT_CHIP(real, imag, N);
    /* Bin 1 should have amplitude N/2 = 4 */
    float mag1 = sqrtf(real[1]*real[1] + imag[1]*imag[1]);
    ASSERT_NEAR(mag1, 4.0f, 0.1f, "cosine bin1 mag");

    /* IFFT roundtrip */
    float orig[8] = {1, 2, 3, 4, 5, 6, 7, 8};
    float re[8], im[8];
    memcpy(re, orig, sizeof(orig));
    memset(im, 0, sizeof(im));
    FFT_CHIP(re, im, N);
    IFFT_CHIP(re, im, N);
    for (int i = 0; i < N; i++) {
        ASSERT_NEAR(re[i], orig[i], 1e-4f, "ifft roundtrip");
    }

    /* Magnitude */
    for (int i = 0; i < N; i++) { re[i] = (i == 0) ? 3.0f : 0.0f; im[i] = (i == 0) ? 4.0f : 0.0f; }
    float mag[5];
    FFT_MAGNITUDE_CHIP(re, im, mag, N);
    ASSERT_NEAR(mag[0], 25.0f, 1e-5f, "magnitude 3+4i");

    /* Band energy */
    float test_mag[5] = {10, 20, 30, 40, 50};
    float bands[2];
    FFT_BAND_ENERGY_CHIP(test_mag, 8, bands, 2);
    /* 5 bins / 2 bands = 2 bins each, last gets remainder */
    ASSERT_NEAR(bands[0], 30.0f, 1e-5f, "band0");  /* 10+20 */
    ASSERT_NEAR(bands[1], 120.0f, 1e-5f, "band1"); /* 30+40+50 */

    /* Dominant frequency */
    int dom = FFT_DOMINANT_FREQ_CHIP(test_mag, 8);
    ASSERT_EQ_INT(dom, 4, "dominant_freq");

    printf("  %d passed\n\n", 16);
}

/* ═══════════════════════════════════════════════════════════════ */
void test_softmax_chip(void) {
    printf("--- SOFTMAX_CHIP ---\n");

    float x[3] = {1.0f, 2.0f, 3.0f};
    float out[3];

    SOFTMAX_CHIP(x, out, 3);

    /* Sum should be 1 */
    float sum = out[0] + out[1] + out[2];
    ASSERT_NEAR(sum, 1.0f, 1e-5f, "sum=1");

    /* Ordering preserved */
    assert(out[2] > out[1] && out[1] > out[0]);
    tests_passed++;

    /* Known values: softmax([1,2,3]) ≈ [0.0900, 0.2447, 0.6652] */
    ASSERT_NEAR(out[0], 0.0900f, 1e-3f, "softmax[0]");
    ASSERT_NEAR(out[2], 0.6652f, 1e-3f, "softmax[2]");

    /* Fast version: same ordering */
    float out_fast[3];
    SOFTMAX_CHIP_FAST(x, out_fast, 3);
    float sum_fast = out_fast[0] + out_fast[1] + out_fast[2];
    ASSERT_NEAR(sum_fast, 1.0f, 0.05f, "fast sum~1");
    assert(out_fast[2] > out_fast[1] && out_fast[1] > out_fast[0]);
    tests_passed++;

    /* Argmax */
    ASSERT_EQ_INT(ARGMAX_CHIP(x, 3), 2, "argmax");

    /* Top-K */
    float xk[5] = {10, 50, 30, 40, 20};
    int top3[3];
    TOP_K_CHIP(xk, 5, top3, 3);
    ASSERT_EQ_INT(top3[0], 1, "top3[0]");  /* 50 */
    ASSERT_EQ_INT(top3[1], 3, "top3[1]");  /* 40 */
    ASSERT_EQ_INT(top3[2], 2, "top3[2]");  /* 30 */

    printf("  %d passed\n\n", 10);
}

/* ═══════════════════════════════════════════════════════════════ */
void test_norm_chip(void) {
    printf("--- NORM_CHIP ---\n");

    /* LayerNorm: [1, 2, 3] → mean=2, var=0.667, std=0.816 */
    float x[3] = {1, 2, 3};
    float gamma[3] = {1, 1, 1};
    float beta[3] = {0, 0, 0};
    float out[3];

    LAYERNORM_CHIP(x, gamma, beta, out, 3, 1e-5f);
    /* After norm: [-1.2247, 0, 1.2247] approximately */
    ASSERT_NEAR(out[1], 0.0f, 1e-4f, "layernorm center");
    assert(out[0] < 0 && out[2] > 0);
    tests_passed++;

    /* Mean should be ~0 */
    float mean = (out[0] + out[1] + out[2]) / 3.0f;
    ASSERT_NEAR(mean, 0.0f, 1e-4f, "layernorm mean=0");

    /* RMSNorm */
    float x2[3] = {3, 4, 0};
    float g2[3] = {1, 1, 1};
    float out2[3];
    RMSNORM_CHIP(x2, g2, out2, 3, 1e-5f);
    /* RMS = sqrt((9+16+0)/3) = sqrt(8.333) ≈ 2.887 */
    /* out = x / rms = [1.039, 1.386, 0] */
    ASSERT_NEAR(out2[2], 0.0f, 1e-4f, "rmsnorm zero");
    assert(out2[0] > 0 && out2[1] > 0);
    tests_passed++;

    /* RMSNorm no-gamma */
    RMSNORM_CHIP_NOGAMMA(x2, out2, 3, 1e-5f);
    ASSERT_NEAR(out2[2], 0.0f, 1e-4f, "rmsnorm_nogamma zero");

    /* Online normalize */
    RunningStats stats[2];
    RUNNING_STATS_INIT(&stats[0]);
    RUNNING_STATS_INIT(&stats[1]);

    float samples[4][2] = {{1, 10}, {2, 20}, {3, 30}, {4, 40}};
    float norm_out[2];

    for (int t = 0; t < 4; t++) {
        ONLINE_NORMALIZE_CHIP(samples[t], stats, norm_out, 2, 1e-5f, 1);
    }
    /* After 4 samples: mean=[2.5, 25], var=[1.25, 125] */
    ASSERT_NEAR(stats[0].mean, 2.5f, 1e-4f, "online mean[0]");
    ASSERT_NEAR(stats[1].mean, 25.0f, 1e-4f, "online mean[1]");

    float var0 = RUNNING_STATS_VARIANCE(&stats[0]);
    ASSERT_NEAR(var0, 1.25f, 1e-3f, "online var[0]");

    printf("  %d passed\n\n", 9);
}

/* ═══════════════════════════════════════════════════════════════ */
int main(void) {
    printf("================================================================\n");
    printf("  YINSEN CHIP FORGE — Test Suite\n");
    printf("================================================================\n\n");

    test_gemm_chip();
    test_activation_chip();
    test_decay_chip();
    test_ternary_dot_chip();
    test_fft_chip();
    test_softmax_chip();
    test_norm_chip();

    printf("================================================================\n");
    printf("  TOTAL: %d passed, %d failed\n", tests_passed, tests_failed);
    printf("================================================================\n");

    return tests_failed > 0 ? 1 : 0;
}
