/*
 * Test Suite: Q15 Fixed-Point CfC vs Float Path
 *
 * Validates that the Q15 fixed-point implementation produces results
 * close enough to the float path for anomaly detection to work.
 *
 * Tests:
 *   1. Activation LUT accuracy (Q15 vs libm)
 *   2. Q15 arithmetic primitives
 *   3. CFC_CELL_SPARSE_Q15 vs CFC_CELL_SPARSE (float) comparison
 *   4. Multi-step divergence (drift over 1000 steps)
 *   5. PCA discriminant Q15 vs float
 *
 * Compile:
 *   cc -O2 -I include -I include/chips test/test_q15.c -lm -o test/test_q15
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

#include "activation_chip.h"
#include "cfc_cell_chip.h"
#include "activation_q15.h"
#include "cfc_cell_q15.h"

/* ============================================================================
 * TEST FRAMEWORK
 * ============================================================================ */

static int tests_run = 0;
static int tests_passed = 0;

#define TEST(cond, msg) do { \
    tests_run++; \
    if (cond) { tests_passed++; printf("  PASS: %s\n", msg); } \
    else { printf("  FAIL: %s\n", msg); } \
} while(0)

/* ============================================================================
 * TEST 1: Activation LUT Accuracy
 * ============================================================================ */

void test_sigmoid_q15_accuracy(void) {
    printf("\n=== Q15 Sigmoid LUT Accuracy ===\n");

    float max_err = 0.0f;
    float sum_err = 0.0f;
    int n_samples = 0;

    /* Sweep input range [-8, +8] in fine steps */
    for (float x = -8.0f; x <= 8.0f; x += 0.01f) {
        float ref = 1.0f / (1.0f + expf(-x));
        int16_t x_q11 = float_to_q11(x);
        int16_t result_q15 = SIGMOID_Q15(x_q11);
        float result_f = q15_to_float(result_q15);

        float err = fabsf(ref - result_f);
        if (err > max_err) max_err = err;
        sum_err += err;
        n_samples++;
    }

    float avg_err = sum_err / n_samples;
    printf("  Max error: %.6f\n", max_err);
    printf("  Avg error: %.6f\n", avg_err);
    printf("  Samples:   %d\n", n_samples);

    TEST(max_err < 0.005f, "Sigmoid Q15 max error < 0.005");
    TEST(avg_err < 0.001f, "Sigmoid Q15 avg error < 0.001");
}

void test_tanh_q15_accuracy(void) {
    printf("\n=== Q15 Tanh LUT Accuracy ===\n");

    float max_err = 0.0f;
    float sum_err = 0.0f;
    int n_samples = 0;

    for (float x = -8.0f; x <= 8.0f; x += 0.01f) {
        float ref = tanhf(x);
        int16_t x_q11 = float_to_q11(x);
        int16_t result_q15 = TANH_Q15(x_q11);
        float result_f = q15_to_float(result_q15);

        float err = fabsf(ref - result_f);
        if (err > max_err) max_err = err;
        sum_err += err;
        n_samples++;
    }

    float avg_err = sum_err / n_samples;
    printf("  Max error: %.6f\n", max_err);
    printf("  Avg error: %.6f\n", avg_err);
    printf("  Samples:   %d\n", n_samples);

    TEST(max_err < 0.005f, "Tanh Q15 max error < 0.005");
    TEST(avg_err < 0.001f, "Tanh Q15 avg error < 0.001");
}

/* ============================================================================
 * TEST 2: Q15 Arithmetic Primitives
 * ============================================================================ */

void test_q15_arithmetic(void) {
    printf("\n=== Q15 Arithmetic ===\n");

    /* q15_mul: 0.5 * 0.5 = 0.25 */
    int16_t half = Q15_HALF;
    int16_t quarter = q15_mul(half, half);
    float quarter_f = q15_to_float(quarter);
    TEST(fabsf(quarter_f - 0.25f) < 0.001f, "Q15 mul: 0.5 * 0.5 ≈ 0.25");

    /* q15_mul: -0.5 * 0.5 = -0.25 */
    int16_t neg_half = -Q15_HALF;
    int16_t neg_quarter = q15_mul(neg_half, half);
    float neg_quarter_f = q15_to_float(neg_quarter);
    TEST(fabsf(neg_quarter_f - (-0.25f)) < 0.001f, "Q15 mul: -0.5 * 0.5 ≈ -0.25");

    /* q15_mul: 1.0 * 1.0 (near overflow) */
    int16_t one_sq = q15_mul(Q15_ONE, Q15_ONE);
    TEST(one_sq > 32700, "Q15 mul: ~1.0 * ~1.0 ≈ ~1.0");

    /* Saturating add at limits */
    int16_t sat = q15_sat_add(Q15_ONE, Q15_HALF);
    TEST(sat == 32767, "Q15 sat_add clamps at max");

    int16_t sat2 = q15_sat_sub(Q15_NEG_ONE, Q15_HALF);
    TEST(sat2 == -32768, "Q15 sat_sub clamps at min");

    /* Format conversions */
    int16_t one_q11 = float_to_q11(1.0f);
    TEST(one_q11 == 2048, "float_to_q11(1.0) == 2048");

    int16_t one_q15 = float_to_q15(0.5f);
    TEST(abs(one_q15 - 16384) < 2, "float_to_q15(0.5) ≈ 16384");

    float roundtrip = q15_to_float(float_to_q15(0.333f));
    TEST(fabsf(roundtrip - 0.333f) < 0.001f, "Q15 roundtrip: 0.333 → Q15 → float");
}

/* ============================================================================
 * TEST 3: CFC_CELL_SPARSE_Q15 vs CFC_CELL_SPARSE (Single Step)
 * ============================================================================ */

#define TEST_INPUT_DIM  2
#define TEST_HIDDEN_DIM 8
#define TEST_CONCAT_DIM (TEST_INPUT_DIM + TEST_HIDDEN_DIM)

/* Generate deterministic pseudo-random ternary weights */
static void make_ternary_weights(float* W, int rows, int cols, unsigned seed) {
    for (int i = 0; i < rows * cols; i++) {
        seed = seed * 1103515245 + 12345;
        int r = (seed >> 16) % 5;  /* 0,1,2,3,4 → more zeros (60% sparse) */
        if (r == 0) W[i] = 1.0f;
        else if (r == 1) W[i] = -1.0f;
        else W[i] = 0.0f;
    }
}

void test_cfc_q15_vs_float_single_step(void) {
    printf("\n=== CfC Q15 vs Float (Single Step) ===\n");

    /* Create shared weights */
    float W_gate[TEST_HIDDEN_DIM * TEST_CONCAT_DIM];
    float W_cand[TEST_HIDDEN_DIM * TEST_CONCAT_DIM];
    float b_gate[TEST_HIDDEN_DIM] = {0.1f, -0.2f, 0.05f, -0.1f, 0.15f, -0.05f, 0.2f, -0.15f};
    float b_cand[TEST_HIDDEN_DIM] = {-0.1f, 0.1f, -0.05f, 0.2f, -0.15f, 0.1f, -0.2f, 0.05f};
    float tau[1] = {1.0f};
    float dt = 0.01f;

    make_ternary_weights(W_gate, TEST_HIDDEN_DIM, TEST_CONCAT_DIM, 42);
    make_ternary_weights(W_cand, TEST_HIDDEN_DIM, TEST_CONCAT_DIM, 137);

    /* Build sparse weights (shared between float and Q15 paths) */
    CfcSparseWeights sw;
    cfc_build_sparse(W_gate, W_cand, 0.5f, TEST_HIDDEN_DIM, TEST_CONCAT_DIM, 1, &sw);

    /* Float path setup */
    float decay_f[TEST_HIDDEN_DIM];
    cfc_precompute_decay(tau, 1, dt, TEST_HIDDEN_DIM, decay_f);

    float x_f[TEST_INPUT_DIM] = {0.5f, -0.3f};
    float h_prev_f[TEST_HIDDEN_DIM] = {0.0f};
    float h_new_f[TEST_HIDDEN_DIM];

    ACTIVATION_LUT_INIT();
    CFC_CELL_SPARSE(x_f, h_prev_f, &sw, b_gate, b_cand, decay_f,
                     TEST_INPUT_DIM, TEST_HIDDEN_DIM, h_new_f);

    /* Q15 path setup */
    int16_t decay_q15[TEST_HIDDEN_DIM];
    cfc_precompute_decay_q15(tau, 1, dt, TEST_HIDDEN_DIM, decay_q15);

    int16_t b_gate_q11[TEST_HIDDEN_DIM];
    int16_t b_cand_q11[TEST_HIDDEN_DIM];
    cfc_convert_biases_q11(b_gate, TEST_HIDDEN_DIM, b_gate_q11);
    cfc_convert_biases_q11(b_cand, TEST_HIDDEN_DIM, b_cand_q11);

    int16_t x_q11[TEST_INPUT_DIM];
    cfc_convert_input_q11(x_f, TEST_INPUT_DIM, x_q11);

    int16_t h_prev_q15[TEST_HIDDEN_DIM] = {0};
    int16_t h_new_q15[TEST_HIDDEN_DIM];

    Q15_LUT_INIT();
    CFC_CELL_SPARSE_Q15(x_q11, h_prev_q15, &sw, b_gate_q11, b_cand_q11,
                         decay_q15, TEST_INPUT_DIM, TEST_HIDDEN_DIM, h_new_q15);

    /* Compare */
    float h_new_q15_as_float[TEST_HIDDEN_DIM];
    cfc_convert_state_to_float(h_new_q15, TEST_HIDDEN_DIM, h_new_q15_as_float);

    float max_err = 0.0f;
    float sum_err = 0.0f;
    for (int i = 0; i < TEST_HIDDEN_DIM; i++) {
        float err = fabsf(h_new_f[i] - h_new_q15_as_float[i]);
        if (err > max_err) max_err = err;
        sum_err += err;
        printf("  h[%d]: float=%.6f  q15=%.6f  err=%.6f\n",
               i, h_new_f[i], h_new_q15_as_float[i], err);
    }

    float avg_err = sum_err / TEST_HIDDEN_DIM;
    printf("  Max error: %.6f\n", max_err);
    printf("  Avg error: %.6f\n", avg_err);

    TEST(max_err < 0.05f, "Single step max error < 0.05");
    TEST(avg_err < 0.02f, "Single step avg error < 0.02");
}

/* ============================================================================
 * TEST 4: Multi-Step Divergence
 * ============================================================================ */

void test_cfc_q15_multi_step_divergence(void) {
    printf("\n=== CfC Q15 vs Float (1000 Steps Divergence) ===\n");

    /* Setup (same as single-step) */
    float W_gate[TEST_HIDDEN_DIM * TEST_CONCAT_DIM];
    float W_cand[TEST_HIDDEN_DIM * TEST_CONCAT_DIM];
    float b_gate[TEST_HIDDEN_DIM] = {0.1f, -0.2f, 0.05f, -0.1f, 0.15f, -0.05f, 0.2f, -0.15f};
    float b_cand[TEST_HIDDEN_DIM] = {-0.1f, 0.1f, -0.05f, 0.2f, -0.15f, 0.1f, -0.2f, 0.05f};
    float tau[1] = {1.0f};
    float dt = 0.01f;

    make_ternary_weights(W_gate, TEST_HIDDEN_DIM, TEST_CONCAT_DIM, 42);
    make_ternary_weights(W_cand, TEST_HIDDEN_DIM, TEST_CONCAT_DIM, 137);

    CfcSparseWeights sw;
    cfc_build_sparse(W_gate, W_cand, 0.5f, TEST_HIDDEN_DIM, TEST_CONCAT_DIM, 1, &sw);

    /* Float path */
    float decay_f[TEST_HIDDEN_DIM];
    cfc_precompute_decay(tau, 1, dt, TEST_HIDDEN_DIM, decay_f);
    ACTIVATION_LUT_INIT();

    float h_f[TEST_HIDDEN_DIM] = {0};

    /* Q15 path */
    int16_t decay_q15[TEST_HIDDEN_DIM];
    cfc_precompute_decay_q15(tau, 1, dt, TEST_HIDDEN_DIM, decay_q15);

    int16_t b_gate_q11[TEST_HIDDEN_DIM], b_cand_q11[TEST_HIDDEN_DIM];
    cfc_convert_biases_q11(b_gate, TEST_HIDDEN_DIM, b_gate_q11);
    cfc_convert_biases_q11(b_cand, TEST_HIDDEN_DIM, b_cand_q11);
    Q15_LUT_INIT();

    int16_t h_q[TEST_HIDDEN_DIM] = {0};

    /* Run both paths for 1000 steps with a synthetic signal */
    int n_steps = 1000;
    float divergence_at[4] = {0};  /* Track at 10, 100, 500, 1000 */

    for (int t = 0; t < n_steps; t++) {
        /* Synthetic input: slow sine + noise-like component */
        float x_f[TEST_INPUT_DIM];
        x_f[0] = 0.5f * sinf(0.01f * t);
        x_f[1] = 0.3f * cosf(0.023f * t);

        int16_t x_q11[TEST_INPUT_DIM];
        cfc_convert_input_q11(x_f, TEST_INPUT_DIM, x_q11);

        float h_new_f[TEST_HIDDEN_DIM];
        int16_t h_new_q[TEST_HIDDEN_DIM];

        CFC_CELL_SPARSE(x_f, h_f, &sw, b_gate, b_cand, decay_f,
                         TEST_INPUT_DIM, TEST_HIDDEN_DIM, h_new_f);
        CFC_CELL_SPARSE_Q15(x_q11, h_q, &sw, b_gate_q11, b_cand_q11,
                             decay_q15, TEST_INPUT_DIM, TEST_HIDDEN_DIM, h_new_q);

        memcpy(h_f, h_new_f, sizeof(h_f));
        memcpy(h_q, h_new_q, sizeof(h_q));

        /* Compute L2 divergence at checkpoints */
        if (t == 9 || t == 99 || t == 499 || t == 999) {
            float l2 = 0.0f;
            for (int i = 0; i < TEST_HIDDEN_DIM; i++) {
                float diff = h_f[i] - q15_to_float(h_q[i]);
                l2 += diff * diff;
            }
            l2 = sqrtf(l2);

            int idx = (t == 9) ? 0 : (t == 99) ? 1 : (t == 499) ? 2 : 3;
            divergence_at[idx] = l2;
            printf("  Step %4d: L2 divergence = %.6f\n", t + 1, l2);
        }
    }

    TEST(divergence_at[0] < 0.1f, "Divergence at step 10 < 0.1");
    TEST(divergence_at[1] < 0.2f, "Divergence at step 100 < 0.2");
    TEST(divergence_at[2] < 0.3f, "Divergence at step 500 < 0.3");
    TEST(divergence_at[3] < 0.5f, "Divergence at step 1000 < 0.5");
}

/* ============================================================================
 * TEST 5: PCA Discriminant Q15 vs Float
 * ============================================================================ */

void test_pca_q15(void) {
    printf("\n=== PCA Discriminant Q15 ===\n");

    /* Synthetic enrolled mean and PCs */
    float mean_f[TEST_HIDDEN_DIM] = {0.1f, -0.2f, 0.05f, 0.3f, -0.1f, 0.15f, -0.25f, 0.08f};
    float pcs_f[5 * TEST_HIDDEN_DIM];  /* 5 PCs */

    /* Fill PCs with orthogonal-ish vectors */
    for (int pc = 0; pc < 5; pc++) {
        for (int i = 0; i < TEST_HIDDEN_DIM; i++) {
            pcs_f[pc * TEST_HIDDEN_DIM + i] =
                (pc == i) ? 0.9f :
                (pc + 1 == i) ? 0.3f : 0.0f;
        }
    }

    /* Convert to Q15 */
    int16_t mean_q15[TEST_HIDDEN_DIM];
    int16_t pcs_q15[5 * TEST_HIDDEN_DIM];
    for (int i = 0; i < TEST_HIDDEN_DIM; i++) {
        mean_q15[i] = float_to_q15(mean_f[i]);
    }
    for (int i = 0; i < 5 * TEST_HIDDEN_DIM; i++) {
        pcs_q15[i] = float_to_q15(pcs_f[i]);
    }

    /* Test with a "normal" hidden state (close to mean) */
    float h_normal_f[TEST_HIDDEN_DIM] = {0.12f, -0.18f, 0.06f, 0.28f, -0.12f, 0.14f, -0.23f, 0.09f};
    int16_t h_normal_q15[TEST_HIDDEN_DIM];
    for (int i = 0; i < TEST_HIDDEN_DIM; i++) {
        h_normal_q15[i] = float_to_q15(h_normal_f[i]);
    }

    int64_t score_normal = cfc_pca_score_q15(h_normal_q15, mean_q15, pcs_q15,
                                               TEST_HIDDEN_DIM, 5);

    /* Test with an "anomalous" hidden state (far from mean) */
    float h_anomaly_f[TEST_HIDDEN_DIM] = {0.8f, 0.7f, -0.9f, -0.6f, 0.85f, -0.75f, 0.65f, -0.8f};
    int16_t h_anomaly_q15[TEST_HIDDEN_DIM];
    for (int i = 0; i < TEST_HIDDEN_DIM; i++) {
        h_anomaly_q15[i] = float_to_q15(h_anomaly_f[i]);
    }

    int64_t score_anomaly = cfc_pca_score_q15(h_anomaly_q15, mean_q15, pcs_q15,
                                                TEST_HIDDEN_DIM, 5);

    printf("  Normal score:  %lld\n", (long long)score_normal);
    printf("  Anomaly score: %lld\n", (long long)score_anomaly);

    TEST(score_anomaly > score_normal, "Anomaly score > Normal score");
    TEST(score_anomaly > 10 * score_normal, "Anomaly score >> Normal score (10x separation)");
}

/* ============================================================================
 * TEST 6: Determinism
 * ============================================================================ */

void test_q15_determinism(void) {
    printf("\n=== Q15 Determinism ===\n");

    float W_gate[TEST_HIDDEN_DIM * TEST_CONCAT_DIM];
    float W_cand[TEST_HIDDEN_DIM * TEST_CONCAT_DIM];
    float b_gate[TEST_HIDDEN_DIM] = {0.1f, -0.2f, 0.05f, -0.1f, 0.15f, -0.05f, 0.2f, -0.15f};
    float b_cand[TEST_HIDDEN_DIM] = {-0.1f, 0.1f, -0.05f, 0.2f, -0.15f, 0.1f, -0.2f, 0.05f};
    float tau[1] = {1.0f};
    float dt = 0.01f;

    make_ternary_weights(W_gate, TEST_HIDDEN_DIM, TEST_CONCAT_DIM, 42);
    make_ternary_weights(W_cand, TEST_HIDDEN_DIM, TEST_CONCAT_DIM, 137);

    CfcSparseWeights sw;
    cfc_build_sparse(W_gate, W_cand, 0.5f, TEST_HIDDEN_DIM, TEST_CONCAT_DIM, 1, &sw);

    int16_t decay_q15[TEST_HIDDEN_DIM];
    cfc_precompute_decay_q15(tau, 1, dt, TEST_HIDDEN_DIM, decay_q15);

    int16_t b_gate_q11[TEST_HIDDEN_DIM], b_cand_q11[TEST_HIDDEN_DIM];
    cfc_convert_biases_q11(b_gate, TEST_HIDDEN_DIM, b_gate_q11);
    cfc_convert_biases_q11(b_cand, TEST_HIDDEN_DIM, b_cand_q11);

    int16_t x_q11[TEST_INPUT_DIM] = {float_to_q11(0.5f), float_to_q11(-0.3f)};

    /* Run twice, same input */
    int16_t h1[TEST_HIDDEN_DIM] = {0}, h2[TEST_HIDDEN_DIM] = {0};
    int16_t out1[TEST_HIDDEN_DIM], out2[TEST_HIDDEN_DIM];

    for (int t = 0; t < 100; t++) {
        CFC_CELL_SPARSE_Q15(x_q11, h1, &sw, b_gate_q11, b_cand_q11,
                             decay_q15, TEST_INPUT_DIM, TEST_HIDDEN_DIM, out1);
        CFC_CELL_SPARSE_Q15(x_q11, h2, &sw, b_gate_q11, b_cand_q11,
                             decay_q15, TEST_INPUT_DIM, TEST_HIDDEN_DIM, out2);
        memcpy(h1, out1, sizeof(h1));
        memcpy(h2, out2, sizeof(h2));
    }

    int match = 1;
    for (int i = 0; i < TEST_HIDDEN_DIM; i++) {
        if (h1[i] != h2[i]) { match = 0; break; }
    }
    TEST(match, "Q15 path is bit-exact deterministic over 100 steps");
}

/* ============================================================================
 * MAIN
 * ============================================================================ */

int main(void) {
    printf("===================================================\n");
    printf("  YINSEN Q15 FIXED-POINT TEST SUITE\n");
    printf("===================================================\n");

    /* Init both LUT systems */
    ACTIVATION_LUT_INIT();
    Q15_LUT_INIT();

    test_sigmoid_q15_accuracy();
    test_tanh_q15_accuracy();
    test_q15_arithmetic();
    test_cfc_q15_vs_float_single_step();
    test_cfc_q15_multi_step_divergence();
    test_pca_q15();
    test_q15_determinism();

    printf("\n===================================================\n");
    printf("  RESULTS: %d/%d passed\n", tests_passed, tests_run);
    if (tests_passed == tests_run) {
        printf("  ALL TESTS PASSED\n");
    } else {
        printf("  %d FAILURES\n", tests_run - tests_passed);
    }
    printf("===================================================\n");

    return (tests_passed == tests_run) ? 0 : 1;
}
