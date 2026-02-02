/*
 * YINSEN CfC Ternary Test Suite
 *
 * Verifies CfC with ternary weights.
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include "../include/cfc_ternary.h"

/* ============================================================================
 * TEST FRAMEWORK
 * ============================================================================ */

static int tests_run = 0;
static int tests_passed = 0;

#define FLOAT_EQ(a, b, tol) (fabsf((a) - (b)) < (tol))

#define TEST(cond, name) do { \
    tests_run++; \
    if (cond) { tests_passed++; } \
    else { printf("  FAIL: %s\n", name); } \
} while(0)

/* ============================================================================
 * TEST WEIGHTS (2-input, 4-hidden)
 * ============================================================================ */

#define TEST_INPUT_DIM  2
#define TEST_HIDDEN_DIM 4
#define TEST_OUTPUT_DIM 2
#define TEST_CONCAT_DIM (TEST_INPUT_DIM + TEST_HIDDEN_DIM)
#define TEST_BYTES_PER_ROW ((TEST_CONCAT_DIM + 3) / 4)  /* 2 bytes for 6 trits */

/* Ternary weights: simple patterns for testing */
static uint8_t test_W_gate[TEST_HIDDEN_DIM * TEST_BYTES_PER_ROW];
static uint8_t test_W_cand[TEST_HIDDEN_DIM * TEST_BYTES_PER_ROW];
static uint8_t test_W_out[TEST_OUTPUT_DIM * ((TEST_HIDDEN_DIM + 3) / 4)];

static const float test_b_gate[TEST_HIDDEN_DIM] = {0.0f, 0.0f, 0.0f, 0.0f};
static const float test_b_cand[TEST_HIDDEN_DIM] = {0.0f, 0.0f, 0.0f, 0.0f};
static const float test_b_out[TEST_OUTPUT_DIM] = {0.0f, 0.0f};
static const float test_tau[1] = {1.0f};

void init_test_weights(void) {
    /* Initialize W_gate: alternating +1, -1 pattern */
    for (int i = 0; i < TEST_HIDDEN_DIM; i++) {
        /* Row i: [+1, -1, +1, -1, +1, -1] */
        test_W_gate[i * TEST_BYTES_PER_ROW + 0] = trit_pack4(1, -1, 1, -1);
        test_W_gate[i * TEST_BYTES_PER_ROW + 1] = trit_pack4(1, -1, 0, 0);
    }

    /* Initialize W_cand: all +1 */
    for (int i = 0; i < TEST_HIDDEN_DIM; i++) {
        test_W_cand[i * TEST_BYTES_PER_ROW + 0] = trit_pack4(1, 1, 1, 1);
        test_W_cand[i * TEST_BYTES_PER_ROW + 1] = trit_pack4(1, 1, 0, 0);
    }

    /* Initialize W_out: identity-ish */
    int out_bytes_per_row = (TEST_HIDDEN_DIM + 3) / 4;
    test_W_out[0 * out_bytes_per_row + 0] = trit_pack4(1, 0, 0, 0);
    test_W_out[1 * out_bytes_per_row + 0] = trit_pack4(0, 1, 0, 0);
}

/* ============================================================================
 * TESTS
 * ============================================================================ */

void test_cfc_ternary_determinism(void) {
    printf("\n=== CfC Ternary Determinism ===\n");

    init_test_weights();

    CfCTernaryParams params = {
        .input_dim = TEST_INPUT_DIM,
        .hidden_dim = TEST_HIDDEN_DIM,
        .W_gate = test_W_gate,
        .b_gate = test_b_gate,
        .W_cand = test_W_cand,
        .b_cand = test_b_cand,
        .tau = test_tau,
        .tau_shared = 1,
    };

    float x[TEST_INPUT_DIM] = {0.5f, -0.3f};
    float h_prev[TEST_HIDDEN_DIM] = {0.1f, 0.2f, 0.0f, 0.0f};
    float h_new1[TEST_HIDDEN_DIM], h_new2[TEST_HIDDEN_DIM];

    yinsen_cfc_ternary_cell(x, h_prev, 0.1f, &params, h_new1);
    yinsen_cfc_ternary_cell(x, h_prev, 0.1f, &params, h_new2);

    int match = 1;
    for (int i = 0; i < TEST_HIDDEN_DIM; i++) {
        if (h_new1[i] != h_new2[i]) match = 0;
    }
    TEST(match, "Identical calls produce identical results");
}

void test_cfc_ternary_bounded(void) {
    printf("\n=== CfC Ternary Bounded ===\n");

    init_test_weights();

    CfCTernaryParams params = {
        .input_dim = TEST_INPUT_DIM,
        .hidden_dim = TEST_HIDDEN_DIM,
        .W_gate = test_W_gate,
        .b_gate = test_b_gate,
        .W_cand = test_W_cand,
        .b_cand = test_b_cand,
        .tau = test_tau,
        .tau_shared = 1,
    };

    float x[TEST_INPUT_DIM] = {0.0f, 0.0f};
    float h[TEST_HIDDEN_DIM] = {1.0f, 1.0f, 1.0f, 1.0f};

    yinsen_cfc_ternary_cell(x, h, 0.1f, &params, h);

    int bounded = 1;
    for (int i = 0; i < TEST_HIDDEN_DIM; i++) {
        if (fabsf(h[i]) > 2.0f) bounded = 0;
    }
    TEST(bounded, "State remains bounded");
}

void test_cfc_ternary_stability(void) {
    printf("\n=== CfC Ternary Stability (1000 iterations) ===\n");

    init_test_weights();

    CfCTernaryParams params = {
        .input_dim = TEST_INPUT_DIM,
        .hidden_dim = TEST_HIDDEN_DIM,
        .W_gate = test_W_gate,
        .b_gate = test_b_gate,
        .W_cand = test_W_cand,
        .b_cand = test_b_cand,
        .tau = test_tau,
        .tau_shared = 1,
    };

    float h[TEST_HIDDEN_DIM] = {0};
    float x[TEST_INPUT_DIM];

    int stable = 1;
    for (int t = 0; t < 1000; t++) {
        x[0] = sinf(t * 0.1f);
        x[1] = cosf(t * 0.1f);

        float h_new[TEST_HIDDEN_DIM];
        yinsen_cfc_ternary_cell(x, h, 0.01f, &params, h_new);
        memcpy(h, h_new, sizeof(h));

        for (int i = 0; i < TEST_HIDDEN_DIM; i++) {
            if (fabsf(h[i]) > 100.0f || isnan(h[i]) || isinf(h[i])) {
                stable = 0;
                break;
            }
        }
        if (!stable) break;
    }
    TEST(stable, "1000 iterations remain stable");
}

void test_cfc_ternary_output(void) {
    printf("\n=== CfC Ternary Output ===\n");

    init_test_weights();

    CfCTernaryOutputParams out_params = {
        .hidden_dim = TEST_HIDDEN_DIM,
        .output_dim = TEST_OUTPUT_DIM,
        .W_out = test_W_out,
        .b_out = test_b_out,
    };

    float h[TEST_HIDDEN_DIM] = {0.5f, 0.3f, 0.1f, 0.2f};
    float probs[TEST_OUTPUT_DIM];

    yinsen_cfc_ternary_output_softmax(h, &out_params, probs);

    float sum = probs[0] + probs[1];
    TEST(FLOAT_EQ(sum, 1.0f, 1e-5f), "Softmax sums to 1");
}

void test_cfc_ternary_memory(void) {
    printf("\n=== CfC Ternary Memory Comparison ===\n");

    init_test_weights();

    CfCTernaryParams params = {
        .input_dim = TEST_INPUT_DIM,
        .hidden_dim = TEST_HIDDEN_DIM,
        .W_gate = test_W_gate,
        .b_gate = test_b_gate,
        .W_cand = test_W_cand,
        .b_cand = test_b_cand,
        .tau = test_tau,
        .tau_shared = 1,
    };

    size_t tern_bytes, float_bytes;
    float ratio;

    yinsen_cfc_ternary_memory_comparison(&params, &tern_bytes, &float_bytes, &ratio);

    printf("  Ternary: %zu bytes\n", tern_bytes);
    printf("  Float:   %zu bytes\n", float_bytes);
    printf("  Ratio:   %.1fx\n", ratio);

    TEST(tern_bytes < float_bytes, "Ternary uses less memory than float");
    TEST(ratio > 2.0f, "At least 2x compression");
}

/* ============================================================================
 * PER-ELEMENT TAU TESTS (verifies the decay[] initialization fix)
 * ============================================================================ */

void test_cfc_ternary_tau_per_element_valid(void) {
    printf("\n=== CfC Ternary Per-Element Tau (all valid) ===\n");

    init_test_weights();

    /* Per-element tau, all valid */
    float tau_per[TEST_HIDDEN_DIM] = {1.0f, 0.5f, 2.0f, 1.5f};

    CfCTernaryParams params = {
        .input_dim = TEST_INPUT_DIM,
        .hidden_dim = TEST_HIDDEN_DIM,
        .W_gate = test_W_gate,
        .b_gate = test_b_gate,
        .W_cand = test_W_cand,
        .b_cand = test_b_cand,
        .tau = tau_per,
        .tau_shared = 0,
    };

    float x[TEST_INPUT_DIM] = {0.5f, -0.3f};
    float h_prev[TEST_HIDDEN_DIM] = {0.1f, 0.2f, 0.0f, 0.0f};
    float h_new[TEST_HIDDEN_DIM];

    yinsen_cfc_ternary_cell(x, h_prev, 0.1f, &params, h_new);

    int all_finite = 1;
    for (int i = 0; i < TEST_HIDDEN_DIM; i++) {
        if (isnan(h_new[i]) || isinf(h_new[i])) all_finite = 0;
    }
    TEST(all_finite, "Per-element tau (all valid) produces finite output");

    /* Determinism check */
    float h_new2[TEST_HIDDEN_DIM];
    yinsen_cfc_ternary_cell(x, h_prev, 0.1f, &params, h_new2);
    int match = 1;
    for (int i = 0; i < TEST_HIDDEN_DIM; i++) {
        if (h_new[i] != h_new2[i]) match = 0;
    }
    TEST(match, "Per-element tau is deterministic");
}

void test_cfc_ternary_tau_per_element_some_invalid(void) {
    printf("\n=== CfC Ternary Per-Element Tau (some invalid) ===\n");

    init_test_weights();

    /* tau[1] and tau[3] are invalid (<= 0) */
    float tau_per[TEST_HIDDEN_DIM] = {1.0f, 0.0f, 2.0f, -1.0f};

    CfCTernaryParams params = {
        .input_dim = TEST_INPUT_DIM,
        .hidden_dim = TEST_HIDDEN_DIM,
        .W_gate = test_W_gate,
        .b_gate = test_b_gate,
        .W_cand = test_W_cand,
        .b_cand = test_b_cand,
        .tau = tau_per,
        .tau_shared = 0,
    };

    float x[TEST_INPUT_DIM] = {0.5f, -0.3f};
    float h_prev[TEST_HIDDEN_DIM] = {0.1f, 0.2f, 0.3f, 0.4f};
    float h_new[TEST_HIDDEN_DIM];

    yinsen_cfc_ternary_cell(x, h_prev, 0.1f, &params, h_new);

    /* Valid tau indices should produce finite results */
    TEST(!isnan(h_new[0]) && !isinf(h_new[0]),
         "tau[0]=1.0 (valid) -> finite h_new[0]");
    TEST(!isnan(h_new[2]) && !isinf(h_new[2]),
         "tau[2]=2.0 (valid) -> finite h_new[2]");

    /* Invalid tau indices should produce NAN */
    TEST(isnan(h_new[1]),
         "tau[1]=0.0 (invalid) -> NAN h_new[1]");
    TEST(isnan(h_new[3]),
         "tau[3]=-1.0 (invalid) -> NAN h_new[3]");
}

void test_cfc_ternary_tau_per_element_all_invalid(void) {
    printf("\n=== CfC Ternary Per-Element Tau (all invalid) ===\n");

    init_test_weights();

    float tau_per[TEST_HIDDEN_DIM] = {0.0f, -1.0f, -0.5f, 0.0f};

    CfCTernaryParams params = {
        .input_dim = TEST_INPUT_DIM,
        .hidden_dim = TEST_HIDDEN_DIM,
        .W_gate = test_W_gate,
        .b_gate = test_b_gate,
        .W_cand = test_W_cand,
        .b_cand = test_b_cand,
        .tau = tau_per,
        .tau_shared = 0,
    };

    float x[TEST_INPUT_DIM] = {0.5f, -0.3f};
    float h_prev[TEST_HIDDEN_DIM] = {0.1f, 0.2f, 0.3f, 0.4f};
    float h_new[TEST_HIDDEN_DIM];

    yinsen_cfc_ternary_cell(x, h_prev, 0.1f, &params, h_new);

    int all_nan = 1;
    for (int i = 0; i < TEST_HIDDEN_DIM; i++) {
        if (!isnan(h_new[i])) all_nan = 0;
    }
    TEST(all_nan, "All tau <= 0 -> all h_new are NAN");
}

/* ============================================================================
 * MAIN
 * ============================================================================ */

int main(void) {
    printf("===================================================\n");
    printf("  YINSEN CfC TERNARY TEST SUITE\n");
    printf("===================================================\n");

    test_cfc_ternary_determinism();
    test_cfc_ternary_bounded();
    test_cfc_ternary_stability();
    test_cfc_ternary_output();
    test_cfc_ternary_memory();
    test_cfc_ternary_tau_per_element_valid();
    test_cfc_ternary_tau_per_element_some_invalid();
    test_cfc_ternary_tau_per_element_all_invalid();

    printf("\n===================================================\n");
    printf("  RESULTS: %d/%d passed\n", tests_passed, tests_run);
    printf("===================================================\n");

    if (tests_passed == tests_run) {
        printf("  ALL TESTS PASSED\n");
        return 0;
    } else {
        printf("  SOME TESTS FAILED\n");
        return 1;
    }
}
