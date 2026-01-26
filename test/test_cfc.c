/*
 * YINSEN CfC Test Suite
 *
 * Verifies Closed-form Continuous-time neural network implementation.
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include "../include/cfc.h"

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
 * TEST WEIGHTS (2-input, 4-hidden, 2-output)
 * ============================================================================ */

#define TEST_INPUT_DIM  2
#define TEST_HIDDEN_DIM 4
#define TEST_OUTPUT_DIM 2
#define TEST_CONCAT_DIM (TEST_INPUT_DIM + TEST_HIDDEN_DIM)

static const float test_W_gate[TEST_HIDDEN_DIM * TEST_CONCAT_DIM] = {
    0.5f, 0.0f, 0.1f, 0.0f, 0.0f, 0.0f,
    0.0f, 0.5f, 0.0f, 0.1f, 0.0f, 0.0f,
    0.0f, 0.0f, 0.0f, 0.0f, 0.1f, 0.0f,
    0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.1f,
};
static const float test_b_gate[TEST_HIDDEN_DIM] = {0.0f, 0.0f, 0.0f, 0.0f};

static const float test_W_cand[TEST_HIDDEN_DIM * TEST_CONCAT_DIM] = {
    1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
    0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f,
    0.5f, 0.5f, 0.0f, 0.0f, 0.0f, 0.0f,
    0.0f, 0.0f, 0.5f, 0.5f, 0.0f, 0.0f,
};
static const float test_b_cand[TEST_HIDDEN_DIM] = {0.0f, 0.0f, 0.0f, 0.0f};

static const float test_tau[1] = {1.0f};

static const float test_W_out[TEST_HIDDEN_DIM * TEST_OUTPUT_DIM] = {
    1.0f, 0.0f,
    0.0f, 1.0f,
    0.5f, 0.0f,
    0.0f, 0.5f,
};
static const float test_b_out[TEST_OUTPUT_DIM] = {0.0f, 0.0f};

/* ============================================================================
 * TESTS
 * ============================================================================ */

void test_cfc_determinism(void) {
    printf("\n=== CfC Determinism ===\n");

    CfCParams params = {
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

    yinsen_cfc_cell(x, h_prev, 0.1f, &params, h_new1);
    yinsen_cfc_cell(x, h_prev, 0.1f, &params, h_new2);

    int match = 1;
    for (int i = 0; i < TEST_HIDDEN_DIM; i++) {
        if (h_new1[i] != h_new2[i]) match = 0;
    }
    TEST(match, "Identical calls produce identical results");
}

void test_cfc_bounded(void) {
    printf("\n=== CfC Bounded Outputs ===\n");

    CfCParams params = {
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

    yinsen_cfc_cell(x, h, 0.1f, &params, h);

    int bounded = 1;
    for (int i = 0; i < TEST_HIDDEN_DIM; i++) {
        if (fabsf(h[i]) > 2.0f) bounded = 0;
    }
    TEST(bounded, "State remains bounded with zero input");
}

void test_cfc_decay(void) {
    printf("\n=== CfC Decay ===\n");

    CfCParams params = {
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

    for (int t = 0; t < 100; t++) {
        float h_new[TEST_HIDDEN_DIM];
        yinsen_cfc_cell(x, h, 0.1f, &params, h_new);
        memcpy(h, h_new, sizeof(h));
    }

    float sum = 0.0f;
    for (int i = 0; i < TEST_HIDDEN_DIM; i++) {
        sum += fabsf(h[i]);
    }
    TEST(sum < 0.5f, "State decays over time");
}

void test_cfc_stability(void) {
    printf("\n=== CfC Numerical Stability (10000 iterations) ===\n");

    CfCParams params = {
        .input_dim = TEST_INPUT_DIM,
        .hidden_dim = TEST_HIDDEN_DIM,
        .W_gate = test_W_gate,
        .b_gate = test_b_gate,
        .W_cand = test_W_cand,
        .b_cand = test_b_cand,
        .tau = test_tau,
        .tau_shared = 1,
    };

    float h[TEST_HIDDEN_DIM] = {0.0f, 0.0f, 0.0f, 0.0f};
    float x[TEST_INPUT_DIM];

    int bounded = 1;
    for (int t = 0; t < 10000; t++) {
        x[0] = sinf(t * 0.1f);
        x[1] = cosf(t * 0.1f);

        float h_new[TEST_HIDDEN_DIM];
        yinsen_cfc_cell(x, h, 0.01f, &params, h_new);
        memcpy(h, h_new, sizeof(h));

        for (int i = 0; i < TEST_HIDDEN_DIM; i++) {
            if (fabsf(h[i]) > 100.0f || isnan(h[i]) || isinf(h[i])) {
                bounded = 0;
                break;
            }
        }
        if (!bounded) break;
    }
    TEST(bounded, "10000 iterations remain stable");
}

void test_cfc_output_softmax(void) {
    printf("\n=== CfC Output Softmax ===\n");

    CfCOutputParams out_params = {
        .hidden_dim = TEST_HIDDEN_DIM,
        .output_dim = TEST_OUTPUT_DIM,
        .W_out = test_W_out,
        .b_out = test_b_out,
    };

    float h[TEST_HIDDEN_DIM] = {0.5f, 0.3f, 0.1f, 0.2f};
    float probs[TEST_OUTPUT_DIM];

    yinsen_cfc_output_softmax(h, &out_params, probs);

    float sum = 0.0f;
    for (int i = 0; i < TEST_OUTPUT_DIM; i++) {
        sum += probs[i];
    }
    TEST(FLOAT_EQ(sum, 1.0f, 1e-5f), "Softmax sums to 1");

    int positive = 1;
    for (int i = 0; i < TEST_OUTPUT_DIM; i++) {
        if (probs[i] < 0.0f) positive = 0;
    }
    TEST(positive, "Softmax outputs are positive");
}

/* ============================================================================
 * MAIN
 * ============================================================================ */

int main(void) {
    printf("===================================================\n");
    printf("  YINSEN CfC TEST SUITE\n");
    printf("===================================================\n");

    test_cfc_determinism();
    test_cfc_bounded();
    test_cfc_decay();
    test_cfc_stability();
    test_cfc_output_softmax();

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
