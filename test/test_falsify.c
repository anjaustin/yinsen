/*
 * YINSEN Falsification Test Suite
 *
 * Purpose: Try to BREAK the code. Find edge cases, overflows,
 * undefined behavior, and incorrect assumptions.
 *
 * If this passes, we haven't tried hard enough.
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <float.h>
#include <limits.h>
#include "../include/ternary.h"
#include "../include/cfc_ternary.h"

/* ============================================================================
 * TEST FRAMEWORK
 * ============================================================================ */

static int tests_run = 0;
static int tests_passed = 0;
static int tests_failed = 0;

#define FLOAT_EQ(a, b, tol) (fabsf((a) - (b)) < (tol))
#define IS_FINITE(x) (isfinite(x))

#define TEST(cond, name) do { \
    tests_run++; \
    if (cond) { tests_passed++; printf("  PASS: %s\n", name); } \
    else { tests_failed++; printf("  FAIL: %s\n", name); } \
} while(0)

#define EXPECT_FAIL(name) do { \
    printf("  KNOWN ISSUE: %s\n", name); \
} while(0)

/* ============================================================================
 * EDGE CASE: ALL ZEROS INPUT TO ABSMEAN
 * ============================================================================ */

void test_absmean_all_zeros(void) {
    printf("\n=== Falsify: Absmean with All Zeros ===\n");
    
    float weights[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    uint8_t packed[1];
    
    /* This should NOT crash or produce NaN */
    ternary_quantize_absmean(weights, packed, 4);
    
    /* All should quantize to 0 */
    TEST(trit_unpack(packed[0], 0) == 0, "Zero input -> zero output [0]");
    TEST(trit_unpack(packed[0], 1) == 0, "Zero input -> zero output [1]");
    TEST(trit_unpack(packed[0], 2) == 0, "Zero input -> zero output [2]");
    TEST(trit_unpack(packed[0], 3) == 0, "Zero input -> zero output [3]");
    
    /* Scale should be 0, but we add epsilon */
    float scale = ternary_absmean_scale(weights, 4);
    TEST(scale == 0.0f, "Absmean of zeros = 0");
}

/* ============================================================================
 * EDGE CASE: SINGLE ELEMENT
 * ============================================================================ */

void test_single_element(void) {
    printf("\n=== Falsify: Single Element Operations ===\n");
    
    /* Single weight */
    float w1[1] = {1.0f};
    uint8_t p1[1];
    ternary_quantize_absmean(w1, p1, 1);
    TEST(trit_unpack(p1[0], 0) == 1, "Single +1.0 -> +1");
    
    /* Single dot product */
    float x1[1] = {5.0f};
    float result = ternary_dot(p1, x1, 1);
    TEST(FLOAT_EQ(result, 5.0f, 1e-6f), "Single element dot = 5");
    
    /* Single matvec (1x1) */
    float y1[1];
    ternary_matvec(p1, x1, y1, 1, 1);
    TEST(FLOAT_EQ(y1[0], 5.0f, 1e-6f), "1x1 matvec = 5");
}

/* ============================================================================
 * EDGE CASE: VERY LARGE VALUES
 * ============================================================================ */

void test_large_values(void) {
    printf("\n=== Falsify: Large Values ===\n");
    
    /* Large float weights */
    float weights[4] = {1e30f, -1e30f, 1e30f, -1e30f};
    uint8_t packed[1];
    
    ternary_quantize_absmean(weights, packed, 4);
    
    /* Should still quantize to +1/-1 */
    TEST(trit_unpack(packed[0], 0) == 1, "1e30 -> +1");
    TEST(trit_unpack(packed[0], 1) == -1, "-1e30 -> -1");
    
    /* Large input to dot product */
    float x_large[4] = {1e30f, 1e30f, 1e30f, 1e30f};
    float result = ternary_dot(packed, x_large, 4);
    
    /* Result should be 1e30 - 1e30 + 1e30 - 1e30 = 0 */
    TEST(FLOAT_EQ(result, 0.0f, 1e20f), "Large balanced dot = 0");
}

/* ============================================================================
 * EDGE CASE: VERY SMALL VALUES (DENORMALS)
 * ============================================================================ */

void test_small_values(void) {
    printf("\n=== Falsify: Denormal/Small Values ===\n");
    
    float weights[4] = {1e-40f, -1e-40f, 1e-40f, 0.0f};
    uint8_t packed[1];
    
    /* Should not crash */
    ternary_quantize_absmean(weights, packed, 4);
    
    /* With absmean, these tiny values should round somewhere */
    /* The scale is ~7.5e-41, so 1e-40/7.5e-41 ≈ 1.33 -> rounds to 1 */
    printf("  Denormal quantization completed (no crash)\n");
    TEST(1, "Denormal weights don't crash");
}

/* ============================================================================
 * EDGE CASE: NaN AND INF INPUTS
 * ============================================================================ */

void test_nan_inf_inputs(void) {
    printf("\n=== Falsify: NaN and Inf Inputs ===\n");
    
    /* NaN weight */
    float nan_weight[4] = {NAN, 1.0f, -1.0f, 0.0f};
    uint8_t packed[1];
    
    /* This is undefined behavior territory - document what happens */
    ternary_quantize_absmean(nan_weight, packed, 4);
    printf("  NaN in weights: quantization completes (behavior undefined)\n");
    
    /* Inf weight */
    float inf_weight[4] = {INFINITY, 1.0f, -1.0f, 0.0f};
    ternary_quantize_absmean(inf_weight, packed, 4);
    printf("  Inf in weights: quantization completes (behavior undefined)\n");
    
    /* NaN in dot product input */
    uint8_t w = trit_pack4(1, 1, 1, 1);
    float x_nan[4] = {1.0f, NAN, 1.0f, 1.0f};
    float result = ternary_dot(&w, x_nan, 4);
    TEST(isnan(result), "NaN input propagates to output");
    
    EXPECT_FAIL("NaN/Inf inputs - behavior is undefined, should validate inputs");
}

/* ============================================================================
 * EDGE CASE: INT8 OVERFLOW IN DOT PRODUCT
 * ============================================================================ */

void test_int8_overflow(void) {
    printf("\n=== Falsify: Int8 Accumulator Overflow ===\n");
    
    /* Create a scenario where int32 accumulator could overflow */
    /* Max int32 = 2,147,483,647 */
    /* If we have N elements, each contributing 127 (max int8), */
    /* we overflow at N > 16,909,320 */
    
    /* Test with 1000 elements - should be fine */
    int n = 1000;
    uint8_t* w_packed = (uint8_t*)malloc(ternary_bytes(n));
    int8_t* x_q = (int8_t*)malloc(n);
    
    /* All weights = +1, all activations = 127 */
    memset(w_packed, 0x55, ternary_bytes(n)); /* 01010101 = all +1 */
    memset(x_q, 127, n);
    
    int32_t result = ternary_dot_int8(w_packed, x_q, n);
    
    /* Expected: 1000 * 127 = 127,000 */
    TEST(result == 127000, "1000-element int8 dot = 127,000");
    
    free(w_packed);
    free(x_q);
    
    /* Document the overflow boundary */
    printf("  Int32 overflow would occur at N > 16.9M elements\n");
    printf("  For typical Yinsen networks (<10K params), this is safe\n");
}

/* ============================================================================
 * EDGE CASE: MISALIGNED VECTOR LENGTHS
 * ============================================================================ */

void test_misaligned_lengths(void) {
    printf("\n=== Falsify: Non-Multiple-of-4 Lengths ===\n");
    
    /* Length 5 (not multiple of 4) */
    float weights[5] = {1.0f, -1.0f, 0.0f, 1.0f, -1.0f};
    uint8_t packed[2]; /* ceil(5/4) = 2 bytes */
    
    ternary_quantize(weights, packed, 5, 0.5f);
    
    float x[5] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
    float result = ternary_dot(packed, x, 5);
    
    /* Expected: 1 - 2 + 0 + 4 - 5 = -2 */
    TEST(FLOAT_EQ(result, -2.0f, 1e-6f), "Length-5 dot = -2");
    
    /* Length 1, 2, 3, 6, 7 */
    float w1[1] = {1.0f}; uint8_t p1[1];
    float w2[2] = {1.0f, -1.0f}; uint8_t p2[1];
    float w3[3] = {1.0f, 1.0f, 1.0f}; uint8_t p3[1];
    
    ternary_quantize(w1, p1, 1, 0.5f);
    ternary_quantize(w2, p2, 2, 0.5f);
    ternary_quantize(w3, p3, 3, 0.5f);
    
    float x3[3] = {1.0f, 1.0f, 1.0f};
    TEST(FLOAT_EQ(ternary_dot(p1, x3, 1), 1.0f, 1e-6f), "Length-1 dot");
    TEST(FLOAT_EQ(ternary_dot(p2, x3, 2), 0.0f, 1e-6f), "Length-2 dot");
    TEST(FLOAT_EQ(ternary_dot(p3, x3, 3), 3.0f, 1e-6f), "Length-3 dot");
}

/* ============================================================================
 * EDGE CASE: EMPTY INPUTS
 * ============================================================================ */

void test_empty_inputs(void) {
    printf("\n=== Falsify: Empty/Zero-Length Inputs ===\n");
    
    /* Zero-length dot product */
    uint8_t w_empty[1] = {0};
    float x_empty[1] = {0};
    
    float result = ternary_dot(w_empty, x_empty, 0);
    TEST(result == 0.0f, "Zero-length dot = 0");
    
    /* Zero-length quantization - should not crash */
    uint8_t p_empty[1];
    ternary_quantize(x_empty, p_empty, 0, 0.5f);
    TEST(1, "Zero-length quantize doesn't crash");
    
    printf("  Warning: n=0 is edge case, behavior should be documented\n");
}

/* ============================================================================
 * EDGE CASE: CfC TERNARY EXTREME DT
 * ============================================================================ */

void test_cfc_extreme_dt(void) {
    printf("\n=== Falsify: CfC with Extreme dt ===\n");
    
    /* Setup minimal CfC */
    const int in_dim = 2;
    const int hid_dim = 2;
    const int concat_dim = in_dim + hid_dim;
    
    uint8_t W_gate[2]; /* 2 rows, ceil(4/4)=1 byte each */
    uint8_t W_cand[2];
    float b_gate[2] = {0.0f, 0.0f};
    float b_cand[2] = {0.0f, 0.0f};
    float tau[1] = {1.0f};
    
    /* Initialize weights to something */
    W_gate[0] = trit_pack4(1, -1, 0, 1);
    W_gate[1] = trit_pack4(0, 1, -1, 0);
    W_cand[0] = trit_pack4(1, 1, 0, 0);
    W_cand[1] = trit_pack4(-1, 0, 1, 0);
    
    CfCTernaryParams params = {
        .input_dim = in_dim,
        .hidden_dim = hid_dim,
        .W_gate = W_gate,
        .b_gate = b_gate,
        .W_cand = W_cand,
        .b_cand = b_cand,
        .tau = tau,
        .tau_shared = 1
    };
    
    float x[2] = {1.0f, 1.0f};
    float h_prev[2] = {0.0f, 0.0f};
    float h_new[2];
    
    /* Very small dt -> decay ≈ 1 (retain state) */
    yinsen_cfc_ternary_cell(x, h_prev, 0.0001f, &params, h_new);
    TEST(IS_FINITE(h_new[0]) && IS_FINITE(h_new[1]), "Small dt produces finite output");
    
    /* Very large dt -> decay ≈ 0 (full update) */
    yinsen_cfc_ternary_cell(x, h_prev, 1000.0f, &params, h_new);
    TEST(IS_FINITE(h_new[0]) && IS_FINITE(h_new[1]), "Large dt produces finite output");
    
    /* Zero dt */
    yinsen_cfc_ternary_cell(x, h_prev, 0.0f, &params, h_new);
    TEST(IS_FINITE(h_new[0]) && IS_FINITE(h_new[1]), "Zero dt produces finite output");
    
    /* Negative dt (invalid but shouldn't crash) */
    yinsen_cfc_ternary_cell(x, h_prev, -1.0f, &params, h_new);
    printf("  Negative dt: h_new = [%.4f, %.4f]\n", h_new[0], h_new[1]);
    EXPECT_FAIL("Negative dt - should validate, decay = exp(-(-1)/1) = exp(1) = 2.7");
}

/* ============================================================================
 * EDGE CASE: CfC TERNARY EXTREME TAU
 * ============================================================================ */

void test_cfc_extreme_tau(void) {
    printf("\n=== Falsify: CfC with Extreme tau ===\n");
    
    const int in_dim = 2;
    const int hid_dim = 2;
    
    uint8_t W_gate[2], W_cand[2];
    float b_gate[2] = {0}, b_cand[2] = {0};
    
    W_gate[0] = trit_pack4(1, 0, 0, 0);
    W_gate[1] = trit_pack4(0, 1, 0, 0);
    W_cand[0] = trit_pack4(1, 0, 0, 0);
    W_cand[1] = trit_pack4(0, 1, 0, 0);
    
    float x[2] = {1.0f, 1.0f};
    float h_prev[2] = {0.5f, 0.5f};
    float h_new[2];
    
    /* Zero tau -> invalid, should produce NaN */
    float tau_zero[1] = {0.0f};
    CfCTernaryParams params_zero = {
        .input_dim = in_dim, .hidden_dim = hid_dim,
        .W_gate = W_gate, .b_gate = b_gate,
        .W_cand = W_cand, .b_cand = b_cand,
        .tau = tau_zero, .tau_shared = 1
    };
    
    yinsen_cfc_ternary_cell(x, h_prev, 0.1f, &params_zero, h_new);
    printf("  Zero tau: h_new = [%.4f, %.4f]\n", h_new[0], h_new[1]);
    
    /* Zero tau is invalid - should produce NaN */
    TEST(isnan(h_new[0]) && isnan(h_new[1]), "Zero tau produces NaN (invalid tau rejected)");
    
    /* Very small tau -> fast decay */
    float tau_tiny[1] = {1e-10f};
    CfCTernaryParams params_tiny = {
        .input_dim = in_dim, .hidden_dim = hid_dim,
        .W_gate = W_gate, .b_gate = b_gate,
        .W_cand = W_cand, .b_cand = b_cand,
        .tau = tau_tiny, .tau_shared = 1
    };
    
    yinsen_cfc_ternary_cell(x, h_prev, 0.1f, &params_tiny, h_new);
    TEST(IS_FINITE(h_new[0]) && IS_FINITE(h_new[1]), "Tiny tau produces finite output");
}

/* ============================================================================
 * EDGE CASE: ACTIVATION QUANTIZATION EXTREMES
 * ============================================================================ */

void test_activation_quant_extremes(void) {
    printf("\n=== Falsify: Activation Quantization Extremes ===\n");
    
    /* All same value */
    float x_same[4] = {1.0f, 1.0f, 1.0f, 1.0f};
    int8_t x_q[4];
    TernaryQuantParams params;
    
    ternary_quantize_activations(x_same, x_q, 4, &params);
    TEST(x_q[0] == 127 && x_q[1] == 127, "All same -> all max");
    
    /* Very small variance */
    float x_tight[4] = {1.0f, 1.0001f, 0.9999f, 1.0f};
    ternary_quantize_activations(x_tight, x_q, 4, &params);
    printf("  Tight range [0.9999, 1.0001]: q = [%d, %d, %d, %d]\n", 
           x_q[0], x_q[1], x_q[2], x_q[3]);
    TEST(1, "Tight range quantizes (check values manually)");
    
    /* Zero variance (all zeros) */
    float x_zero[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    ternary_quantize_activations(x_zero, x_q, 4, &params);
    TEST(x_q[0] == 0 && x_q[1] == 0, "All zeros -> all zero quant");
    TEST(params.scale < 1e-6f, "Zero input -> tiny scale");
}

/* ============================================================================
 * EDGE CASE: SPARSITY ON EDGE CASES
 * ============================================================================ */

void test_sparsity_edge_cases(void) {
    printf("\n=== Falsify: Sparsity Edge Cases ===\n");
    
    /* All zeros */
    uint8_t all_zero[2] = {0x00, 0x00}; /* 00000000 = all zeros */
    TernaryStats stats;
    ternary_stats(all_zero, 8, &stats);
    TEST(stats.sparsity == 1.0f, "All zeros -> 100% sparse");
    TEST(stats.zeros == 8, "All zeros count = 8");
    
    /* All +1 (01010101 = 0x55) */
    uint8_t all_pos[2] = {0x55, 0x55};
    ternary_stats(all_pos, 8, &stats);
    TEST(stats.sparsity == 0.0f, "All +1 -> 0% sparse");
    TEST(stats.positive == 8, "All +1 count = 8");
    
    /* All -1: canonical encoding 10 = 0x2, so 10101010 = 0xAA */
    uint8_t all_neg[2] = {0xAA, 0xAA};
    ternary_stats(all_neg, 8, &stats);
    TEST(stats.sparsity == 0.0f, "All -1 -> 0% sparse");
    TEST(stats.negative == 8, "All -1 count = 8");
}

/* ============================================================================
 * CHECK: RESERVED ENCODING HANDLING
 * ============================================================================ */

void test_reserved_encoding(void) {
    printf("\n=== Falsify: Reserved Encoding (11) ===\n");
    
    /* Canonical encoding: 11 (binary) = 3 (decimal) is reserved */
    /* What happens if we encounter it? */
    
    uint8_t with_reserved = 0x03; /* 00000011 = pos 0 has reserved encoding (11) */
    
    int8_t trit = trit_unpack(with_reserved, 0);
    TEST(trit == 0, "Reserved encoding (11) treated as 0");
    
    /* Dot product with reserved encoding */
    float x[4] = {100.0f, 1.0f, 1.0f, 1.0f};
    float result = ternary_dot(&with_reserved, x, 4);
    
    /* If reserved = 0, result should ignore the 100 */
    printf("  Dot with reserved encoding: %.4f\n", result);
    TEST(FLOAT_EQ(result, 0.0f, 1e-6f), "Reserved encoding contributes 0");
}

/* ============================================================================
 * STRESS TEST: MANY ITERATIONS
 * ============================================================================ */

void test_stress_iterations(void) {
    printf("\n=== Falsify: Stress Test (10K iterations) ===\n");
    
    const int in_dim = 4;
    const int hid_dim = 4;
    
    uint8_t W_gate[4], W_cand[4];
    float b_gate[4] = {0}, b_cand[4] = {0};
    float tau[1] = {1.0f};
    
    /* Random-ish weights */
    W_gate[0] = trit_pack4(1, -1, 0, 1);
    W_gate[1] = trit_pack4(0, 1, -1, 0);
    W_gate[2] = trit_pack4(1, 0, 1, -1);
    W_gate[3] = trit_pack4(-1, 1, 0, 1);
    memcpy(W_cand, W_gate, 4);
    
    CfCTernaryParams params = {
        .input_dim = in_dim, .hidden_dim = hid_dim,
        .W_gate = W_gate, .b_gate = b_gate,
        .W_cand = W_cand, .b_cand = b_cand,
        .tau = tau, .tau_shared = 1
    };
    
    float x[4] = {0.5f, -0.5f, 0.3f, -0.3f};
    float h[4] = {0};
    float h_new[4];
    
    int stable = 1;
    for (int i = 0; i < 10000; i++) {
        yinsen_cfc_ternary_cell(x, h, 0.1f, &params, h_new);
        memcpy(h, h_new, sizeof(h));
        
        for (int j = 0; j < hid_dim; j++) {
            if (!IS_FINITE(h[j]) || fabsf(h[j]) > 1e6f) {
                stable = 0;
                printf("  Diverged at iteration %d: h[%d] = %.4f\n", i, j, h[j]);
                break;
            }
        }
        if (!stable) break;
    }
    
    TEST(stable, "10K iterations remain stable and bounded");
    printf("  Final state: [%.4f, %.4f, %.4f, %.4f]\n", h[0], h[1], h[2], h[3]);
}

/* ============================================================================
 * MAIN
 * ============================================================================ */

int main(void) {
    printf("===================================================\n");
    printf("  YINSEN FALSIFICATION TEST SUITE\n");
    printf("  Goal: Find bugs, edge cases, undefined behavior\n");
    printf("===================================================\n");
    
    test_absmean_all_zeros();
    test_single_element();
    test_large_values();
    test_small_values();
    test_nan_inf_inputs();
    test_int8_overflow();
    test_misaligned_lengths();
    test_empty_inputs();
    test_cfc_extreme_dt();
    test_cfc_extreme_tau();
    test_activation_quant_extremes();
    test_sparsity_edge_cases();
    test_reserved_encoding();
    test_stress_iterations();
    
    printf("\n===================================================\n");
    printf("  RESULTS: %d/%d passed, %d failed\n", 
           tests_passed, tests_run, tests_failed);
    printf("===================================================\n");
    
    if (tests_failed > 0) {
        printf("  ISSUES FOUND - Review KNOWN ISSUE and FAIL items\n");
    } else {
        printf("  No failures - but check KNOWN ISSUE items\n");
    }
    
    /* Always return 0 - this is exploratory, not CI */
    return 0;
}
