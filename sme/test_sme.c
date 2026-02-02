/*
 * test_sme.c - Verification tests for SME 16x16 ternary kernels
 *
 * Test strategy:
 * 1. Verify reference implementation against known values
 * 2. Test weight packing/unpacking roundtrip
 * 3. Random sample testing (100K samples for statistical confidence)
 * 4. Boundary tests (all zeros, all ones, all negatives, mixed)
 * 5. Property-based tests (linearity, distributivity)
 *
 * Copyright 2026 Trix Research
 */

#include "ternary_sme.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

#define EPSILON 1e-5f

static int tests_run = 0;
static int tests_passed = 0;

#define TEST(name) \
    do { \
        printf("  %-50s ", name); \
        tests_run++; \
    } while(0)

#define PASS() \
    do { \
        printf("PASS\n"); \
        tests_passed++; \
    } while(0)

#define FAIL(msg, ...) \
    do { \
        printf("FAIL: " msg "\n", ##__VA_ARGS__); \
    } while(0)

#define ASSERT(cond, msg, ...) \
    do { \
        if (!(cond)) { \
            FAIL(msg, ##__VA_ARGS__); \
            return; \
        } \
    } while(0)

#define ASSERT_FLOAT_EQ(a, b, msg, ...) \
    ASSERT(fabsf((a) - (b)) < EPSILON, msg " (expected %f, got %f)", ##__VA_ARGS__, (b), (a))

/* ============================================================================
 * Test Utilities
 * ============================================================================ */

static uint32_t xorshift32(uint32_t* state) {
    uint32_t x = *state;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    *state = x;
    return x;
}

static float random_float(uint32_t* state) {
    return (float)(xorshift32(state) & 0xFFFF) / 32768.0f - 1.0f;
}

static uint32_t random_weights(uint32_t* state) {
    // Generate 16 random 2-bit trits (only use 0, 1, 2)
    uint32_t w = 0;
    for (int i = 0; i < 16; i++) {
        uint32_t trit = xorshift32(state) % 3;  // 0, 1, or 2
        w |= (trit << (i * 2));
    }
    return w;
}

/* ============================================================================
 * Reference Verification Tests
 * ============================================================================ */

static void test_dot16_all_zeros(void) {
    TEST("dot16: all zero weights");
    
    float activations[16] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
    uint32_t weights = 0;  // All zeros
    
    float result = sme_dot16_ref(activations, weights);
    ASSERT_FLOAT_EQ(result, 0.0f, "all zeros should give 0");
    PASS();
}

static void test_dot16_all_ones(void) {
    TEST("dot16: all +1 weights");
    
    float activations[16] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
    // All +1: each trit is 0b01
    uint32_t weights = 0x55555555;  // 0b01 repeated 16 times
    
    float result = sme_dot16_ref(activations, weights);
    float expected = 1+2+3+4+5+6+7+8+9+10+11+12+13+14+15+16;  // = 136
    ASSERT_FLOAT_EQ(result, expected, "all +1");
    PASS();
}

static void test_dot16_all_negatives(void) {
    TEST("dot16: all -1 weights");
    
    float activations[16] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
    // All -1: each trit is 0b10
    uint32_t weights = 0xAAAAAAAA;  // 0b10 repeated 16 times
    
    float result = sme_dot16_ref(activations, weights);
    float expected = -(1+2+3+4+5+6+7+8+9+10+11+12+13+14+15+16);  // = -136
    ASSERT_FLOAT_EQ(result, expected, "all -1");
    PASS();
}

static void test_dot16_alternating(void) {
    TEST("dot16: alternating +1/-1 weights");
    
    float activations[16];
    for (int i = 0; i < 16; i++) activations[i] = 1.0f;
    
    // Alternating: +1, -1, +1, -1, ...
    // In bits: 01, 10, 01, 10, ... = 0x99999999
    uint32_t weights = 0x99999999;
    
    float result = sme_dot16_ref(activations, weights);
    // 8 positives - 8 negatives = 0
    ASSERT_FLOAT_EQ(result, 0.0f, "alternating should cancel");
    PASS();
}

static void test_dot16_single_positive(void) {
    TEST("dot16: single +1 weight at position 0");
    
    float activations[16] = {42.0f, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    uint32_t weights = 0x1;  // +1 at position 0 only
    
    float result = sme_dot16_ref(activations, weights);
    ASSERT_FLOAT_EQ(result, 42.0f, "single +1 at pos 0");
    PASS();
}

static void test_dot16_single_negative(void) {
    TEST("dot16: single -1 weight at position 15");
    
    float activations[16] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 42.0f};
    uint32_t weights = 0x2 << 30;  // -1 at position 15 (bits 30-31)
    
    float result = sme_dot16_ref(activations, weights);
    ASSERT_FLOAT_EQ(result, -42.0f, "single -1 at pos 15");
    PASS();
}

/* ============================================================================
 * Weight Packing Tests
 * ============================================================================ */

static void test_weight_buffer_size(void) {
    TEST("weight_buffer_size: 16x16");
    size_t size = sme_weight_buffer_size(16, 16);
    // 1 tile, 16 uint32_t
    ASSERT(size == 16 * sizeof(uint32_t), "16x16 should be 64 bytes, got %zu", size);
    PASS();
}

static void test_weight_buffer_size_32x32(void) {
    TEST("weight_buffer_size: 32x32");
    size_t size = sme_weight_buffer_size(32, 32);
    // 4 tiles (2x2), each 16 uint32_t
    ASSERT(size == 4 * 16 * sizeof(uint32_t), "32x32 should be 256 bytes, got %zu", size);
    PASS();
}

static void test_weight_buffer_size_non_aligned(void) {
    TEST("weight_buffer_size: 17x17 (non-aligned)");
    size_t size = sme_weight_buffer_size(17, 17);
    // Rounds up to 32x32 = 4 tiles
    ASSERT(size == 4 * 16 * sizeof(uint32_t), "17x17 should round to 32x32, got %zu", size);
    PASS();
}

static void test_pack_weights_identity(void) {
    TEST("pack_weights: 16x16 identity-like");
    
    // Create source weights where w[i,i] = +1, others = 0
    uint8_t src[64] = {0};  // 16*16*2 bits = 512 bits = 64 bytes
    
    // Set diagonal elements to +1 (0b01)
    for (int i = 0; i < 16; i++) {
        size_t bit_idx = (i * 16 + i) * 2;  // row i, col i
        size_t byte_idx = bit_idx / 8;
        size_t bit_offset = bit_idx % 8;
        src[byte_idx] |= (0x1 << bit_offset);
    }
    
    uint32_t dst[16];
    sme_pack_weights(dst, src, 16, 16);
    
    // Verify: each dst[i] should have +1 only at position i
    for (int i = 0; i < 16; i++) {
        uint32_t expected = 0x1 << (i * 2);  // +1 at position i
        ASSERT(dst[i] == expected, "row %d: expected 0x%08x, got 0x%08x", i, expected, dst[i]);
    }
    PASS();
}

/* ============================================================================
 * Matrix-Vector Tests
 * ============================================================================ */

static void test_matvec_16x16_identity(void) {
    TEST("matvec: 16x16 identity-like");
    
    // Create weights where w[i,i] = +1
    uint32_t weights[16];
    for (int i = 0; i < 16; i++) {
        weights[i] = 0x1 << (i * 2);  // +1 at position i
    }
    
    float input[16] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
    float output[16];
    
    sme_matvec_ref(output, weights, input, 16, 16);
    
    // Output should equal input for identity
    for (int i = 0; i < 16; i++) {
        ASSERT_FLOAT_EQ(output[i], input[i], "output[%d]", i);
    }
    PASS();
}

static void test_matvec_16x16_all_ones(void) {
    TEST("matvec: 16x16 all +1 weights");
    
    uint32_t weights[16];
    for (int i = 0; i < 16; i++) {
        weights[i] = 0x55555555;  // All +1
    }
    
    float input[16];
    for (int i = 0; i < 16; i++) input[i] = 1.0f;
    float output[16];
    
    sme_matvec_ref(output, weights, input, 16, 16);
    
    // Each output should be sum of all inputs = 16
    for (int i = 0; i < 16; i++) {
        ASSERT_FLOAT_EQ(output[i], 16.0f, "output[%d]", i);
    }
    PASS();
}

/* ============================================================================
 * 32x32 Matrix-Vector Tests
 * ============================================================================ */

static void test_matvec_32x32_identity(void) {
    TEST("matvec: 32x32 identity-like");

    /* 4 tiles: T00, T01, T10, T11 — each 16 uint32_t */
    uint32_t weights[64];
    memset(weights, 0, sizeof(weights));

    /* T00: rows 0-15, cols 0-15 — diagonal +1 */
    for (int i = 0; i < 16; i++) {
        weights[i] = 0x1 << (i * 2);
    }
    /* T01: rows 0-15, cols 16-31 — all zero (already zeroed) */
    /* T10: rows 16-31, cols 0-15 — all zero (already zeroed) */
    /* T11: rows 16-31, cols 16-31 — diagonal +1 */
    for (int i = 0; i < 16; i++) {
        weights[48 + i] = 0x1 << (i * 2);
    }

    float input[32];
    for (int i = 0; i < 32; i++) input[i] = (float)(i + 1);
    float output[32];

    sme_matvec(output, weights, input, 32, 32);

    for (int i = 0; i < 32; i++) {
        ASSERT_FLOAT_EQ(output[i], input[i], "identity output[%d]", i);
    }
    PASS();
}

static void test_matvec_32x32_all_ones(void) {
    TEST("matvec: 32x32 all +1 weights");

    uint32_t weights[64];
    for (int i = 0; i < 64; i++) {
        weights[i] = 0x55555555;  /* All +1 */
    }

    float input[32];
    for (int i = 0; i < 32; i++) input[i] = 1.0f;
    float output[32];

    sme_matvec(output, weights, input, 32, 32);

    /* Each output should be sum of all 32 inputs = 32 */
    for (int i = 0; i < 32; i++) {
        ASSERT_FLOAT_EQ(output[i], 32.0f, "all-ones output[%d]", i);
    }
    PASS();
}

static void test_matvec_32x32_block_diagonal(void) {
    TEST("matvec: 32x32 block diagonal (+1 top, -1 bottom)");

    uint32_t weights[64];
    memset(weights, 0, sizeof(weights));

    /* T00: all +1 (rows 0-15 sum cols 0-15) */
    for (int i = 0; i < 16; i++) weights[i] = 0x55555555;
    /* T01: all zero */
    /* T10: all zero */
    /* T11: all -1 (rows 16-31 negate-sum cols 16-31) */
    for (int i = 0; i < 16; i++) weights[48 + i] = 0xAAAAAAAA;

    float input[32];
    for (int i = 0; i < 32; i++) input[i] = 1.0f;
    float output[32];

    sme_matvec(output, weights, input, 32, 32);

    /* Rows 0-15: sum of input[0:15] = 16 */
    for (int i = 0; i < 16; i++) {
        ASSERT_FLOAT_EQ(output[i], 16.0f, "block diag output[%d]", i);
    }
    /* Rows 16-31: -sum of input[16:31] = -16 */
    for (int i = 16; i < 32; i++) {
        ASSERT_FLOAT_EQ(output[i], -16.0f, "block diag output[%d]", i);
    }
    PASS();
}

static void test_sme_matches_ref_matvec_32x32(void) {
    TEST("SME matvec 32x32 matches reference");

    if (!sme_available()) {
        printf("SKIP (no SME)\n");
        tests_passed++;
        return;
    }

    uint32_t rng = 0xFEEDFACE;

    uint32_t weights[64];
    for (int i = 0; i < 64; i++) {
        weights[i] = random_weights(&rng);
    }

    float input[32];
    for (int i = 0; i < 32; i++) {
        input[i] = random_float(&rng);
    }

    float output_ref[32], output_sme[32];
    memset(output_ref, 0, sizeof(output_ref));

    sme_matvec_ref(output_ref, weights, input, 32, 32);
    sme_matvec(output_sme, weights, input, 32, 32);

    for (int i = 0; i < 32; i++) {
        ASSERT_FLOAT_EQ(output_sme[i], output_ref[i], "32x32 output[%d]", i);
    }
    PASS();
}

static void test_random_matvec_32x32(void) {
    TEST("random matvec 32x32: 10K samples");

    uint32_t rng = 0xDEADC0DE;

    for (int sample = 0; sample < 10000; sample++) {
        float input[32];
        for (int i = 0; i < 32; i++) {
            input[i] = random_float(&rng);
        }

        uint32_t weights[64];
        for (int i = 0; i < 64; i++) {
            weights[i] = random_weights(&rng);
        }

        float output_ref[32], output_sme[32];
        memset(output_ref, 0, sizeof(output_ref));
        sme_matvec_ref(output_ref, weights, input, 32, 32);
        sme_matvec(output_sme, weights, input, 32, 32);

        for (int i = 0; i < 32; i++) {
            if (fabsf(output_ref[i] - output_sme[i]) >= EPSILON) {
                FAIL("sample %d, output[%d]: ref=%f, sme=%f",
                     sample, i, output_ref[i], output_sme[i]);
                return;
            }
        }
    }
    PASS();
}

/* ============================================================================
 * SME vs Reference Tests
 * ============================================================================ */

static void test_sme_matches_ref_dot16(void) {
    TEST("SME dot16 matches reference");
    
    if (!sme_available()) {
        printf("SKIP (no SME)\n");
        tests_passed++;  // Count as pass on non-SME hardware
        return;
    }
    
    float activations[16] = {1, -2, 3, -4, 5, -6, 7, -8, 9, -10, 11, -12, 13, -14, 15, -16};
    uint32_t weights = 0x12345678;  // Arbitrary pattern
    
    float ref_result = sme_dot16_ref(activations, weights);
    float sme_result = sme_dot16(activations, weights);
    
    ASSERT_FLOAT_EQ(sme_result, ref_result, "SME should match reference");
    PASS();
}

static void test_sme_matches_ref_matvec(void) {
    TEST("SME matvec matches reference");
    
    if (!sme_available()) {
        printf("SKIP (no SME)\n");
        tests_passed++;
        return;
    }
    
    uint32_t weights[16];
    for (int i = 0; i < 16; i++) {
        weights[i] = 0x12345678 + i;  // Arbitrary pattern
    }
    
    float input[16] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
    float output_ref[16], output_sme[16];
    
    memset(output_ref, 0, sizeof(output_ref));
    sme_matvec_ref(output_ref, weights, input, 16, 16);
    sme_matvec(output_sme, weights, input, 16, 16);
    
    for (int i = 0; i < 16; i++) {
        ASSERT_FLOAT_EQ(output_sme[i], output_ref[i], "output[%d]", i);
    }
    PASS();
}

/* ============================================================================
 * Random Sample Tests
 * ============================================================================ */

static void test_random_dot16(void) {
    TEST("random dot16: 100K samples");
    
    uint32_t rng = 0xDEADBEEF;
    
    for (int sample = 0; sample < 100000; sample++) {
        float activations[16];
        for (int i = 0; i < 16; i++) {
            activations[i] = random_float(&rng);
        }
        
        uint32_t weights = random_weights(&rng);
        
        float ref = sme_dot16_ref(activations, weights);
        float sme = sme_dot16(activations, weights);
        
        if (fabsf(ref - sme) >= EPSILON) {
            FAIL("sample %d: ref=%f, sme=%f", sample, ref, sme);
            return;
        }
    }
    PASS();
}

static void test_random_matvec(void) {
    TEST("random matvec 16x16: 10K samples");
    
    uint32_t rng = 0xCAFEBABE;
    
    for (int sample = 0; sample < 10000; sample++) {
        float input[16];
        for (int i = 0; i < 16; i++) {
            input[i] = random_float(&rng);
        }
        
        uint32_t weights[16];
        for (int i = 0; i < 16; i++) {
            weights[i] = random_weights(&rng);
        }
        
        float output_ref[16], output_sme[16];
        memset(output_ref, 0, sizeof(output_ref));
        sme_matvec_ref(output_ref, weights, input, 16, 16);
        sme_matvec(output_sme, weights, input, 16, 16);
        
        for (int i = 0; i < 16; i++) {
            if (fabsf(output_ref[i] - output_sme[i]) >= EPSILON) {
                FAIL("sample %d, output[%d]: ref=%f, sme=%f", 
                     sample, i, output_ref[i], output_sme[i]);
                return;
            }
        }
    }
    PASS();
}

/* ============================================================================
 * Property-Based Tests
 * ============================================================================ */

static void test_linearity_scaling(void) {
    TEST("linearity: f(kx, w) = k * f(x, w)");
    
    uint32_t rng = 0xBADC0DE;
    
    for (int sample = 0; sample < 1000; sample++) {
        float activations[16];
        for (int i = 0; i < 16; i++) {
            activations[i] = random_float(&rng);
        }
        
        uint32_t weights = random_weights(&rng);
        float k = random_float(&rng) * 10.0f;  // Scale factor
        
        // f(x, w)
        float f_x = sme_dot16(activations, weights);
        
        // f(kx, w)
        float scaled[16];
        for (int i = 0; i < 16; i++) scaled[i] = k * activations[i];
        float f_kx = sme_dot16(scaled, weights);
        
        // Should be equal: f(kx, w) = k * f(x, w)
        if (fabsf(f_kx - k * f_x) >= EPSILON * (1 + fabsf(k))) {
            FAIL("sample %d: f(kx)=%f, k*f(x)=%f", sample, f_kx, k * f_x);
            return;
        }
    }
    PASS();
}

static void test_linearity_additivity(void) {
    TEST("linearity: f(x+y, w) = f(x, w) + f(y, w)");
    
    uint32_t rng = 0x12345678;
    
    for (int sample = 0; sample < 1000; sample++) {
        float x[16], y[16], xy[16];
        for (int i = 0; i < 16; i++) {
            x[i] = random_float(&rng);
            y[i] = random_float(&rng);
            xy[i] = x[i] + y[i];
        }
        
        uint32_t weights = random_weights(&rng);
        
        float f_x = sme_dot16(x, weights);
        float f_y = sme_dot16(y, weights);
        float f_xy = sme_dot16(xy, weights);
        
        if (fabsf(f_xy - (f_x + f_y)) >= EPSILON * 3) {
            FAIL("sample %d: f(x+y)=%f, f(x)+f(y)=%f", sample, f_xy, f_x + f_y);
            return;
        }
    }
    PASS();
}

/* ============================================================================
 * Main
 * ============================================================================ */

int main(int argc, char** argv) {
    printf("\n");
    printf("=================================================================\n");
    printf("  Yinsen SME 16x16 Ternary Kernel Tests\n");
    printf("=================================================================\n");
    printf("\n");
    
    printf("Hardware detection:\n");
    printf("  SME available: %s\n", sme_available() ? "YES" : "NO");
    printf("\n");
    
    printf("Reference implementation tests:\n");
    test_dot16_all_zeros();
    test_dot16_all_ones();
    test_dot16_all_negatives();
    test_dot16_alternating();
    test_dot16_single_positive();
    test_dot16_single_negative();
    printf("\n");
    
    printf("Weight packing tests:\n");
    test_weight_buffer_size();
    test_weight_buffer_size_32x32();
    test_weight_buffer_size_non_aligned();
    test_pack_weights_identity();
    printf("\n");
    
    printf("Matrix-vector tests:\n");
    test_matvec_16x16_identity();
    test_matvec_16x16_all_ones();
    printf("\n");
    
    printf("32x32 matrix-vector tests:\n");
    test_matvec_32x32_identity();
    test_matvec_32x32_all_ones();
    test_matvec_32x32_block_diagonal();
    test_sme_matches_ref_matvec_32x32();
    test_random_matvec_32x32();
    printf("\n");

    printf("SME vs reference tests:\n");
    test_sme_matches_ref_dot16();
    test_sme_matches_ref_matvec();
    printf("\n");
    
    printf("Random sample tests:\n");
    test_random_dot16();
    test_random_matvec();
    printf("\n");
    
    printf("Property-based tests:\n");
    test_linearity_scaling();
    test_linearity_additivity();
    printf("\n");
    
    printf("=================================================================\n");
    printf("  Results: %d/%d tests passed\n", tests_passed, tests_run);
    printf("=================================================================\n");
    printf("\n");
    
    return (tests_passed == tests_run) ? 0 : 1;
}
