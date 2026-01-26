/*
 * YINSEN Ternary Test Suite
 *
 * Exhaustive verification of ternary weight operations.
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include "../include/ternary.h"

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
 * TRIT ENCODING TESTS
 * ============================================================================ */

void test_trit_encode_decode(void) {
    printf("\n=== Trit Encoding/Decoding ===\n");

    /* Test encode */
    TEST(trit_encode(0) == TRIT_ZERO, "encode(0) = TRIT_ZERO");
    TEST(trit_encode(1) == TRIT_POS, "encode(1) = TRIT_POS");
    TEST(trit_encode(-1) == TRIT_NEG, "encode(-1) = TRIT_NEG");

    /* Test unpack at each position */
    uint8_t packed = trit_pack4(1, -1, 0, 1);

    TEST(trit_unpack(packed, 0) == 1, "unpack pos 0 = +1");
    TEST(trit_unpack(packed, 1) == -1, "unpack pos 1 = -1");
    TEST(trit_unpack(packed, 2) == 0, "unpack pos 2 = 0");
    TEST(trit_unpack(packed, 3) == 1, "unpack pos 3 = +1");

    /* Test all 81 combinations of 4 trits (3^4 = 81) */
    int all_correct = 1;
    int8_t vals[] = {-1, 0, 1};

    for (int a = 0; a < 3; a++) {
        for (int b = 0; b < 3; b++) {
            for (int c = 0; c < 3; c++) {
                for (int d = 0; d < 3; d++) {
                    uint8_t p = trit_pack4(vals[a], vals[b], vals[c], vals[d]);
                    if (trit_unpack(p, 0) != vals[a] ||
                        trit_unpack(p, 1) != vals[b] ||
                        trit_unpack(p, 2) != vals[c] ||
                        trit_unpack(p, 3) != vals[d]) {
                        all_correct = 0;
                    }
                }
            }
        }
    }
    TEST(all_correct, "All 81 pack/unpack combinations correct");
}

/* ============================================================================
 * TERNARY DOT PRODUCT TESTS
 * ============================================================================ */

void test_ternary_dot(void) {
    printf("\n=== Ternary Dot Product ===\n");

    /* Simple case: [+1, -1, 0, +1] . [1, 2, 3, 4] = 1 - 2 + 0 + 4 = 3 */
    uint8_t w1 = trit_pack4(1, -1, 0, 1);
    float x1[4] = {1.0f, 2.0f, 3.0f, 4.0f};

    float result1 = ternary_dot(&w1, x1, 4);
    TEST(FLOAT_EQ(result1, 3.0f, 1e-6f), "dot([+1,-1,0,+1], [1,2,3,4]) = 3");

    /* All zeros */
    uint8_t w2 = trit_pack4(0, 0, 0, 0);
    float result2 = ternary_dot(&w2, x1, 4);
    TEST(FLOAT_EQ(result2, 0.0f, 1e-6f), "dot([0,0,0,0], x) = 0");

    /* All positive */
    uint8_t w3 = trit_pack4(1, 1, 1, 1);
    float result3 = ternary_dot(&w3, x1, 4);
    TEST(FLOAT_EQ(result3, 10.0f, 1e-6f), "dot([+1,+1,+1,+1], [1,2,3,4]) = 10");

    /* All negative */
    uint8_t w4 = trit_pack4(-1, -1, -1, -1);
    float result4 = ternary_dot(&w4, x1, 4);
    TEST(FLOAT_EQ(result4, -10.0f, 1e-6f), "dot([-1,-1,-1,-1], [1,2,3,4]) = -10");

    /* Longer vector (8 elements, 2 bytes) */
    uint8_t w5[2];
    w5[0] = trit_pack4(1, 1, 1, 1);
    w5[1] = trit_pack4(-1, -1, -1, -1);
    float x5[8] = {1, 1, 1, 1, 1, 1, 1, 1};
    float result5 = ternary_dot(w5, x5, 8);
    TEST(FLOAT_EQ(result5, 0.0f, 1e-6f), "8-element balanced = 0");
}

/* ============================================================================
 * TERNARY MATVEC TESTS
 * ============================================================================ */

void test_ternary_matvec(void) {
    printf("\n=== Ternary Matrix-Vector Multiply ===\n");

    /* 2x4 matrix, 4-element input */
    /* Row 0: [+1, -1, 0, +1] */
    /* Row 1: [0, +1, +1, 0]  */
    uint8_t W[2];
    W[0] = trit_pack4(1, -1, 0, 1);
    W[1] = trit_pack4(0, 1, 1, 0);

    float x[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    float y[2];

    ternary_matvec(W, x, y, 2, 4);

    /* y[0] = 1 - 2 + 0 + 4 = 3 */
    /* y[1] = 0 + 2 + 3 + 0 = 5 */
    TEST(FLOAT_EQ(y[0], 3.0f, 1e-6f), "matvec row 0 = 3");
    TEST(FLOAT_EQ(y[1], 5.0f, 1e-6f), "matvec row 1 = 5");

    /* With bias */
    float bias[2] = {0.5f, -1.0f};
    float y_bias[2];
    ternary_matvec_bias(W, x, bias, y_bias, 2, 4);

    TEST(FLOAT_EQ(y_bias[0], 3.5f, 1e-6f), "matvec+bias row 0 = 3.5");
    TEST(FLOAT_EQ(y_bias[1], 4.0f, 1e-6f), "matvec+bias row 1 = 4.0");
}

/* ============================================================================
 * QUANTIZATION TESTS
 * ============================================================================ */

void test_quantize(void) {
    printf("\n=== Ternary Quantization ===\n");

    float weights[8] = {0.6f, -0.7f, 0.1f, -0.1f, 0.9f, -0.9f, 0.0f, 0.5f};
    uint8_t packed[2];

    ternary_quantize(weights, packed, 8, 0.5f);

    /* Expected with threshold 0.5:
     * 0.6 > 0.5  -> +1
     * -0.7 < -0.5 -> -1
     * 0.1        -> 0
     * -0.1       -> 0
     * 0.9 > 0.5  -> +1
     * -0.9 < -0.5 -> -1
     * 0.0        -> 0
     * 0.5        -> 0 (not strictly greater)
     */
    TEST(trit_unpack(packed[0], 0) == 1, "quantize 0.6 -> +1");
    TEST(trit_unpack(packed[0], 1) == -1, "quantize -0.7 -> -1");
    TEST(trit_unpack(packed[0], 2) == 0, "quantize 0.1 -> 0");
    TEST(trit_unpack(packed[0], 3) == 0, "quantize -0.1 -> 0");
    TEST(trit_unpack(packed[1], 0) == 1, "quantize 0.9 -> +1");
    TEST(trit_unpack(packed[1], 1) == -1, "quantize -0.9 -> -1");
    TEST(trit_unpack(packed[1], 2) == 0, "quantize 0.0 -> 0");
    TEST(trit_unpack(packed[1], 3) == 0, "quantize 0.5 -> 0");
}

/* ============================================================================
 * ROUNDTRIP TEST
 * ============================================================================ */

void test_roundtrip(void) {
    printf("\n=== Pack/Unpack Roundtrip ===\n");

    /* Pack then unpack should give original trits */
    int8_t original[12] = {1, -1, 0, 1, -1, -1, 0, 0, 1, 1, -1, 0};
    uint8_t packed[3];

    /* Manual pack */
    packed[0] = trit_pack4(original[0], original[1], original[2], original[3]);
    packed[1] = trit_pack4(original[4], original[5], original[6], original[7]);
    packed[2] = trit_pack4(original[8], original[9], original[10], original[11]);

    /* Unpack to float */
    float unpacked[12];
    ternary_unpack_to_float(packed, unpacked, 12);

    int all_match = 1;
    for (int i = 0; i < 12; i++) {
        if ((int)unpacked[i] != original[i]) {
            all_match = 0;
        }
    }
    TEST(all_match, "12-trit roundtrip preserves values");
}

/* ============================================================================
 * MEMORY COMPRESSION TEST
 * ============================================================================ */

void test_compression(void) {
    printf("\n=== Memory Compression ===\n");

    size_t tern_bytes, float_bytes;
    float ratio;

    ternary_memory_stats(1024, &tern_bytes, &float_bytes, &ratio);

    TEST(tern_bytes == 256, "1024 trits = 256 bytes");
    TEST(float_bytes == 4096, "1024 floats = 4096 bytes");
    TEST(FLOAT_EQ(ratio, 16.0f, 0.1f), "Compression ratio = 16x");
}

/* ============================================================================
 * EXHAUSTIVE SMALL MATRIX TEST
 * ============================================================================ */

void test_exhaustive_2x2(void) {
    printf("\n=== Exhaustive 2x2 Ternary Matvec ===\n");

    /* Test all possible 2x2 ternary matrices (3^4 = 81)
     * against reference float computation */

    int8_t vals[] = {-1, 0, 1};
    float x[2] = {2.0f, 3.0f};
    int all_correct = 1;
    int count = 0;

    for (int w00 = 0; w00 < 3; w00++) {
        for (int w01 = 0; w01 < 3; w01++) {
            for (int w10 = 0; w10 < 3; w10++) {
                for (int w11 = 0; w11 < 3; w11++) {
                    /* Reference: float computation */
                    float ref_y0 = vals[w00] * x[0] + vals[w01] * x[1];
                    float ref_y1 = vals[w10] * x[0] + vals[w11] * x[1];

                    /* Ternary computation */
                    uint8_t W[2];
                    W[0] = trit_pack4(vals[w00], vals[w01], 0, 0);
                    W[1] = trit_pack4(vals[w10], vals[w11], 0, 0);

                    float y[2];
                    ternary_matvec(W, x, y, 2, 2);

                    if (!FLOAT_EQ(y[0], ref_y0, 1e-6f) ||
                        !FLOAT_EQ(y[1], ref_y1, 1e-6f)) {
                        all_correct = 0;
                    }
                    count++;
                }
            }
        }
    }

    TEST(all_correct, "All 81 2x2 matrices match reference");
    printf("  Tested %d matrix configurations\n", count);
}

/* ============================================================================
 * SPARSITY TEST
 * ============================================================================ */

void test_sparsity(void) {
    printf("\n=== Sparsity Counting ===\n");

    uint8_t sparse[2];
    sparse[0] = trit_pack4(0, 0, 1, 0);  /* 1 nonzero */
    sparse[1] = trit_pack4(-1, 0, 0, 1); /* 2 nonzero */

    int nnz = ternary_count_nonzero(sparse, 8);
    TEST(nnz == 3, "Count nonzero = 3");

    uint8_t dense[1];
    dense[0] = trit_pack4(1, -1, 1, -1);
    int nnz_dense = ternary_count_nonzero(dense, 4);
    TEST(nnz_dense == 4, "Dense count = 4");
}

/* ============================================================================
 * MAIN
 * ============================================================================ */

int main(void) {
    printf("===================================================\n");
    printf("  YINSEN TERNARY TEST SUITE\n");
    printf("===================================================\n");

    test_trit_encode_decode();
    test_ternary_dot();
    test_ternary_matvec();
    test_quantize();
    test_roundtrip();
    test_compression();
    test_exhaustive_2x2();
    test_sparsity();

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
