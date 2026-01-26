/*
 * YINSEN Test Suite - Frozen Shapes
 *
 * Verifies the mathematical correctness of all frozen shapes.
 * Every test here is falsifiable - it either passes or fails.
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "../include/apu.h"
#include "../include/onnx_shapes.h"

/* ============================================================================
 * TEST FRAMEWORK
 * ============================================================================ */

static int total_tests = 0;
static int total_passed = 0;

#define FLOAT_EQ(a, b, tol) (fabsf((a) - (b)) < (tol))

#define TEST(cond, name) do { \
    total_tests++; \
    if (cond) { \
        total_passed++; \
    } else { \
        printf("  FAIL: %s\n", name); \
    } \
} while(0)

/* ============================================================================
 * LOGIC SHAPES - Complete Truth Tables
 * ============================================================================ */

void test_logic_truth_tables(void) {
    printf("\n=== Logic Shapes - Truth Tables ===\n");

    /* XOR: a + b - 2ab */
    TEST(FLOAT_EQ(yinsen_xor(0.0f, 0.0f), 0.0f, 1e-6f), "XOR(0,0)=0");
    TEST(FLOAT_EQ(yinsen_xor(0.0f, 1.0f), 1.0f, 1e-6f), "XOR(0,1)=1");
    TEST(FLOAT_EQ(yinsen_xor(1.0f, 0.0f), 1.0f, 1e-6f), "XOR(1,0)=1");
    TEST(FLOAT_EQ(yinsen_xor(1.0f, 1.0f), 0.0f, 1e-6f), "XOR(1,1)=0");

    /* AND: a * b */
    TEST(FLOAT_EQ(yinsen_and(0.0f, 0.0f), 0.0f, 1e-6f), "AND(0,0)=0");
    TEST(FLOAT_EQ(yinsen_and(0.0f, 1.0f), 0.0f, 1e-6f), "AND(0,1)=0");
    TEST(FLOAT_EQ(yinsen_and(1.0f, 0.0f), 0.0f, 1e-6f), "AND(1,0)=0");
    TEST(FLOAT_EQ(yinsen_and(1.0f, 1.0f), 1.0f, 1e-6f), "AND(1,1)=1");

    /* OR: a + b - ab */
    TEST(FLOAT_EQ(yinsen_or(0.0f, 0.0f), 0.0f, 1e-6f), "OR(0,0)=0");
    TEST(FLOAT_EQ(yinsen_or(0.0f, 1.0f), 1.0f, 1e-6f), "OR(0,1)=1");
    TEST(FLOAT_EQ(yinsen_or(1.0f, 0.0f), 1.0f, 1e-6f), "OR(1,0)=1");
    TEST(FLOAT_EQ(yinsen_or(1.0f, 1.0f), 1.0f, 1e-6f), "OR(1,1)=1");

    /* NOT: 1 - a */
    TEST(FLOAT_EQ(yinsen_not(0.0f), 1.0f, 1e-6f), "NOT(0)=1");
    TEST(FLOAT_EQ(yinsen_not(1.0f), 0.0f, 1e-6f), "NOT(1)=0");

    /* NAND: 1 - ab */
    TEST(FLOAT_EQ(yinsen_nand(0.0f, 0.0f), 1.0f, 1e-6f), "NAND(0,0)=1");
    TEST(FLOAT_EQ(yinsen_nand(0.0f, 1.0f), 1.0f, 1e-6f), "NAND(0,1)=1");
    TEST(FLOAT_EQ(yinsen_nand(1.0f, 0.0f), 1.0f, 1e-6f), "NAND(1,0)=1");
    TEST(FLOAT_EQ(yinsen_nand(1.0f, 1.0f), 0.0f, 1e-6f), "NAND(1,1)=0");

    /* NOR: 1 - a - b + ab */
    TEST(FLOAT_EQ(yinsen_nor(0.0f, 0.0f), 1.0f, 1e-6f), "NOR(0,0)=1");
    TEST(FLOAT_EQ(yinsen_nor(0.0f, 1.0f), 0.0f, 1e-6f), "NOR(0,1)=0");
    TEST(FLOAT_EQ(yinsen_nor(1.0f, 0.0f), 0.0f, 1e-6f), "NOR(1,0)=0");
    TEST(FLOAT_EQ(yinsen_nor(1.0f, 1.0f), 0.0f, 1e-6f), "NOR(1,1)=0");

    /* XNOR: 1 - a - b + 2ab */
    TEST(FLOAT_EQ(yinsen_xnor(0.0f, 0.0f), 1.0f, 1e-6f), "XNOR(0,0)=1");
    TEST(FLOAT_EQ(yinsen_xnor(0.0f, 1.0f), 0.0f, 1e-6f), "XNOR(0,1)=0");
    TEST(FLOAT_EQ(yinsen_xnor(1.0f, 0.0f), 0.0f, 1e-6f), "XNOR(1,0)=0");
    TEST(FLOAT_EQ(yinsen_xnor(1.0f, 1.0f), 1.0f, 1e-6f), "XNOR(1,1)=1");
}

/* ============================================================================
 * FULL ADDER - Exhaustive Test
 * ============================================================================ */

void test_full_adder(void) {
    printf("\n=== Full Adder - All 8 Combinations ===\n");

    float expected_sum[8]   = {0, 1, 1, 0, 1, 0, 0, 1};
    float expected_carry[8] = {0, 0, 0, 1, 0, 1, 1, 1};

    int idx = 0;
    int all_pass = 1;
    for (int a = 0; a <= 1; a++) {
        for (int b = 0; b <= 1; b++) {
            for (int cin = 0; cin <= 1; cin++) {
                float sum, carry;
                yinsen_full_adder((float)a, (float)b, (float)cin, &sum, &carry);

                if (!FLOAT_EQ(sum, expected_sum[idx], 1e-6f) ||
                    !FLOAT_EQ(carry, expected_carry[idx], 1e-6f)) {
                    printf("  FAIL: FA(%d,%d,%d)\n", a, b, cin);
                    all_pass = 0;
                }
                idx++;
            }
        }
    }

    total_tests++;
    if (all_pass) total_passed++;
}

/* ============================================================================
 * RIPPLE ADDER - Exhaustive 8-bit Test (65,536 combinations)
 * ============================================================================ */

void test_ripple_adder(void) {
    printf("\n=== Ripple Adder - 256x256 Exhaustive ===\n");

    int errors = 0;

    for (int a = 0; a < 256; a++) {
        for (int b = 0; b < 256; b++) {
            float a_bits[8], b_bits[8], result_bits[8];
            for (int i = 0; i < 8; i++) {
                a_bits[i] = (a >> i) & 1 ? 1.0f : 0.0f;
                b_bits[i] = (b >> i) & 1 ? 1.0f : 0.0f;
            }

            float carry;
            yinsen_ripple_add_8bit(a_bits, b_bits, 0.0f, result_bits, &carry);

            int result = 0;
            for (int i = 0; i < 8; i++) {
                if (result_bits[i] > 0.5f) result |= (1 << i);
            }
            int carry_int = (carry > 0.5f) ? 1 : 0;

            int expected = (a + b) & 0xFF;
            int expected_carry = (a + b) > 255 ? 1 : 0;

            if (result != expected || carry_int != expected_carry) {
                errors++;
            }
        }
    }

    total_tests++;
    if (errors == 0) {
        total_passed++;
        printf("  All 65536 additions correct\n");
    } else {
        printf("  FAIL: %d errors\n", errors);
    }
}

/* ============================================================================
 * ONNX ACTIVATIONS
 * ============================================================================ */

void test_activations(void) {
    printf("\n=== ONNX Activations ===\n");

    /* ReLU */
    TEST(FLOAT_EQ(yinsen_relu(-1.0f), 0.0f, 1e-6f), "ReLU(-1)=0");
    TEST(FLOAT_EQ(yinsen_relu(0.0f), 0.0f, 1e-6f), "ReLU(0)=0");
    TEST(FLOAT_EQ(yinsen_relu(1.0f), 1.0f, 1e-6f), "ReLU(1)=1");

    /* Sigmoid */
    TEST(FLOAT_EQ(yinsen_sigmoid(0.0f), 0.5f, 1e-5f), "Sigmoid(0)=0.5");
    TEST(yinsen_sigmoid(-10.0f) < 0.001f, "Sigmoid(-10)~0");
    TEST(yinsen_sigmoid(10.0f) > 0.999f, "Sigmoid(10)~1");

    /* Tanh */
    TEST(FLOAT_EQ(yinsen_tanh(0.0f), 0.0f, 1e-6f), "Tanh(0)=0");
    TEST(yinsen_tanh(10.0f) > 0.999f, "Tanh(10)~1");
    TEST(yinsen_tanh(-10.0f) < -0.999f, "Tanh(-10)~-1");
}

/* ============================================================================
 * SOFTMAX
 * ============================================================================ */

void test_softmax(void) {
    printf("\n=== Softmax ===\n");

    float logits[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    float probs[4];

    yinsen_softmax(logits, probs, 4);

    float sum = probs[0] + probs[1] + probs[2] + probs[3];
    TEST(FLOAT_EQ(sum, 1.0f, 1e-5f), "Softmax sums to 1");
    TEST(probs[0] < probs[1] && probs[1] < probs[2] && probs[2] < probs[3],
         "Softmax preserves order");

    /* Numerical stability with large values */
    float large[3] = {1000.0f, 1001.0f, 1002.0f};
    float large_probs[3];
    yinsen_softmax(large, large_probs, 3);
    float large_sum = large_probs[0] + large_probs[1] + large_probs[2];
    TEST(FLOAT_EQ(large_sum, 1.0f, 1e-4f) && !isnan(large_probs[0]),
         "Softmax stable with large values");
}

/* ============================================================================
 * MATMUL
 * ============================================================================ */

void test_matmul(void) {
    printf("\n=== MatMul ===\n");

    float A[6] = {1, 2, 3, 4, 5, 6};  /* 2x3 */
    float B[6] = {1, 2, 3, 4, 5, 6};  /* 3x2 */
    float C[4];

    yinsen_matmul(A, B, C, 2, 2, 3);

    /* Expected: [[22, 28], [49, 64]] */
    TEST(FLOAT_EQ(C[0], 22.0f, 1e-4f), "MatMul[0,0]=22");
    TEST(FLOAT_EQ(C[1], 28.0f, 1e-4f), "MatMul[0,1]=28");
    TEST(FLOAT_EQ(C[2], 49.0f, 1e-4f), "MatMul[1,0]=49");
    TEST(FLOAT_EQ(C[3], 64.0f, 1e-4f), "MatMul[1,1]=64");
}

/* ============================================================================
 * MAIN
 * ============================================================================ */

int main(void) {
    printf("===================================================\n");
    printf("  YINSEN TEST SUITE - Frozen Shapes\n");
    printf("===================================================\n");

    test_logic_truth_tables();
    test_full_adder();
    test_ripple_adder();
    test_activations();
    test_softmax();
    test_matmul();

    printf("\n===================================================\n");
    printf("  RESULTS: %d/%d passed\n", total_passed, total_tests);
    printf("===================================================\n");

    if (total_passed == total_tests) {
        printf("  ALL TESTS PASSED\n");
        return 0;
    } else {
        printf("  SOME TESTS FAILED\n");
        return 1;
    }
}
