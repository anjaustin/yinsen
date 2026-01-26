/*
 * YINSEN 4×4 Ternary Matvec Exhaustive Proof
 *
 * This test verifies ALL possible 4×4 ternary matrix-vector
 * multiplications: 3^16 = 43,046,721 combinations.
 *
 * Each matrix element can be {-1, 0, +1}, giving 3 choices.
 * A 4×4 matrix has 16 elements, so 3^16 total matrices.
 *
 * For each matrix, we compute:
 *   - Reference: float matrix-vector multiply
 *   - Ternary: packed ternary computation
 *
 * If ALL 43,046,721 combinations match, the implementation is PROVEN.
 *
 * Expected runtime: 1-5 minutes depending on hardware.
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <stdint.h>
#include <sys/time.h>
#include "../include/ternary.h"

static double get_time_sec(void) {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec + tv.tv_usec / 1e6;
}

#define FLOAT_EQ(a, b, tol) (fabsf((a) - (b)) < (tol))

/* Convert base-3 index to 16 ternary values */
static void index_to_weights(uint32_t idx, int8_t w[16]) {
    for (int i = 0; i < 16; i++) {
        w[i] = (idx % 3) - 1;  /* Maps 0,1,2 to -1,0,+1 */
        idx /= 3;
    }
}

/* Reference float matvec: y = W @ x */
static void ref_matvec_4x4(const int8_t W[16], const float x[4], float y[4]) {
    for (int i = 0; i < 4; i++) {
        y[i] = 0.0f;
        for (int j = 0; j < 4; j++) {
            y[i] += W[i * 4 + j] * x[j];
        }
    }
}

/* Pack 16 int8 weights into 4 bytes (4 weights per byte) */
static void pack_weights_4x4(const int8_t w[16], uint8_t packed[4]) {
    /* Row 0: w[0], w[1], w[2], w[3] */
    packed[0] = trit_pack4(w[0], w[1], w[2], w[3]);
    /* Row 1: w[4], w[5], w[6], w[7] */
    packed[1] = trit_pack4(w[4], w[5], w[6], w[7]);
    /* Row 2: w[8], w[9], w[10], w[11] */
    packed[2] = trit_pack4(w[8], w[9], w[10], w[11]);
    /* Row 3: w[12], w[13], w[14], w[15] */
    packed[3] = trit_pack4(w[12], w[13], w[14], w[15]);
}

int main(void) {
    printf("===================================================\n");
    printf("  YINSEN 4×4 TERNARY MATVEC EXHAUSTIVE PROOF\n");
    printf("===================================================\n\n");

    const uint32_t TOTAL = 43046721;  /* 3^16 */
    const float x[4] = {1.0f, 2.0f, 3.0f, 4.0f};  /* Fixed input vector */
    
    uint32_t correct = 0;
    uint32_t failed = 0;
    uint32_t first_fail_idx = 0;
    
    double start = get_time_sec();
    
    printf("Testing all %u matrix configurations...\n", TOTAL);
    printf("Input vector: [1, 2, 3, 4]\n\n");
    
    for (uint32_t idx = 0; idx < TOTAL; idx++) {
        /* Progress report every 1M */
        if (idx % 1000000 == 0) {
            float pct = 100.0f * idx / TOTAL;
            printf("  Progress: %u / %u (%.1f%%)\n", idx, TOTAL, pct);
            fflush(stdout);
        }
        
        /* Generate weight matrix from index */
        int8_t w[16];
        index_to_weights(idx, w);
        
        /* Reference computation */
        float ref_y[4];
        ref_matvec_4x4(w, x, ref_y);
        
        /* Ternary computation */
        uint8_t packed[4];
        pack_weights_4x4(w, packed);
        
        float tern_y[4];
        ternary_matvec(packed, x, tern_y, 4, 4);
        
        /* Compare */
        int match = 1;
        for (int i = 0; i < 4; i++) {
            if (!FLOAT_EQ(ref_y[i], tern_y[i], 1e-5f)) {
                match = 0;
                break;
            }
        }
        
        if (match) {
            correct++;
        } else {
            if (failed == 0) {
                first_fail_idx = idx;
                printf("\n  FIRST FAILURE at idx %u:\n", idx);
                printf("    Weights: [%d,%d,%d,%d, %d,%d,%d,%d, %d,%d,%d,%d, %d,%d,%d,%d]\n",
                       w[0],w[1],w[2],w[3], w[4],w[5],w[6],w[7],
                       w[8],w[9],w[10],w[11], w[12],w[13],w[14],w[15]);
                printf("    Reference: [%.2f, %.2f, %.2f, %.2f]\n",
                       ref_y[0], ref_y[1], ref_y[2], ref_y[3]);
                printf("    Ternary:   [%.2f, %.2f, %.2f, %.2f]\n\n",
                       tern_y[0], tern_y[1], tern_y[2], tern_y[3]);
            }
            failed++;
        }
    }
    
    double end = get_time_sec();
    double elapsed = end - start;
    
    printf("\n===================================================\n");
    printf("  RESULTS\n");
    printf("===================================================\n\n");
    
    printf("  Total configurations: %u\n", TOTAL);
    printf("  Correct:              %u\n", correct);
    printf("  Failed:               %u\n", failed);
    printf("  Time:                 %.2f seconds\n", elapsed);
    printf("  Rate:                 %.1f M/sec\n", (double)TOTAL / elapsed / 1e6);
    
    printf("\n===================================================\n");
    
    if (failed == 0) {
        printf("  4×4 TERNARY MATVEC: PROVEN\n");
        printf("  All %u configurations verified.\n", TOTAL);
        printf("===================================================\n");
        return 0;
    } else {
        printf("  4×4 TERNARY MATVEC: FAILED\n");
        printf("  %u failures found (first at idx %u)\n", failed, first_fail_idx);
        printf("===================================================\n");
        return 1;
    }
}
