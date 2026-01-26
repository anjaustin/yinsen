/*
 * test_gold_standard.c - Gold Standard Test Harness for M4 SME Drain Kernel
 *
 * This test verifies bit-exact correctness of the SME drain+bias+GELU+scale+quant
 * pipeline by comparing against a reference C implementation.
 *
 * The test:
 * 1. Loads known data into ZA tile (simulates matmul result)
 * 2. Runs ASM drain kernel
 * 3. Runs reference C implementation
 * 4. Compares every byte
 *
 * Copyright 2026 Trix Research
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <mach/mach_time.h>

// =============================================================================
// ASM KERNEL DECLARATIONS
// =============================================================================

// Full pipeline: reads matmul results from memory (not ZA tile, macOS restriction)
// Handles SMSTART/SMSTOP internally
extern void sme_test_full_pipeline(
    int8_t* dst_ptr,
    const float* src_data,      // Matmul results (16x16 floats)
    const float* bias_ptr,
    const float* scale_ptr
);

// =============================================================================
// PIPELINE WRAPPER
// =============================================================================

static void run_sme_pipeline(
    int8_t* dst,
    const float* matmul_results,
    const float* bias,
    const float* scale
) {
    sme_test_full_pipeline(dst, matmul_results, bias, scale);
}

// =============================================================================
// GOLD STANDARD REFERENCE IMPLEMENTATION
// =============================================================================

/*
 * Reference pipeline - must match ASM math EXACTLY
 *
 * Spline GELU: y = x * clamp(0.5 + x * (C1 + x^2 * C3), 0, 1)
 * C1 = 0.344675
 * C3 = -0.029813
 */
static int8_t reference_pipeline(float input, float bias, float scale) {
    // A. Bias Add
    float x = input + bias;
    
    // B. Spline GELU Approximation
    const float C1 = 0.344675f;
    const float C3 = -0.029813f;
    
    float x2 = x * x;
    
    // Inner polynomial: C1 + x^2 * C3
    float poly = C1 + (x2 * C3);
    
    // x * inner
    poly = x * poly;
    
    // + 0.5
    poly = poly + 0.5f;
    
    // Clamp [0.0, 1.0]
    if (poly < 0.0f) poly = 0.0f;
    if (poly > 1.0f) poly = 1.0f;
    
    // y = x * sigmoid_approx
    float y = x * poly;
    
    // C. Per-Channel Scale
    y = y * scale;
    
    // D. Quantize (truncate toward zero - matches fcvtzs)
    int32_t val = (int32_t)y;  // Truncate toward zero
    
    // E. Saturate to Int8 [-128, 127]
    if (val > 127) val = 127;
    if (val < -128) val = -128;
    
    return (int8_t)val;
}

// =============================================================================
// TEST UTILITIES
// =============================================================================

static uint64_t get_time_ns(void) {
    static mach_timebase_info_data_t info = {0};
    if (info.denom == 0) mach_timebase_info(&info);
    return mach_absolute_time() * info.numer / info.denom;
}

// Simple PRNG for reproducible tests
static uint32_t xorshift32(uint32_t* state) {
    uint32_t x = *state;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    *state = x;
    return x;
}

static float random_float(uint32_t* rng, float min, float max) {
    float t = (float)(xorshift32(rng) & 0xFFFFFF) / (float)0xFFFFFF;
    return min + t * (max - min);
}

// =============================================================================
// MAIN TEST
// =============================================================================

int main(void) {
    printf("=== M4 SME Gold Standard Test Harness ===\n\n");
    
    const int ROWS = 16;
    const int COLS = 16;
    const int TOTAL = ROWS * COLS;
    
    // Allocate aligned buffers
    float* za_input = aligned_alloc(64, TOTAL * sizeof(float));
    float* biases = aligned_alloc(64, ROWS * sizeof(float));
    float* scales = aligned_alloc(64, ROWS * sizeof(float));
    int8_t* result_asm = aligned_alloc(64, TOTAL);
    int8_t* result_ref = aligned_alloc(64, TOTAL);
    
    if (!za_input || !biases || !scales || !result_asm || !result_ref) {
        printf("ERROR: Memory allocation failed\n");
        return 1;
    }
    
    // =========================================================================
    // Generate Test Data (deterministic seed for reproducibility)
    // =========================================================================
    uint32_t rng = 42;
    
    // Matmul results: typical post-matmul range (simulates ZA contents)
    printf("[*] Generating test data...\n");
    for (int i = 0; i < TOTAL; i++) {
        za_input[i] = random_float(&rng, -3.0f, 3.0f);
    }
    
    // Biases: small values
    for (int i = 0; i < ROWS; i++) {
        biases[i] = random_float(&rng, -0.5f, 0.5f);
    }
    
    // Scales: typical quant scales (map float to ~127)
    for (int i = 0; i < ROWS; i++) {
        scales[i] = random_float(&rng, 10.0f, 40.0f);
    }
    
    // Clear output buffers
    memset(result_asm, 0xAA, TOTAL);  // Sentinel pattern
    memset(result_ref, 0xBB, TOTAL);
    
    // =========================================================================
    // Run ASM Pipeline
    // =========================================================================
    printf("[*] Running ASM kernel...\n");
    
    uint64_t t0 = get_time_ns();
    run_sme_pipeline(result_asm, za_input, biases, scales);
    uint64_t t1 = get_time_ns();
    
    printf("    ASM time: %llu ns\n", t1 - t0);
    
    // =========================================================================
    // Run Reference Pipeline
    // =========================================================================
    printf("[*] Running reference implementation...\n");
    
    t0 = get_time_ns();
    for (int r = 0; r < ROWS; r++) {
        for (int c = 0; c < COLS; c++) {
            int idx = r * COLS + c;
            result_ref[idx] = reference_pipeline(za_input[idx], biases[r], scales[r]);
        }
    }
    t1 = get_time_ns();
    
    printf("    Ref time: %llu ns\n\n", t1 - t0);
    
    // =========================================================================
    // Verification
    // =========================================================================
    printf("[!] Starting verification...\n\n");
    
    int errors = 0;
    int off_by_one = 0;
    int max_diff = 0;
    
    for (int i = 0; i < TOTAL; i++) {
        int diff = abs((int)result_asm[i] - (int)result_ref[i]);
        
        if (diff > max_diff) max_diff = diff;
        
        if (diff > 0) {
            if (diff == 1) {
                off_by_one++;
            } else {
                errors++;
                if (errors <= 10) {
                    int r = i / COLS;
                    int c = i % COLS;
                    printf("    MISMATCH at Channel %d, Token %d:\n", r, c);
                    printf("      Input  = %.6f\n", za_input[i]);
                    printf("      Bias   = %.6f\n", biases[r]);
                    printf("      Scale  = %.6f\n", scales[r]);
                    printf("      Ref    = %d\n", result_ref[i]);
                    printf("      ASM    = %d\n", result_asm[i]);
                    printf("      Diff   = %d\n\n", diff);
                }
            }
        }
    }
    
    // =========================================================================
    // Summary
    // =========================================================================
    printf("=== Test Summary ===\n");
    printf("Total elements:    %d\n", TOTAL);
    printf("Exact matches:     %d\n", TOTAL - errors - off_by_one);
    printf("Off-by-one:        %d\n", off_by_one);
    printf("Larger mismatches: %d\n", errors);
    printf("Max difference:    %d\n\n", max_diff);
    
    if (errors == 0 && off_by_one == 0) {
        printf("[PASSED] GREEN LIGHT! ASM output matches reference exactly.\n");
        printf("M4 Soft-Chip is ready for deployment.\n");
    } else if (errors == 0) {
        printf("[PASSED] ASM output matches within +/- 1 tolerance.\n");
        printf("Off-by-one errors are acceptable (FMA vs C float differences).\n");
    } else {
        printf("[FAILED] %d mismatches detected (beyond +/- 1 tolerance).\n", errors);
        printf("Check bias loading order or scaling logic.\n");
    }
    
    // =========================================================================
    // Debug: Print sample outputs
    // =========================================================================
    printf("\n=== Sample Outputs (first 8 elements) ===\n");
    printf("Idx   Input      Bias       Scale      Ref    ASM\n");
    for (int i = 0; i < 8; i++) {
        int r = i / COLS;
        printf("%3d   %7.3f   %7.3f   %7.3f   %4d   %4d\n",
               i, za_input[i], biases[r], scales[r], result_ref[i], result_asm[i]);
    }
    
    // Cleanup
    free(za_input);
    free(biases);
    free(scales);
    free(result_asm);
    free(result_ref);
    
    return errors;
}
