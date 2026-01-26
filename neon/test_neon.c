/*
 * test_neon.c - Test NEON ternary matmul
 */
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <mach/mach_time.h>

// External declarations
extern void neon_ternary_matvec_ref(
    int32_t* out, const int8_t* act, const uint8_t* wgt, int N, int K);

extern void neon_ternary_matvec(
    int32_t* out, const int8_t* act, const uint8_t* wgt, int N, int K);

#if defined(__ARM_FEATURE_DOTPROD)
extern void neon_ternary_matvec_sdot(
    int32_t* out, const int8_t* act, const uint8_t* wgt, int N, int K);
extern void neon_ternary_matvec_sdot_4oc(
    int32_t* out, const int8_t* act, const uint8_t* wgt, int N, int K);
extern void neon_ternary_matvec_sdot_8oc(
    int32_t* out, const int8_t* act, const uint8_t* wgt, int N, int K);
#endif

extern void pack_weights_k_vertical(
    uint8_t* packed, const int8_t* weights, int N, int K);

static uint64_t get_time_ns(void) {
    static mach_timebase_info_data_t info = {0};
    if (info.denom == 0) mach_timebase_info(&info);
    return mach_absolute_time() * info.numer / info.denom;
}

static uint32_t xorshift(uint32_t* state) {
    uint32_t x = *state;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    *state = x;
    return x;
}

int main(void) {
    printf("=== NEON Ternary MatVec Test ===\n\n");
    
    // Test with realistic LLM dimensions
    const int N = 4096;   // Output channels (hidden dim)
    const int K = 4096;   // Input channels
    const int K_packed = K / 4;
    
    // Allocate
    int8_t* weights = malloc(N * K);
    uint8_t* packed = malloc(N * K_packed);
    int8_t* act = malloc(K);
    int32_t* out_ref = malloc(N * sizeof(int32_t));
    int32_t* out_neon = malloc(N * sizeof(int32_t));
    
    // Initialize with random ternary weights and activations
    uint32_t rng = 42;
    for (int i = 0; i < N * K; i++) {
        int r = xorshift(&rng) % 3;
        weights[i] = (r == 0) ? 0 : (r == 1) ? 1 : -1;
    }
    for (int i = 0; i < K; i++) {
        act[i] = (int8_t)((xorshift(&rng) % 256) - 128);
    }
    
    // Pack weights
    printf("Packing weights...\n");
    pack_weights_k_vertical(packed, weights, N, K);
    
    // Verify packing
    int pack_errors = 0;
    for (int n = 0; n < N && pack_errors < 5; n++) {
        for (int k = 0; k < K; k += 4) {
            uint8_t byte = packed[n * K_packed + k/4];
            for (int i = 0; i < 4; i++) {
                int trit = (byte >> (i * 2)) & 0x3;
                int8_t expected = weights[n * K + k + i];
                int8_t decoded = (trit == 1) ? 1 : (trit == 2) ? -1 : 0;
                if (decoded != expected) {
                    printf("Pack error at [%d,%d]: expected %d, got %d (trit=%d)\n",
                           n, k+i, expected, decoded, trit);
                    pack_errors++;
                }
            }
        }
    }
    if (pack_errors == 0) {
        printf("  Packing verified OK\n\n");
    }
    
    // Run reference
    printf("Running reference...\n");
    memset(out_ref, 0, N * sizeof(int32_t));
    neon_ternary_matvec_ref(out_ref, act, packed, N, K);
    
    // Run NEON
    printf("Running NEON...\n");
    memset(out_neon, 0, N * sizeof(int32_t));
    neon_ternary_matvec(out_neon, act, packed, N, K);
    
    // Compare
    int errors = 0;
    for (int i = 0; i < N; i++) {
        if (out_ref[i] != out_neon[i]) {
            errors++;
            if (errors <= 5) {
                printf("  Mismatch [%d]: ref=%d, neon=%d\n", 
                       i, out_ref[i], out_neon[i]);
            }
        }
    }
    
    printf("\nResults: %d/%d correct\n", N - errors, N);
    if (errors == 0) {
        printf("[PASSED] NEON matches reference!\n\n");
    } else {
        printf("[FAILED] %d errors\n\n", errors);
    }
    
    // Benchmark
    printf("Benchmark:\n");
    const int iters = 10000;
    
    // Reference
    uint64_t t0 = get_time_ns();
    for (int i = 0; i < iters; i++) {
        neon_ternary_matvec_ref(out_ref, act, packed, N, K);
    }
    uint64_t t1 = get_time_ns();
    double ref_ns = (double)(t1 - t0) / iters;
    
    // NEON TBL
    t0 = get_time_ns();
    for (int i = 0; i < iters; i++) {
        neon_ternary_matvec(out_neon, act, packed, N, K);
    }
    t1 = get_time_ns();
    double neon_ns = (double)(t1 - t0) / iters;
    
    double ops = 2.0 * N * K;  // Each element: multiply + accumulate
    double ref_gops = ops / ref_ns;
    double neon_gops = ops / neon_ns;
    
    printf("  Reference: %.1f us (%.2f GOP/s)\n", ref_ns / 1000, ref_gops);
    printf("  NEON TBL:  %.1f us (%.2f GOP/s)\n", neon_ns / 1000, neon_gops);
    printf("  Speedup:   %.1fx\n", ref_ns / neon_ns);
    
#if defined(__ARM_FEATURE_DOTPROD)
    int32_t* out_sdot = malloc(N * sizeof(int32_t));
    
    // Verify SDOT 8OC
    memset(out_sdot, 0, N * sizeof(int32_t));
    neon_ternary_matvec_sdot_8oc(out_sdot, act, packed, N, K);
    
    int sdot_errors = 0;
    for (int i = 0; i < N; i++) {
        if (out_ref[i] != out_sdot[i]) {
            sdot_errors++;
            if (sdot_errors <= 5) {
                printf("  SDOT 8OC mismatch [%d]: ref=%d, sdot=%d\n",
                       i, out_ref[i], out_sdot[i]);
            }
        }
    }
    if (sdot_errors == 0) {
        printf("\nSDOT 8OC verification: PASSED\n");
    } else {
        printf("\nSDOT 8OC verification: FAILED (%d errors)\n", sdot_errors);
    }
    
    // Benchmark SDOT 1OC
    t0 = get_time_ns();
    for (int i = 0; i < iters; i++) {
        neon_ternary_matvec_sdot(out_sdot, act, packed, N, K);
    }
    t1 = get_time_ns();
    double sdot_1oc_ns = (double)(t1 - t0) / iters;
    double sdot_1oc_gops = ops / sdot_1oc_ns;
    
    // Benchmark SDOT 4OC
    t0 = get_time_ns();
    for (int i = 0; i < iters; i++) {
        neon_ternary_matvec_sdot_4oc(out_sdot, act, packed, N, K);
    }
    t1 = get_time_ns();
    double sdot_4oc_ns = (double)(t1 - t0) / iters;
    double sdot_4oc_gops = ops / sdot_4oc_ns;
    
    // Benchmark SDOT 8OC (the new hotness)
    t0 = get_time_ns();
    for (int i = 0; i < iters; i++) {
        neon_ternary_matvec_sdot_8oc(out_sdot, act, packed, N, K);
    }
    t1 = get_time_ns();
    double sdot_8oc_ns = (double)(t1 - t0) / iters;
    double sdot_8oc_gops = ops / sdot_8oc_ns;
    
    printf("  SDOT 1OC: %.1f us (%.2f GOP/s)\n", sdot_1oc_ns / 1000, sdot_1oc_gops);
    printf("  SDOT 4OC: %.1f us (%.2f GOP/s)\n", sdot_4oc_ns / 1000, sdot_4oc_gops);
    printf("  SDOT 8OC: %.1f us (%.2f GOP/s)\n", sdot_8oc_ns / 1000, sdot_8oc_gops);
    printf("  8OC vs 1OC: %.1fx\n", sdot_1oc_ns / sdot_8oc_ns);
    
    free(out_sdot);
    
    // Use best result for estimate
    double best_gops = sdot_8oc_gops;
#else
    double best_gops = neon_gops;
#endif
    
    // Estimate for 7B model
    printf("\n7B Model Estimate:\n");
    // Total ops per token: ~6.5B ternary ops
    double ops_per_token = 6.5e9;
    double ns_per_token = ops_per_token / (best_gops * 1e9) * 1e9;
    double tok_per_sec = 1e9 / ns_per_token;
    printf("  Est. tok/sec: %.1f\n", tok_per_sec);
    
    free(weights);
    free(packed);
    free(act);
    free(out_ref);
    free(out_neon);
    
    return errors > 0 ? 1 : 0;
}
