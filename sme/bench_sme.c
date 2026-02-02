/*
 * bench_sme.c - Benchmarks for SME 16x16 ternary kernels
 *
 * Measures throughput in Gop/s (billions of ternary multiply-accumulate operations per second)
 * for comparison with Metal 8x8 kernels.
 *
 * Copyright 2026 Trix Research
 */

#include "ternary_sme.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <mach/mach_time.h>

/* ============================================================================
 * Timing Utilities
 * ============================================================================ */

static double get_time_ns(void) {
    static mach_timebase_info_data_t timebase = {0};
    if (timebase.denom == 0) {
        mach_timebase_info(&timebase);
    }
    return (double)mach_absolute_time() * timebase.numer / timebase.denom;
}

/* ============================================================================
 * PRNG for reproducible random data
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
    uint32_t w = 0;
    for (int i = 0; i < 16; i++) {
        uint32_t trit = xorshift32(state) % 3;
        w |= (trit << (i * 2));
    }
    return w;
}

/* ============================================================================
 * Benchmarks
 * ============================================================================ */

static void bench_dot16(void) {
    printf("Benchmark: dot16 (16-element ternary dot product)\n");
    
    const int WARMUP = 10000;
    const int ITERS = 1000000;
    
    uint32_t rng = 0xDEADBEEF;
    
    // Generate test data
    float activations[16];
    for (int i = 0; i < 16; i++) {
        activations[i] = random_float(&rng);
    }
    uint32_t weights = random_weights(&rng);
    
    // Warmup
    volatile float sink = 0;
    for (int i = 0; i < WARMUP; i++) {
        sink += sme_dot16(activations, weights);
    }
    
    // Timed run
    double start = get_time_ns();
    for (int i = 0; i < ITERS; i++) {
        sink += sme_dot16(activations, weights);
    }
    double end = get_time_ns();
    
    double elapsed_s = (end - start) / 1e9;
    double ops_per_call = 16;  // 16 multiply-adds
    double total_ops = (double)ITERS * ops_per_call;
    double gops = total_ops / elapsed_s / 1e9;
    
    printf("  Iterations:  %d\n", ITERS);
    printf("  Time:        %.3f ms\n", (end - start) / 1e6);
    printf("  Throughput:  %.2f Gop/s\n", gops);
    printf("  Per call:    %.1f ns\n", (end - start) / ITERS);
    printf("\n");
    
    (void)sink;
}

static void bench_matvec_16x16(void) {
    printf("Benchmark: matvec 16x16\n");
    
    const int WARMUP = 1000;
    const int ITERS = 100000;
    
    uint32_t rng = 0xCAFEBABE;
    
    // Generate test data
    float input[16];
    for (int i = 0; i < 16; i++) {
        input[i] = random_float(&rng);
    }
    
    uint32_t weights[16];
    for (int i = 0; i < 16; i++) {
        weights[i] = random_weights(&rng);
    }
    
    float output[16];
    
    // Warmup
    for (int i = 0; i < WARMUP; i++) {
        sme_matvec(output, weights, input, 16, 16);
    }
    
    // Timed run
    double start = get_time_ns();
    for (int i = 0; i < ITERS; i++) {
        sme_matvec(output, weights, input, 16, 16);
    }
    double end = get_time_ns();
    
    double elapsed_s = (end - start) / 1e9;
    double ops_per_call = 16 * 16;  // 256 multiply-adds
    double total_ops = (double)ITERS * ops_per_call;
    double gops = total_ops / elapsed_s / 1e9;
    
    printf("  Iterations:  %d\n", ITERS);
    printf("  Time:        %.3f ms\n", (end - start) / 1e6);
    printf("  Throughput:  %.2f Gop/s\n", gops);
    printf("  Per call:    %.1f ns\n", (end - start) / ITERS);
    printf("\n");
}

static void bench_matvec_32x32(void) {
    printf("Benchmark: matvec 32x32\n");

    const int WARMUP = 1000;
    const int ITERS = 100000;

    uint32_t rng = 0xFACEFEED;

    /* Generate test data */
    float input[32];
    for (int i = 0; i < 32; i++) {
        input[i] = random_float(&rng);
    }

    uint32_t weights[64];  /* 4 tiles of 16 */
    for (int i = 0; i < 64; i++) {
        weights[i] = random_weights(&rng);
    }

    float output[32];

    /* Warmup */
    for (int i = 0; i < WARMUP; i++) {
        sme_matvec(output, weights, input, 32, 32);
    }

    /* Timed run */
    double start = get_time_ns();
    for (int i = 0; i < ITERS; i++) {
        sme_matvec(output, weights, input, 32, 32);
    }
    double end = get_time_ns();

    double elapsed_s = (end - start) / 1e9;
    double ops_per_call = 32 * 32;  /* 1024 ternary multiply-adds */
    double total_ops = (double)ITERS * ops_per_call;
    double gops = total_ops / elapsed_s / 1e9;

    printf("  Iterations:  %d\n", ITERS);
    printf("  Time:        %.3f ms\n", (end - start) / 1e6);
    printf("  Throughput:  %.2f Gop/s\n", gops);
    printf("  Per call:    %.1f ns\n", (end - start) / ITERS);
    printf("\n");
}

static void bench_matvec_large(size_t M, size_t K) {
    printf("Benchmark: matvec %zux%zu\n", M, K);
    
    const int WARMUP = 100;
    const int ITERS = 1000;
    
    uint32_t rng = 0x12345678;
    
    // Allocate test data
    float* input = malloc(K * sizeof(float));
    float* output = malloc(M * sizeof(float));
    
    size_t weight_size = sme_weight_buffer_size(M, K);
    uint32_t* weights = malloc(weight_size);
    
    // Generate random data
    for (size_t i = 0; i < K; i++) {
        input[i] = random_float(&rng);
    }
    
    // Generate random weights (direct into SME format for simplicity)
    for (size_t i = 0; i < weight_size / sizeof(uint32_t); i++) {
        weights[i] = random_weights(&rng);
    }
    
    // Warmup
    for (int i = 0; i < WARMUP; i++) {
        sme_matvec(output, weights, input, M, K);
    }
    
    // Timed run
    double start = get_time_ns();
    for (int i = 0; i < ITERS; i++) {
        sme_matvec(output, weights, input, M, K);
    }
    double end = get_time_ns();
    
    double elapsed_s = (end - start) / 1e9;
    double ops_per_call = (double)M * K;
    double total_ops = (double)ITERS * ops_per_call;
    double gops = total_ops / elapsed_s / 1e9;
    
    printf("  Iterations:  %d\n", ITERS);
    printf("  Time:        %.3f ms\n", (end - start) / 1e6);
    printf("  Throughput:  %.2f Gop/s\n", gops);
    printf("  Per call:    %.3f ms\n", (end - start) / ITERS / 1e6);
    printf("\n");
    
    free(input);
    free(output);
    free(weights);
}

static void bench_matmul(size_t M, size_t K, size_t N) {
    printf("Benchmark: matmul %zux%zux%zu\n", M, K, N);
    
    const int WARMUP = 10;
    const int ITERS = 100;
    
    uint32_t rng = 0xABCDEF01;
    
    // Allocate
    float* A = malloc(M * K * sizeof(float));
    float* C = malloc(M * N * sizeof(float));
    
    size_t K_aligned = (K + 15) & ~15;
    size_t N_aligned = (N + 15) & ~15;
    size_t weight_size = (K_aligned / 16) * (N_aligned / 16) * 16 * sizeof(uint32_t);
    uint32_t* W = malloc(weight_size);
    
    // Generate random data
    for (size_t i = 0; i < M * K; i++) {
        A[i] = random_float(&rng);
    }
    for (size_t i = 0; i < weight_size / sizeof(uint32_t); i++) {
        W[i] = random_weights(&rng);
    }
    
    // Warmup
    for (int i = 0; i < WARMUP; i++) {
        sme_matmul(C, A, W, M, K, N);
    }
    
    // Timed run
    double start = get_time_ns();
    for (int i = 0; i < ITERS; i++) {
        sme_matmul(C, A, W, M, K, N);
    }
    double end = get_time_ns();
    
    double elapsed_s = (end - start) / 1e9;
    double ops_per_call = (double)M * K * N;
    double total_ops = (double)ITERS * ops_per_call;
    double gops = total_ops / elapsed_s / 1e9;
    
    printf("  Iterations:  %d\n", ITERS);
    printf("  Time:        %.3f ms\n", (end - start) / 1e6);
    printf("  Throughput:  %.2f Gop/s\n", gops);
    printf("  Per call:    %.3f ms\n", (end - start) / ITERS / 1e6);
    printf("\n");
    
    free(A);
    free(C);
    free(W);
}

static void bench_batch(size_t M, size_t K, size_t batch_size) {
    printf("Benchmark: matvec_batch %zux%zu x %zu\n", M, K, batch_size);
    
    const int WARMUP = 10;
    const int ITERS = 100;
    
    uint32_t rng = 0x87654321;
    
    // Allocate
    float* input = malloc(batch_size * K * sizeof(float));
    float* output = malloc(batch_size * M * sizeof(float));
    
    size_t weight_size = sme_weight_buffer_size(M, K);
    uint32_t* weights = malloc(weight_size);
    
    // Generate random data
    for (size_t i = 0; i < batch_size * K; i++) {
        input[i] = random_float(&rng);
    }
    for (size_t i = 0; i < weight_size / sizeof(uint32_t); i++) {
        weights[i] = random_weights(&rng);
    }
    
    // Warmup
    for (int i = 0; i < WARMUP; i++) {
        sme_matvec_batch(output, weights, input, M, K, batch_size);
    }
    
    // Timed run
    double start = get_time_ns();
    for (int i = 0; i < ITERS; i++) {
        sme_matvec_batch(output, weights, input, M, K, batch_size);
    }
    double end = get_time_ns();
    
    double elapsed_s = (end - start) / 1e9;
    double ops_per_call = (double)M * K * batch_size;
    double total_ops = (double)ITERS * ops_per_call;
    double gops = total_ops / elapsed_s / 1e9;
    
    printf("  Iterations:  %d\n", ITERS);
    printf("  Time:        %.3f ms\n", (end - start) / 1e6);
    printf("  Throughput:  %.2f Gop/s\n", gops);
    printf("  Per call:    %.3f ms\n", (end - start) / ITERS / 1e6);
    printf("  Per vector:  %.3f us\n", (end - start) / ITERS / batch_size / 1e3);
    printf("\n");
    
    free(input);
    free(output);
    free(weights);
}

/* ============================================================================
 * Main
 * ============================================================================ */

int main(int argc, char** argv) {
    printf("\n");
    printf("=================================================================\n");
    printf("  Yinsen SME 16x16 Ternary Kernel Benchmarks\n");
    printf("=================================================================\n");
    printf("\n");
    
    printf("Hardware:\n");
    printf("  SME available: %s\n", sme_available() ? "YES" : "NO");
    printf("\n");
    
    printf("-----------------------------------------------------------------\n");
    printf("Small operations:\n");
    printf("-----------------------------------------------------------------\n");
    bench_dot16();
    bench_matvec_16x16();
    bench_matvec_32x32();
    
    printf("-----------------------------------------------------------------\n");
    printf("Matrix-vector (typical layer sizes):\n");
    printf("-----------------------------------------------------------------\n");
    bench_matvec_large(256, 256);
    bench_matvec_large(512, 512);
    bench_matvec_large(1024, 1024);
    bench_matvec_large(2048, 2048);
    bench_matvec_large(4096, 4096);
    
    printf("-----------------------------------------------------------------\n");
    printf("Full matrix multiply (for comparison with Metal):\n");
    printf("-----------------------------------------------------------------\n");
    bench_matmul(256, 256, 256);
    bench_matmul(512, 512, 512);
    bench_matmul(1024, 1024, 1024);
    
    printf("-----------------------------------------------------------------\n");
    printf("Batched operations (SME sweet spot):\n");
    printf("-----------------------------------------------------------------\n");
    bench_batch(512, 512, 1);
    bench_batch(512, 512, 8);
    bench_batch(512, 512, 16);
    bench_batch(512, 512, 32);
    bench_batch(512, 512, 64);
    
    printf("=================================================================\n");
    printf("  Benchmarks complete\n");
    printf("=================================================================\n");
    printf("\n");
    
    return 0;
}
