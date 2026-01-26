/*
 * bench_epilogue.c - Benchmark SME epilogue kernel
 */
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <mach/mach_time.h>

extern void sme_epilogue_bias_gelu_scale_quant(
    int8_t* dst,
    const float* src,
    const float* bias,
    const float* scale
);

static uint64_t get_time_ns(void) {
    static mach_timebase_info_data_t info = {0};
    if (info.denom == 0) mach_timebase_info(&info);
    return mach_absolute_time() * info.numer / info.denom;
}

int main(void) {
    printf("=== SME Epilogue Kernel Benchmark ===\n\n");
    
    // Allocate aligned buffers
    float src[256] __attribute__((aligned(64)));
    float bias[16] __attribute__((aligned(64)));
    float scale[16] __attribute__((aligned(64)));
    int8_t dst[256] __attribute__((aligned(64)));
    
    // Initialize
    for (int i = 0; i < 256; i++) src[i] = ((float)(i % 100) - 50) * 0.1f;
    for (int i = 0; i < 16; i++) {
        bias[i] = 0.1f;
        scale[i] = 20.0f;
    }
    
    // Warmup
    for (int i = 0; i < 100; i++) {
        sme_epilogue_bias_gelu_scale_quant(dst, src, bias, scale);
    }
    
    // Benchmark
    const int iterations = 10000;
    uint64_t start = get_time_ns();
    
    for (int i = 0; i < iterations; i++) {
        sme_epilogue_bias_gelu_scale_quant(dst, src, bias, scale);
    }
    
    uint64_t end = get_time_ns();
    
    double total_ns = (double)(end - start);
    double per_call_ns = total_ns / iterations;
    
    // Each call processes 256 elements with: bias add, GELU (7 ops), scale mul, quant
    // ~10 FLOPs per element = 2560 FLOPs per call
    double flops_per_call = 256.0 * 10.0;
    double gflops = (flops_per_call * iterations) / total_ns;
    
    printf("16x16 epilogue (bias + GELU + scale + quant):\n");
    printf("  Iterations:    %d\n", iterations);
    printf("  Total time:    %.2f ms\n", total_ns / 1e6);
    printf("  Per call:      %.1f ns\n", per_call_ns);
    printf("  Throughput:    %.2f GFLOP/s\n", gflops);
    printf("  Elements/sec:  %.2f M/s\n", (256.0 * iterations) / (total_ns / 1e3));
    
    // Compare to reference (scalar)
    printf("\nScalar reference:\n");
    start = get_time_ns();
    for (int iter = 0; iter < iterations; iter++) {
        for (int r = 0; r < 16; r++) {
            float b = bias[r];
            float s = scale[r];
            for (int c = 0; c < 16; c++) {
                float x = src[r * 16 + c] + b;
                float x2 = x * x;
                float poly = 0.344675f + x2 * (-0.029813f);
                poly = x * poly + 0.5f;
                if (poly < 0) poly = 0;
                if (poly > 1) poly = 1;
                float y = x * poly * s;
                int32_t v = (int32_t)y;
                if (v > 127) v = 127;
                if (v < -128) v = -128;
                dst[r * 16 + c] = (int8_t)v;
            }
        }
    }
    end = get_time_ns();
    
    double ref_ns = (double)(end - start) / iterations;
    printf("  Per call:      %.1f ns\n", ref_ns);
    printf("  Speedup:       %.1fx\n", ref_ns / per_call_ns);
    
    return 0;
}
