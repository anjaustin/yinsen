/*
 * bench_cfc_chip.c — Head-to-head: Chip vs Yinsen CfC vs Ternary CfC
 *
 * Falsification target: "The chip cannot improve yinsen's performance.
 * The algorithm is identical."
 *
 * What we measure:
 *   1. Correctness: Are chip and yinsen CfC bit-identical?
 *   2. Latency: Single-step inference at multiple sizes
 *   3. Throughput: Sustained sequence processing
 *   4. Ternary comparison: Does ternary CfC beat float CfC?
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>
#include <mach/mach_time.h>

#include "cfc.h"
#include "cfc_cell_chip.h"
#include "cfc_ternary.h"
#include "ternary.h"

/* ========================================================================
 * Timing
 * ======================================================================== */

static uint64_t get_time_ns(void) {
    static mach_timebase_info_data_t info = {0};
    if (info.denom == 0) mach_timebase_info(&info);
    return mach_absolute_time() * info.numer / info.denom;
}

/* ========================================================================
 * RNG (deterministic)
 * ======================================================================== */

static uint32_t rng_state = 12345;

static float randf(void) {
    rng_state ^= rng_state << 13;
    rng_state ^= rng_state >> 17;
    rng_state ^= rng_state << 5;
    return (float)(rng_state & 0xFFFF) / 65536.0f - 0.5f;
}

/* ========================================================================
 * Test a single size configuration
 * ======================================================================== */

typedef struct {
    int input_dim;
    int hidden_dim;
    int iters;
    /* Results */
    double chip_ns;
    double yinsen_ns;
    double chip_fixed_ns;
    double yinsen_fixed_ns;
    double ternary_ns;
    int bit_identical;
    int fixed_bit_identical;
    float max_error;
} BenchResult;

static void bench_size(int input_dim, int hidden_dim, int iters, BenchResult* result) {
    result->input_dim = input_dim;
    result->hidden_dim = hidden_dim;
    result->iters = iters;

    const int concat_dim = input_dim + hidden_dim;
    const int weight_count = hidden_dim * concat_dim;

    /* Allocate */
    float* W_gate = calloc(weight_count, sizeof(float));
    float* b_gate = calloc(hidden_dim, sizeof(float));
    float* W_cand = calloc(weight_count, sizeof(float));
    float* b_cand = calloc(hidden_dim, sizeof(float));
    float* tau = malloc(sizeof(float));
    tau[0] = 1.0f;

    float* x = malloc(input_dim * sizeof(float));
    float* h_prev = malloc(hidden_dim * sizeof(float));
    float* h_chip = malloc(hidden_dim * sizeof(float));
    float* h_yinsen = malloc(hidden_dim * sizeof(float));
    float* h_ternary = malloc(hidden_dim * sizeof(float));

    /* Initialize with deterministic random values */
    rng_state = 42 + input_dim * 1000 + hidden_dim;
    for (int i = 0; i < weight_count; i++) {
        W_gate[i] = randf() * 0.2f;
        W_cand[i] = randf() * 0.2f;
    }
    for (int i = 0; i < hidden_dim; i++) {
        b_gate[i] = randf() * 0.1f;
        b_cand[i] = randf() * 0.1f;
    }
    for (int i = 0; i < input_dim; i++) {
        x[i] = randf();
    }
    for (int i = 0; i < hidden_dim; i++) {
        h_prev[i] = randf() * 0.5f;
    }

    float dt = 0.01f;

    /* ── Correctness: Chip vs Yinsen (single call) ── */

    CfCParams yinsen_params = {
        .input_dim = input_dim,
        .hidden_dim = hidden_dim,
        .W_gate = W_gate,
        .b_gate = b_gate,
        .W_cand = W_cand,
        .b_cand = b_cand,
        .tau = tau,
        .tau_shared = 1
    };

    yinsen_cfc_cell(x, h_prev, dt, &yinsen_params, h_yinsen);

    CFC_CELL_GENERIC(x, h_prev, dt, W_gate, b_gate,
                     W_cand, b_cand, tau, 1, input_dim, hidden_dim, h_chip);

    result->bit_identical = 1;
    result->max_error = 0.0f;
    for (int i = 0; i < hidden_dim; i++) {
        if (h_chip[i] != h_yinsen[i]) {
            result->bit_identical = 0;
        }
        float err = fabsf(h_chip[i] - h_yinsen[i]);
        if (err > result->max_error) result->max_error = err;
    }

    /* ── Correctness: Fixed-dt variants ── */

    float* decay_pre = malloc(hidden_dim * sizeof(float));
    cfc_precompute_decay(tau, 1, dt, hidden_dim, decay_pre);

    float* h_chip_fixed = malloc(hidden_dim * sizeof(float));
    float* h_yinsen_fixed = malloc(hidden_dim * sizeof(float));

    CFC_CELL_FIXED(x, h_prev, W_gate, b_gate, W_cand, b_cand,
                   decay_pre, input_dim, hidden_dim, h_chip_fixed);

    CfCParamsFixed yinsen_fixed_params = {
        .input_dim = input_dim,
        .hidden_dim = hidden_dim,
        .W_gate = W_gate,
        .b_gate = b_gate,
        .W_cand = W_cand,
        .b_cand = b_cand,
        .decay = decay_pre
    };
    yinsen_cfc_cell_fixed(x, h_prev, &yinsen_fixed_params, h_yinsen_fixed);

    result->fixed_bit_identical = 1;
    for (int i = 0; i < hidden_dim; i++) {
        if (h_chip_fixed[i] != h_yinsen_fixed[i]) {
            result->fixed_bit_identical = 0;
        }
    }

    /* ── Benchmark: Chip GENERIC ── */
    {
        /* Warmup */
        for (int i = 0; i < 100; i++) {
            CFC_CELL_GENERIC(x, h_chip, dt, W_gate, b_gate,
                             W_cand, b_cand, tau, 1, input_dim, hidden_dim, h_chip);
        }
        memcpy(h_chip, h_prev, hidden_dim * sizeof(float));

        uint64_t t0 = get_time_ns();
        for (int i = 0; i < iters; i++) {
            CFC_CELL_GENERIC(x, h_chip, dt, W_gate, b_gate,
                             W_cand, b_cand, tau, 1, input_dim, hidden_dim, h_chip);
        }
        uint64_t t1 = get_time_ns();
        result->chip_ns = (double)(t1 - t0) / iters;
    }

    /* ── Benchmark: Yinsen GENERIC ── */
    {
        for (int i = 0; i < 100; i++) {
            yinsen_cfc_cell(x, h_yinsen, dt, &yinsen_params, h_yinsen);
        }
        memcpy(h_yinsen, h_prev, hidden_dim * sizeof(float));

        uint64_t t0 = get_time_ns();
        for (int i = 0; i < iters; i++) {
            yinsen_cfc_cell(x, h_yinsen, dt, &yinsen_params, h_yinsen);
        }
        uint64_t t1 = get_time_ns();
        result->yinsen_ns = (double)(t1 - t0) / iters;
    }

    /* ── Benchmark: Chip FIXED ── */
    {
        memcpy(h_chip, h_prev, hidden_dim * sizeof(float));
        for (int i = 0; i < 100; i++) {
            CFC_CELL_FIXED(x, h_chip, W_gate, b_gate, W_cand, b_cand,
                           decay_pre, input_dim, hidden_dim, h_chip);
        }
        memcpy(h_chip, h_prev, hidden_dim * sizeof(float));

        uint64_t t0 = get_time_ns();
        for (int i = 0; i < iters; i++) {
            CFC_CELL_FIXED(x, h_chip, W_gate, b_gate, W_cand, b_cand,
                           decay_pre, input_dim, hidden_dim, h_chip);
        }
        uint64_t t1 = get_time_ns();
        result->chip_fixed_ns = (double)(t1 - t0) / iters;
    }

    /* ── Benchmark: Yinsen FIXED ── */
    {
        memcpy(h_yinsen, h_prev, hidden_dim * sizeof(float));
        for (int i = 0; i < 100; i++) {
            yinsen_cfc_cell_fixed(x, h_yinsen, &yinsen_fixed_params, h_yinsen);
        }
        memcpy(h_yinsen, h_prev, hidden_dim * sizeof(float));

        uint64_t t0 = get_time_ns();
        for (int i = 0; i < iters; i++) {
            yinsen_cfc_cell_fixed(x, h_yinsen, &yinsen_fixed_params, h_yinsen);
        }
        uint64_t t1 = get_time_ns();
        result->yinsen_fixed_ns = (double)(t1 - t0) / iters;
    }

    /* ── Benchmark: Ternary CfC ── */
    {
        /* Quantize weights to ternary */
        int bytes_per_row = (concat_dim + 3) / 4;
        uint8_t* W_gate_t = calloc(hidden_dim * bytes_per_row, 1);
        uint8_t* W_cand_t = calloc(hidden_dim * bytes_per_row, 1);

        /* Quantize each row */
        for (int row = 0; row < hidden_dim; row++) {
            for (int col = 0; col < concat_dim; col += 4) {
                int remaining = concat_dim - col;
                if (remaining > 4) remaining = 4;

                uint8_t packed = 0;
                for (int k = 0; k < remaining; k++) {
                    float w = W_gate[row * concat_dim + col + k];
                    int8_t trit = (w > 0.05f) ? 1 : (w < -0.05f) ? -1 : 0;
                    uint8_t encoded = (trit == 1) ? 0x01 : (trit == -1) ? 0x02 : 0x00;
                    packed |= (encoded << (k * 2));
                }
                W_gate_t[row * bytes_per_row + col / 4] = packed;

                packed = 0;
                for (int k = 0; k < remaining; k++) {
                    float w = W_cand[row * concat_dim + col + k];
                    int8_t trit = (w > 0.05f) ? 1 : (w < -0.05f) ? -1 : 0;
                    uint8_t encoded = (trit == 1) ? 0x01 : (trit == -1) ? 0x02 : 0x00;
                    packed |= (encoded << (k * 2));
                }
                W_cand_t[row * bytes_per_row + col / 4] = packed;
            }
        }

        CfCTernaryParams ternary_params = {
            .input_dim = input_dim,
            .hidden_dim = hidden_dim,
            .W_gate = W_gate_t,
            .b_gate = b_gate,
            .W_cand = W_cand_t,
            .b_cand = b_cand,
            .tau = tau,
            .tau_shared = 1
        };

        memcpy(h_ternary, h_prev, hidden_dim * sizeof(float));
        for (int i = 0; i < 100; i++) {
            yinsen_cfc_ternary_cell(x, h_ternary, dt, &ternary_params, h_ternary);
        }
        memcpy(h_ternary, h_prev, hidden_dim * sizeof(float));

        uint64_t t0 = get_time_ns();
        for (int i = 0; i < iters; i++) {
            yinsen_cfc_ternary_cell(x, h_ternary, dt, &ternary_params, h_ternary);
        }
        uint64_t t1 = get_time_ns();
        result->ternary_ns = (double)(t1 - t0) / iters;

        free(W_gate_t);
        free(W_cand_t);
    }

    free(W_gate); free(b_gate); free(W_cand); free(b_cand);
    free(tau); free(x); free(h_prev);
    free(h_chip); free(h_yinsen); free(h_ternary);
    free(decay_pre); free(h_chip_fixed); free(h_yinsen_fixed);
}

/* ========================================================================
 * Main
 * ======================================================================== */

int main(void) {
    printf("================================================================\n");
    printf("  CfC BENCHMARK: Chip vs Yinsen vs Ternary\n");
    printf("  Falsification target: 'Chip cannot improve performance'\n");
    printf("================================================================\n\n");

    typedef struct { int in; int hid; int iters; const char* label; } Config;
    Config configs[] = {
        {4,   8,    1000000, "4x8   (tiny MCU)"},
        {4,   16,   500000,  "4x16  (small MCU)"},
        {8,   32,   200000,  "8x32  (medium)"},
        {32,  64,   50000,   "32x64 (large)"},
        {64,  128,  10000,   "64x128 (LLM-scale)"},
        {128, 256,  5000,    "128x256 (big)"},
    };
    int n_configs = sizeof(configs) / sizeof(configs[0]);

    BenchResult results[6];

    for (int c = 0; c < n_configs; c++) {
        printf("--- %s ---\n", configs[c].label);
        bench_size(configs[c].in, configs[c].hid, configs[c].iters, &results[c]);
        BenchResult* r = &results[c];

        printf("  Correctness (generic): %s",
               r->bit_identical ? "BIT-IDENTICAL" : "DIFFER");
        if (!r->bit_identical) {
            printf(" (max error: %.2e)", r->max_error);
        }
        printf("\n");
        printf("  Correctness (fixed):   %s\n",
               r->fixed_bit_identical ? "BIT-IDENTICAL" : "DIFFER");

        printf("  Generic:  chip=%.1f ns  yinsen=%.1f ns  ratio=%.3fx\n",
               r->chip_ns, r->yinsen_ns, r->chip_ns / r->yinsen_ns);
        printf("  Fixed-dt: chip=%.1f ns  yinsen=%.1f ns  ratio=%.3fx\n",
               r->chip_fixed_ns, r->yinsen_fixed_ns,
               r->chip_fixed_ns / r->yinsen_fixed_ns);
        printf("  Ternary:  %.1f ns  vs chip-generic: %.3fx\n",
               r->ternary_ns, r->ternary_ns / r->chip_ns);
        printf("  Fixed vs Generic speedup: %.3fx (chip)  %.3fx (yinsen)\n",
               r->chip_ns / r->chip_fixed_ns,
               r->yinsen_ns / r->yinsen_fixed_ns);
        printf("\n");
    }

    /* ── Summary table ── */
    printf("================================================================\n");
    printf("  SUMMARY TABLE\n");
    printf("================================================================\n\n");

    printf("%-14s | %8s %8s %5s | %8s %8s %5s | %8s %5s\n",
           "Size", "Chip", "Yinsen", "Ratio",
           "ChipFx", "YinFx", "Ratio",
           "Ternary", "vs Chip");
    printf("%-14s-+-%8s-%8s-%5s-+-%8s-%8s-%5s-+-%8s-%5s\n",
           "--------------", "--------", "--------", "-----",
           "--------", "--------", "-----",
           "--------", "-----");

    for (int c = 0; c < n_configs; c++) {
        BenchResult* r = &results[c];
        printf("%-14s | %7.0f %7.0f %5.3f | %7.0f %7.0f %5.3f | %7.0f %5.3f\n",
               configs[c].label,
               r->chip_ns, r->yinsen_ns, r->chip_ns / r->yinsen_ns,
               r->chip_fixed_ns, r->yinsen_fixed_ns,
               r->chip_fixed_ns / r->yinsen_fixed_ns,
               r->ternary_ns, r->ternary_ns / r->chip_ns);
    }

    /* ── Verdict ── */
    printf("\n================================================================\n");
    printf("  VERDICT\n");
    printf("================================================================\n\n");

    int all_identical = 1;
    int chip_ever_faster = 0;
    int ternary_ever_faster = 0;

    for (int c = 0; c < n_configs; c++) {
        if (!results[c].bit_identical) all_identical = 0;
        if (results[c].chip_ns < results[c].yinsen_ns * 0.95) chip_ever_faster = 1;
        if (results[c].ternary_ns < results[c].chip_ns * 0.95) ternary_ever_faster = 1;
    }

    printf("  Bit-identical outputs: %s\n",
           all_identical ? "YES (all sizes)" : "NO (divergence detected)");
    printf("  Chip faster than Yinsen (>5%%): %s\n",
           chip_ever_faster ? "YES -- CLAIM FALSIFIED" : "NO -- claim holds");
    printf("  Ternary faster than float (>5%%): %s\n",
           ternary_ever_faster ? "YES" : "NO");

    if (chip_ever_faster) {
        printf("\n  *** FALSIFIED: The chip IS faster than yinsen CfC. ***\n");
        printf("  Investigate: flat args vs struct, NaN-init overhead,\n");
        printf("  tau validation branches.\n");
    } else {
        printf("\n  CONFIRMED: Chip and Yinsen CfC have equivalent performance.\n");
        printf("  The algorithm is the bottleneck, not the API wrapper.\n");
    }

    printf("\n");
    return 0;
}
