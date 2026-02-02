/*
 * Keystroke Biometric Probe — LMM experiment runner
 *
 * Answers the questions from the RAW phase:
 *   1. Execution time per keystroke
 *   2. Harder test: same speed, different rhythm (jitter only)
 *   3. Hardest test: same speed, same jitter, different key distribution
 *   4. What happens with distance_penalty = 0 (projection only)?
 *   5. What happens with dt-only input (no key_code)?
 *   6. Minimum enrollment length for separation
 *
 * Compile:
 *   cc -O2 -I../include -I../include/chips keystroke_probe.c -lm -o keystroke_probe
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <sys/time.h>

#include "onnx_shapes.h"
#include "chips/cfc_cell_chip.h"
#include "chips/norm_chip.h"
#include "chips/gemm_chip.h"
#include "chips/activation_chip.h"

#define INPUT_DIM   2
#define HIDDEN_DIM  8
#define OUTPUT_DIM  1
#define CONCAT_DIM  (INPUT_DIM + HIDDEN_DIM)

/* Same weights as the main demo */
static const float W_gate[HIDDEN_DIM * CONCAT_DIM] = {
     0.1f,  0.8f,   0.1f, -0.1f,  0.0f,  0.0f,  0.0f,  0.0f,  0.0f,  0.0f,
     0.7f,  0.2f,  -0.1f,  0.1f,  0.0f,  0.0f,  0.0f,  0.0f,  0.0f,  0.0f,
     0.4f,  0.5f,   0.0f,  0.0f,  0.1f, -0.1f,  0.0f,  0.0f,  0.0f,  0.0f,
     0.1f,  0.6f,   0.0f,  0.0f,  0.0f,  0.0f,  0.1f, -0.1f,  0.0f,  0.0f,
     0.6f,  0.1f,   0.0f,  0.0f,  0.0f,  0.0f,  0.0f,  0.0f,  0.1f, -0.1f,
     0.2f,  0.3f,   0.2f,  0.2f,  0.1f,  0.0f,  0.0f,  0.0f,  0.0f,  0.0f,
     0.3f,  0.3f,   0.1f,  0.0f,  0.0f,  0.1f,  0.0f,  0.0f,  0.0f,  0.0f,
     0.2f,  0.7f,  -0.1f,  0.0f,  0.0f,  0.0f,  0.0f,  0.1f,  0.0f,  0.0f,
};
static const float b_gate[HIDDEN_DIM] = {
    -0.5f, -0.3f, -0.4f, -0.5f, -0.3f, -0.4f, -0.3f, -0.5f
};
static const float W_cand[HIDDEN_DIM * CONCAT_DIM] = {
     0.5f,  0.3f,   0.1f,  0.0f, -0.1f,  0.0f,  0.0f,  0.0f,  0.0f,  0.0f,
     0.3f,  0.5f,   0.0f,  0.1f,  0.0f, -0.1f,  0.0f,  0.0f,  0.0f,  0.0f,
     0.4f,  0.4f,  -0.1f,  0.0f,  0.0f,  0.0f,  0.1f,  0.0f,  0.0f,  0.0f,
     0.2f,  0.6f,   0.0f, -0.1f,  0.0f,  0.0f,  0.0f,  0.1f,  0.0f,  0.0f,
     0.6f,  0.2f,   0.0f,  0.0f, -0.1f,  0.0f,  0.0f,  0.0f,  0.1f,  0.0f,
     0.3f,  0.4f,   0.1f,  0.1f,  0.0f,  0.0f, -0.1f,  0.0f,  0.0f,  0.0f,
     0.4f,  0.3f,   0.0f,  0.0f,  0.1f,  0.1f,  0.0f, -0.1f,  0.0f,  0.0f,
     0.3f,  0.5f,   0.0f,  0.0f,  0.0f,  0.0f,  0.1f,  0.0f, -0.1f,  0.0f,
};
static const float b_cand[HIDDEN_DIM] = {0};
static const float tau[HIDDEN_DIM] = {
    0.05f, 0.10f, 0.20f, 0.50f, 0.05f, 0.15f, 0.30f, 0.80f
};
static const float W_out[OUTPUT_DIM * HIDDEN_DIM] = {
    0.3f, -0.2f, 0.4f, 0.1f, -0.3f, 0.2f, 0.3f, -0.1f
};
static const float b_out[OUTPUT_DIM] = { 0.0f };

/* ── Helpers ── */

static double get_time_us(void) {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (double)tv.tv_sec * 1e6 + (double)tv.tv_usec;
}

static float sim_dt(float mean, float jitter, unsigned int *s) {
    float u = (float)rand_r(s) / (float)RAND_MAX;
    float v = (float)rand_r(s) / (float)RAND_MAX;
    float dt = mean + jitter * (u + v - 1.0f);
    return dt < 0.02f ? 0.02f : dt;
}

static float sim_key(unsigned int *s) {
    return ((float)(32 + rand_r(s) % 95) - 32.0f) / 94.0f;
}

/* Process one keystroke, returns score. distance_coeff controls the penalty. */
static float process_one(
    float key_norm, float dt,
    float *h_state, RunningStats *istats,
    const float *h_mean, const float *h_std,
    float distance_coeff
) {
    float raw[INPUT_DIM] = { key_norm, dt };
    float norm[INPUT_DIM];
    ONLINE_NORMALIZE_CHIP(raw, istats, norm, INPUT_DIM, 1e-6f, 1);

    float h_new[HIDDEN_DIM];
    CFC_CELL_GENERIC(norm, h_state, dt,
                     W_gate, b_gate, W_cand, b_cand,
                     tau, 0, INPUT_DIM, HIDDEN_DIM, h_new);
    memcpy(h_state, h_new, sizeof(float) * HIDDEN_DIM);

    float logit[OUTPUT_DIM];
    MATVEC_CHIP(h_new, W_out, b_out, logit, OUTPUT_DIM, HIDDEN_DIM);

    float dist = 0.0f;
    for (int i = 0; i < HIDDEN_DIM; i++) {
        float d = h_new[i] - h_mean[i];
        float v = h_std[i] * h_std[i];
        if (v > 1e-8f) dist += (d * d) / v;
    }
    dist /= (float)HIDDEN_DIM;

    return SIGMOID_CHIP(logit[0] - distance_coeff * dist);
}

/* Enroll: run N keystrokes, capture h_mean/h_std */
static void enroll(int n_keys, float mean_dt, float jitter, unsigned int *seed,
                   float *h_mean_out, float *h_std_out) {
    float h_state[HIDDEN_DIM] = {0};
    RunningStats istats[INPUT_DIM];
    RunningStats hstats[HIDDEN_DIM];
    for (int i = 0; i < INPUT_DIM; i++) RUNNING_STATS_INIT(&istats[i]);
    for (int i = 0; i < HIDDEN_DIM; i++) RUNNING_STATS_INIT(&hstats[i]);

    for (int k = 0; k < n_keys; k++) {
        float dt = sim_dt(mean_dt, jitter, seed);
        float key = sim_key(seed);
        float raw[INPUT_DIM] = { key, dt };
        float norm[INPUT_DIM];
        ONLINE_NORMALIZE_CHIP(raw, istats, norm, INPUT_DIM, 1e-6f, 1);

        float h_new[HIDDEN_DIM];
        CFC_CELL_GENERIC(norm, h_state, dt,
                         W_gate, b_gate, W_cand, b_cand,
                         tau, 0, INPUT_DIM, HIDDEN_DIM, h_new);
        memcpy(h_state, h_new, sizeof(float) * HIDDEN_DIM);

        for (int i = 0; i < HIDDEN_DIM; i++)
            RUNNING_STATS_UPDATE(&hstats[i], h_new[i]);
    }

    for (int i = 0; i < HIDDEN_DIM; i++) {
        h_mean_out[i] = hstats[i].mean;
        h_std_out[i] = sqrtf(RUNNING_STATS_VARIANCE(&hstats[i]) + 1e-8f);
    }
}

/* Authenticate: run N keystrokes, return avg score */
static float authenticate(int n_keys, float mean_dt, float jitter, unsigned int *seed,
                          const float *h_mean, const float *h_std,
                          float distance_coeff) {
    float h_state[HIDDEN_DIM] = {0};
    RunningStats istats[INPUT_DIM];
    for (int i = 0; i < INPUT_DIM; i++) RUNNING_STATS_INIT(&istats[i]);

    float sum = 0.0f;
    for (int k = 0; k < n_keys; k++) {
        float dt = sim_dt(mean_dt, jitter, seed);
        float key = sim_key(seed);
        float s = process_one(key, dt, h_state, istats, h_mean, h_std, distance_coeff);
        sum += s;
    }
    return sum / (float)n_keys;
}

/* ═══════════════════════════════════════════════════════════════════════════ */

int main(void) {
    printf("═══════════════════════════════════════════════════\n");
    printf("  Keystroke Biometric Probe — LMM Experiments\n");
    printf("═══════════════════════════════════════════════════\n\n");

    /* ── Experiment 1: Execution time ── */
    {
        printf("EXPERIMENT 1: Execution time per keystroke\n");
        float h_state[HIDDEN_DIM] = {0};
        RunningStats istats[INPUT_DIM];
        for (int i = 0; i < INPUT_DIM; i++) RUNNING_STATS_INIT(&istats[i]);

        float dummy_mean[HIDDEN_DIM] = {0};
        float dummy_std[HIDDEN_DIM];
        for (int i = 0; i < HIDDEN_DIM; i++) dummy_std[i] = 1.0f;

        unsigned int seed = 77;
        int N = 10000;

        /* Warmup */
        for (int k = 0; k < 100; k++) {
            float dt = sim_dt(0.15f, 0.04f, &seed);
            float key = sim_key(&seed);
            process_one(key, dt, h_state, istats, dummy_mean, dummy_std, 0.5f);
        }

        double t0 = get_time_us();
        for (int k = 0; k < N; k++) {
            float dt = sim_dt(0.15f, 0.04f, &seed);
            float key = sim_key(&seed);
            process_one(key, dt, h_state, istats, dummy_mean, dummy_std, 0.5f);
        }
        double t1 = get_time_us();
        double per_key = (t1 - t0) / (double)N;

        printf("  %d keystrokes in %.1f us\n", N, t1 - t0);
        printf("  Per keystroke: %.1f ns\n", per_key * 1000.0);
        printf("  (includes: normalize + CfC + MATVEC + sigmoid + distance)\n\n");
    }

    /* ── Experiment 2: Easy test (3x speed difference, baseline) ── */
    {
        printf("EXPERIMENT 2: Easy test (User A=0.12s, User B=0.35s)\n");
        unsigned int sa = 42, sb = 99;
        float hm[HIDDEN_DIM], hs[HIDDEN_DIM];
        enroll(80, 0.12f, 0.03f, &sa, hm, hs);

        sa = 200; sb = 300;  /* fresh seeds for auth */
        float a = authenticate(50, 0.12f, 0.03f, &sa, hm, hs, 0.5f);
        float b = authenticate(50, 0.35f, 0.10f, &sb, hm, hs, 0.5f);
        printf("  User A (enrolled): %.3f\n", a);
        printf("  User B (impostor): %.3f\n", b);
        printf("  Separation: %.3f\n\n", a - b);
    }

    /* ── Experiment 3: Harder test (same speed, different jitter) ── */
    {
        printf("EXPERIMENT 3: Same speed, different jitter (A=0.15s/0.02j, B=0.15s/0.08j)\n");
        unsigned int sa = 42;
        float hm[HIDDEN_DIM], hs[HIDDEN_DIM];
        enroll(80, 0.15f, 0.02f, &sa, hm, hs);

        unsigned int sb = 99;
        sa = 200;
        float a = authenticate(50, 0.15f, 0.02f, &sa, hm, hs, 0.5f);
        float b = authenticate(50, 0.15f, 0.08f, &sb, hm, hs, 0.5f);
        printf("  User A (enrolled): %.3f\n", a);
        printf("  User B (impostor): %.3f\n", b);
        printf("  Separation: %.3f\n\n", a - b);
    }

    /* ── Experiment 4: Hardest test (same speed, same jitter, different seed only) ── */
    {
        printf("EXPERIMENT 4: Same speed, same jitter, different random seed\n");
        printf("  (This tests if the CfC finds signal in SEQUENCE, not just statistics)\n");
        unsigned int sa = 42;
        float hm[HIDDEN_DIM], hs[HIDDEN_DIM];
        enroll(80, 0.15f, 0.04f, &sa, hm, hs);

        unsigned int sb = 99;
        sa = 200;
        float a = authenticate(50, 0.15f, 0.04f, &sa, hm, hs, 0.5f);
        float b = authenticate(50, 0.15f, 0.04f, &sb, hm, hs, 0.5f);
        printf("  User A (enrolled): %.3f\n", a);
        printf("  User B (same stats): %.3f\n", b);
        printf("  Separation: %.3f\n", a - b);
        printf("  (Expect ZERO or near-zero — untrained weights can't distinguish)\n\n");
    }

    /* ── Experiment 5: Distance penalty sweep ── */
    {
        printf("EXPERIMENT 5: Distance penalty coefficient sweep\n");
        printf("  %-12s  %-10s  %-10s  %-10s\n", "Coeff", "User A", "User B", "Sep");

        float coeffs[] = { 0.0f, 0.1f, 0.25f, 0.5f, 1.0f, 2.0f };
        int nc = sizeof(coeffs) / sizeof(coeffs[0]);

        for (int c = 0; c < nc; c++) {
            unsigned int sa = 42;
            float hm[HIDDEN_DIM], hs[HIDDEN_DIM];
            enroll(80, 0.12f, 0.03f, &sa, hm, hs);

            sa = 200;
            unsigned int sb = 300;
            float a = authenticate(50, 0.12f, 0.03f, &sa, hm, hs, coeffs[c]);
            float b = authenticate(50, 0.35f, 0.10f, &sb, hm, hs, coeffs[c]);
            printf("  %-12.2f  %-10.3f  %-10.3f  %-10.3f\n", coeffs[c], a, b, a - b);
        }
        printf("\n");
    }

    /* ── Experiment 6: Minimum enrollment length ── */
    {
        printf("EXPERIMENT 6: Minimum enrollment length for separation\n");
        printf("  %-12s  %-10s  %-10s  %-10s\n", "Enroll#", "User A", "User B", "Sep");

        int enroll_lengths[] = { 5, 10, 20, 40, 80, 160 };
        int ne = sizeof(enroll_lengths) / sizeof(enroll_lengths[0]);

        for (int e = 0; e < ne; e++) {
            unsigned int sa = 42;
            float hm[HIDDEN_DIM], hs[HIDDEN_DIM];
            enroll(enroll_lengths[e], 0.12f, 0.03f, &sa, hm, hs);

            sa = 200;
            unsigned int sb = 300;
            float a = authenticate(50, 0.12f, 0.03f, &sa, hm, hs, 0.5f);
            float b = authenticate(50, 0.35f, 0.10f, &sb, hm, hs, 0.5f);
            printf("  %-12d  %-10.3f  %-10.3f  %-10.3f\n", enroll_lengths[e], a, b, a - b);
        }
        printf("\n");
    }

    printf("═══════════════════════════════════════════════════\n");
    printf("  Probe complete.\n");
    printf("═══════════════════════════════════════════════════\n");

    return 0;
}
