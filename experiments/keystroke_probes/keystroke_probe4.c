/*
 * Keystroke Probe 4 — Ablation: mean-only vs PCA, PC sweep, multi-session
 *
 * Answers:
 *   1. Mean-only scoring vs PCA scoring (does PCA add signal?)
 *   2. N_PCS sweep: 0, 1, 2, 3, 5, 7
 *   3. Euclidean distance from mean vs PCA z-score
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "onnx_shapes.h"
#include "chips/cfc_cell_chip.h"
#include "chips/activation_chip.h"

#define INPUT_DIM   2
#define HIDDEN_DIM  8
#define CONCAT_DIM  (INPUT_DIM + HIDDEN_DIM)
#define MAX_SAMPLES 200
#define POWER_ITERS 20

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

static float sim_dt(float mean, float jitter, unsigned int *s) {
    float u = (float)rand_r(s) / (float)RAND_MAX;
    float v = (float)rand_r(s) / (float)RAND_MAX;
    float dt = mean + jitter * (u + v - 1.0f);
    return dt < 0.02f ? 0.02f : dt;
}

static float sim_key(unsigned int *s) {
    return ((float)(32 + rand_r(s) % 95) - 32.0f) / 94.0f;
}

static void cfc_step(float key_norm, float dt, float *h_state) {
    float input[INPUT_DIM] = { key_norm, dt };
    float h_new[HIDDEN_DIM];
    CFC_CELL_GENERIC(input, h_state, dt,
                     W_gate, b_gate, W_cand, b_cand,
                     tau, 0, INPUT_DIM, HIDDEN_DIM, h_new);
    memcpy(h_state, h_new, sizeof(float) * HIDDEN_DIM);
}

static float dotf(const float *a, const float *b, int n) {
    float s = 0;
    for (int i = 0; i < n; i++) s += a[i] * b[i];
    return s;
}

static float vec_norm(float *v, int n) {
    float s = 0;
    for (int i = 0; i < n; i++) s += v[i] * v[i];
    s = sqrtf(s);
    if (s > 1e-10f) for (int i = 0; i < n; i++) v[i] /= s;
    return s;
}

/* Collect hidden state samples from a session */
static int collect_samples(
    int n_keys, float mean_dt, float jitter, unsigned int *seed,
    float samples[][HIDDEN_DIM], int max_samples
) {
    float h_state[HIDDEN_DIM] = {0};
    /* 10 warmup steps */
    for (int k = 0; k < 10; k++)
        cfc_step(sim_key(seed), sim_dt(mean_dt, jitter, seed), h_state);

    int n = 0;
    for (int k = 0; k < n_keys && n < max_samples; k++) {
        cfc_step(sim_key(seed), sim_dt(mean_dt, jitter, seed), h_state);
        memcpy(samples[n], h_state, sizeof(float) * HIDDEN_DIM);
        n++;
    }
    return n;
}

/* Get final hidden state from a session */
static void get_final_h(
    int n_keys, float mean_dt, float jitter, unsigned int *seed,
    float *h_out
) {
    float h_state[HIDDEN_DIM] = {0};
    for (int k = 0; k < 10; k++)
        cfc_step(sim_key(seed), sim_dt(mean_dt, jitter, seed), h_state);
    for (int k = 0; k < n_keys; k++)
        cfc_step(sim_key(seed), sim_dt(mean_dt, jitter, seed), h_state);
    memcpy(h_out, h_state, sizeof(float) * HIDDEN_DIM);
}

/* ═══════════════════════════════════════════════════════════════════════════ */

/* Score using MEAN ONLY: euclidean distance from enrollment mean */
static float score_mean_only(const float *h, const float *mean, const float *std_per_dim) {
    float dist = 0;
    for (int i = 0; i < HIDDEN_DIM; i++) {
        float z = (h[i] - mean[i]) / (std_per_dim[i] + 1e-8f);
        dist += z * z;
    }
    dist /= HIDDEN_DIM;
    return 1.0f / (1.0f + expf(-(2.0f - dist)));
}

/* Score using PCA: mean + N principal components */
static float score_pca(
    const float *h, const float *mean,
    const float pcs[][HIDDEN_DIM], const float *proj_mean, const float *proj_std,
    int n_pcs
) {
    float centered[HIDDEN_DIM];
    for (int i = 0; i < HIDDEN_DIM; i++)
        centered[i] = h[i] - mean[i];

    float dist = 0;
    for (int pc = 0; pc < n_pcs; pc++) {
        float proj = dotf(centered, pcs[pc], HIDDEN_DIM);
        float z = (proj - proj_mean[pc]) / (proj_std[pc] + 1e-8f);
        dist += z * z;
    }
    dist /= (n_pcs > 0 ? n_pcs : 1);
    return 1.0f / (1.0f + expf(-(2.0f - dist)));
}

/* Learn enrollment stats + PCs (up to max_pcs) */
static void learn_enrollment(
    const float samples[][HIDDEN_DIM], int n_samples,
    float *mean_out, float *std_out,
    float pcs_out[][HIDDEN_DIM], float *proj_mean_out, float *proj_std_out,
    int max_pcs
) {
    /* Mean */
    for (int i = 0; i < HIDDEN_DIM; i++) {
        float s = 0;
        for (int t = 0; t < n_samples; t++) s += samples[t][i];
        mean_out[i] = s / n_samples;
    }

    /* Per-dim std */
    for (int i = 0; i < HIDDEN_DIM; i++) {
        float s = 0;
        for (int t = 0; t < n_samples; t++) {
            float d = samples[t][i] - mean_out[i];
            s += d * d;
        }
        std_out[i] = sqrtf(s / n_samples + 1e-8f);
    }

    /* Center */
    float centered[n_samples][HIDDEN_DIM];
    for (int t = 0; t < n_samples; t++)
        for (int i = 0; i < HIDDEN_DIM; i++)
            centered[t][i] = samples[t][i] - mean_out[i];

    /* PCs via power iteration with deflation */
    for (int pc = 0; pc < max_pcs; pc++) {
        float v[HIDDEN_DIM];
        for (int i = 0; i < HIDDEN_DIM; i++)
            v[i] = centered[0][i] + 0.01f * (i + 1);
        vec_norm(v, HIDDEN_DIM);

        for (int iter = 0; iter < POWER_ITERS; iter++) {
            float v_new[HIDDEN_DIM] = {0};
            for (int t = 0; t < n_samples; t++) {
                float p = dotf(centered[t], v, HIDDEN_DIM);
                for (int i = 0; i < HIDDEN_DIM; i++)
                    v_new[i] += p * centered[t][i];
            }
            for (int i = 0; i < HIDDEN_DIM; i++) v_new[i] /= n_samples;
            memcpy(v, v_new, sizeof(v));
            vec_norm(v, HIDDEN_DIM);
        }

        memcpy(pcs_out[pc], v, sizeof(float) * HIDDEN_DIM);

        float ps = 0, ps2 = 0;
        for (int t = 0; t < n_samples; t++) {
            float p = dotf(centered[t], v, HIDDEN_DIM);
            ps += p; ps2 += p * p;
        }
        proj_mean_out[pc] = ps / n_samples;
        float var = ps2 / n_samples - proj_mean_out[pc] * proj_mean_out[pc];
        proj_std_out[pc] = sqrtf(var > 0 ? var : 1e-8f);

        /* Deflate */
        for (int t = 0; t < n_samples; t++) {
            float p = dotf(centered[t], v, HIDDEN_DIM);
            for (int i = 0; i < HIDDEN_DIM; i++)
                centered[t][i] -= p * v[i];
        }
    }
}

/* Run a test at a given difficulty, compare mean-only vs PCA at various N_PCS */
static void ablation_test(
    const char *label,
    float a_speed, float a_jitter,
    float b_speed, float b_jitter,
    int n_runs
) {
    printf("  %s\n", label);
    printf("    %-12s  %-10s  %-10s  %-10s  %-8s\n",
           "Method", "Avg A", "Avg B", "Sep", "A wins");

    int pc_counts[] = { 0, 1, 2, 3, 5, 7 };
    int n_pc_tests = sizeof(pc_counts) / sizeof(pc_counts[0]);

    for (int pi = 0; pi < n_pc_tests; pi++) {
        int npcs = pc_counts[pi];
        float sum_a = 0, sum_b = 0;
        int a_wins = 0;

        for (int r = 0; r < n_runs; r++) {
            /* Enroll */
            unsigned int enroll_seed = 100 + r * 31;
            float samples[MAX_SAMPLES][HIDDEN_DIM];
            int ns = collect_samples(80, a_speed, a_jitter, &enroll_seed,
                                     samples, MAX_SAMPLES);

            float mean[HIDDEN_DIM], std[HIDDEN_DIM];
            float pcs[7][HIDDEN_DIM], pm[7], ps[7];
            learn_enrollment(samples, ns, mean, std, pcs, pm, ps, npcs > 0 ? npcs : 1);

            /* Auth A */
            unsigned int sa = 3000 + r * 47;
            float ha[HIDDEN_DIM];
            get_final_h(50, a_speed, a_jitter, &sa, ha);

            /* Auth B */
            unsigned int sb = 7000 + r * 67;
            float hb[HIDDEN_DIM];
            get_final_h(50, b_speed, b_jitter, &sb, hb);

            float score_a, score_b;
            if (npcs == 0) {
                score_a = score_mean_only(ha, mean, std);
                score_b = score_mean_only(hb, mean, std);
            } else {
                score_a = score_pca(ha, mean, pcs, pm, ps, npcs);
                score_b = score_pca(hb, mean, pcs, pm, ps, npcs);
            }

            sum_a += score_a;
            sum_b += score_b;
            if (score_a > score_b) a_wins++;
        }

        float avg_a = sum_a / n_runs, avg_b = sum_b / n_runs;
        char method[32];
        if (npcs == 0) sprintf(method, "mean-only");
        else sprintf(method, "PCA(%d)", npcs);

        printf("    %-12s  %-10.3f  %-10.3f  %+-10.3f  %d/%d\n",
               method, avg_a, avg_b, avg_a - avg_b, a_wins, n_runs);
    }
    printf("\n");
}

int main(void) {
    printf("═══════════════════════════════════════════════════\n");
    printf("  Probe 4 — Ablation: Mean vs PCA, PC count sweep\n");
    printf("═══════════════════════════════════════════════════\n\n");

    int N = 20;

    ablation_test("TEST 1: Easy (A=0.12s, B=0.35s)",
                  0.12f, 0.03f, 0.35f, 0.10f, N);

    ablation_test("TEST 2: Medium (A=0.12s, B=0.18s)",
                  0.12f, 0.03f, 0.18f, 0.045f, N);

    ablation_test("TEST 3: Hard (A=0.15s/0.02j, B=0.15s/0.08j)",
                  0.15f, 0.02f, 0.15f, 0.08f, N);

    ablation_test("TEST 4: Control (A=B=0.15s/0.04j)",
                  0.15f, 0.04f, 0.15f, 0.04f, N);

    printf("═══════════════════════════════════════════════════\n");
    return 0;
}
