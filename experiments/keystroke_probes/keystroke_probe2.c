/*
 * Keystroke Probe 2 — Template matching in hidden state space
 *
 * Tests Node 8 Option B: skip the learned projection entirely.
 * During enrollment, record the hidden state trajectory (rolling window).
 * During auth, compute cosine similarity to the enrolled trajectory.
 *
 * Also tests: with vs without Welford normalization.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "onnx_shapes.h"
#include "chips/cfc_cell_chip.h"
#include "chips/norm_chip.h"
#include "chips/activation_chip.h"

#define INPUT_DIM   2
#define HIDDEN_DIM  8
#define CONCAT_DIM  (INPUT_DIM + HIDDEN_DIM)

/* Same weights */
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

/* ── Helpers ── */

static float sim_dt(float mean, float jitter, unsigned int *s) {
    float u = (float)rand_r(s) / (float)RAND_MAX;
    float v = (float)rand_r(s) / (float)RAND_MAX;
    float dt = mean + jitter * (u + v - 1.0f);
    return dt < 0.02f ? 0.02f : dt;
}

static float sim_key(unsigned int *s) {
    return ((float)(32 + rand_r(s) % 95) - 32.0f) / 94.0f;
}

static float cosine_sim(const float *a, const float *b, int n) {
    float dot = 0, na = 0, nb = 0;
    for (int i = 0; i < n; i++) {
        dot += a[i] * b[i];
        na += a[i] * a[i];
        nb += b[i] * b[i];
    }
    float denom = sqrtf(na) * sqrtf(nb);
    return denom > 1e-10f ? dot / denom : 0.0f;
}

static float euclidean_dist(const float *a, const float *b, int n) {
    float sum = 0;
    for (int i = 0; i < n; i++) {
        float d = a[i] - b[i];
        sum += d * d;
    }
    return sqrtf(sum);
}

/* Run CfC on a keystroke sequence, return final hidden state */
static void run_sequence(
    int n_keys, float mean_dt, float jitter, unsigned int *seed,
    int use_norm,
    float *h_final  /* output: [HIDDEN_DIM] */
) {
    float h_state[HIDDEN_DIM] = {0};
    RunningStats istats[INPUT_DIM];
    for (int i = 0; i < INPUT_DIM; i++) RUNNING_STATS_INIT(&istats[i]);

    for (int k = 0; k < n_keys; k++) {
        float dt = sim_dt(mean_dt, jitter, seed);
        float key = sim_key(seed);
        float raw[INPUT_DIM] = { key, dt };
        float input[INPUT_DIM];

        if (use_norm) {
            ONLINE_NORMALIZE_CHIP(raw, istats, input, INPUT_DIM, 1e-6f, 1);
        } else {
            input[0] = raw[0];
            input[1] = raw[1];
        }

        float h_new[HIDDEN_DIM];
        CFC_CELL_GENERIC(input, h_state, dt,
                         W_gate, b_gate, W_cand, b_cand,
                         tau, 0, INPUT_DIM, HIDDEN_DIM, h_new);
        memcpy(h_state, h_new, sizeof(float) * HIDDEN_DIM);
    }
    memcpy(h_final, h_state, sizeof(float) * HIDDEN_DIM);
}

/* Run CfC, collect hidden state trajectory (last W steps) */
#define WINDOW 20
static void run_sequence_trajectory(
    int n_keys, float mean_dt, float jitter, unsigned int *seed,
    int use_norm,
    float trajectory[WINDOW][HIDDEN_DIM]  /* output */
) {
    float h_state[HIDDEN_DIM] = {0};
    RunningStats istats[INPUT_DIM];
    for (int i = 0; i < INPUT_DIM; i++) RUNNING_STATS_INIT(&istats[i]);

    int start_record = n_keys - WINDOW;
    if (start_record < 0) start_record = 0;

    for (int k = 0; k < n_keys; k++) {
        float dt = sim_dt(mean_dt, jitter, seed);
        float key = sim_key(seed);
        float raw[INPUT_DIM] = { key, dt };
        float input[INPUT_DIM];

        if (use_norm) {
            ONLINE_NORMALIZE_CHIP(raw, istats, input, INPUT_DIM, 1e-6f, 1);
        } else {
            input[0] = raw[0];
            input[1] = raw[1];
        }

        float h_new[HIDDEN_DIM];
        CFC_CELL_GENERIC(input, h_state, dt,
                         W_gate, b_gate, W_cand, b_cand,
                         tau, 0, INPUT_DIM, HIDDEN_DIM, h_new);
        memcpy(h_state, h_new, sizeof(float) * HIDDEN_DIM);

        if (k >= start_record) {
            int idx = k - start_record;
            memcpy(trajectory[idx], h_state, sizeof(float) * HIDDEN_DIM);
        }
    }
}

/* Compare two trajectories: avg cosine similarity of corresponding steps */
static float trajectory_similarity(
    float traj_a[WINDOW][HIDDEN_DIM],
    float traj_b[WINDOW][HIDDEN_DIM]
) {
    float sum = 0;
    for (int t = 0; t < WINDOW; t++) {
        sum += cosine_sim(traj_a[t], traj_b[t], HIDDEN_DIM);
    }
    return sum / (float)WINDOW;
}

/* Compare final hidden states across multiple runs */
static void multi_run_similarity(
    const char *label,
    int n_runs, int n_keys,
    float mean_dt_a, float jitter_a,
    float mean_dt_b, float jitter_b,
    int use_norm
) {
    float h_finals_a[n_runs][HIDDEN_DIM];
    float h_finals_b[n_runs][HIDDEN_DIM];

    /* Generate multiple runs of User A */
    for (int r = 0; r < n_runs; r++) {
        unsigned int seed = 1000 + r * 37;
        run_sequence(n_keys, mean_dt_a, jitter_a, &seed, use_norm, h_finals_a[r]);
    }

    /* Generate multiple runs of User B */
    for (int r = 0; r < n_runs; r++) {
        unsigned int seed = 5000 + r * 53;
        run_sequence(n_keys, mean_dt_b, jitter_b, &seed, use_norm, h_finals_b[r]);
    }

    /* Intra-class similarity (A vs A) */
    float intra_sum = 0; int intra_n = 0;
    for (int i = 0; i < n_runs; i++) {
        for (int j = i+1; j < n_runs; j++) {
            intra_sum += cosine_sim(h_finals_a[i], h_finals_a[j], HIDDEN_DIM);
            intra_n++;
        }
    }
    float intra_cos = intra_sum / (float)intra_n;

    /* Inter-class similarity (A vs B) */
    float inter_sum = 0; int inter_n = 0;
    for (int i = 0; i < n_runs; i++) {
        for (int j = 0; j < n_runs; j++) {
            inter_sum += cosine_sim(h_finals_a[i], h_finals_b[j], HIDDEN_DIM);
            inter_n++;
        }
    }
    float inter_cos = inter_sum / (float)inter_n;

    /* Also euclidean */
    float intra_euc = 0;
    for (int i = 0; i < n_runs; i++)
        for (int j = i+1; j < n_runs; j++)
            intra_euc += euclidean_dist(h_finals_a[i], h_finals_a[j], HIDDEN_DIM);
    intra_euc /= (float)intra_n;

    float inter_euc = 0;
    for (int i = 0; i < n_runs; i++)
        for (int j = 0; j < n_runs; j++)
            inter_euc += euclidean_dist(h_finals_a[i], h_finals_b[j], HIDDEN_DIM);
    inter_euc /= (float)inter_n;

    printf("  %s\n", label);
    printf("    Intra-class cosine (A vs A): %.4f\n", intra_cos);
    printf("    Inter-class cosine (A vs B): %.4f\n", inter_cos);
    printf("    Cosine separation:           %.4f\n", intra_cos - inter_cos);
    printf("    Intra-class euclid (A vs A): %.4f\n", intra_euc);
    printf("    Inter-class euclid (A vs B): %.4f\n", inter_euc);
    printf("    Euclid ratio (inter/intra):  %.2fx\n\n",
           intra_euc > 1e-8 ? inter_euc / intra_euc : 0.0f);
}

int main(void) {
    printf("═══════════════════════════════════════════════════\n");
    printf("  Keystroke Probe 2 — Hidden State Template Matching\n");
    printf("═══════════════════════════════════════════════════\n\n");

    int N_RUNS = 20;
    int N_KEYS = 80;

    /* ── Test 1: Easy case (3x speed diff), with norm ── */
    printf("TEST 1: 3x speed difference, WITH normalization\n");
    multi_run_similarity("A=0.12s, B=0.35s", N_RUNS, N_KEYS,
                         0.12f, 0.03f, 0.35f, 0.10f, 1);

    /* ── Test 2: Easy case, WITHOUT norm ── */
    printf("TEST 2: 3x speed difference, WITHOUT normalization\n");
    multi_run_similarity("A=0.12s, B=0.35s", N_RUNS, N_KEYS,
                         0.12f, 0.03f, 0.35f, 0.10f, 0);

    /* ── Test 3: Same speed, different jitter, with norm ── */
    printf("TEST 3: Same speed, different jitter, WITH normalization\n");
    multi_run_similarity("A=0.15s/0.02j, B=0.15s/0.08j", N_RUNS, N_KEYS,
                         0.15f, 0.02f, 0.15f, 0.08f, 1);

    /* ── Test 4: Same speed, different jitter, WITHOUT norm ── */
    printf("TEST 4: Same speed, different jitter, WITHOUT normalization\n");
    multi_run_similarity("A=0.15s/0.02j, B=0.15s/0.08j", N_RUNS, N_KEYS,
                         0.15f, 0.02f, 0.15f, 0.08f, 0);

    /* ── Test 5: Same everything (control — should be no separation) ── */
    printf("TEST 5: Same speed, same jitter (CONTROL — expect zero separation)\n");
    multi_run_similarity("A=0.15s/0.04j, B=0.15s/0.04j", N_RUNS, N_KEYS,
                         0.15f, 0.04f, 0.15f, 0.04f, 1);

    /* ── Test 6: Trajectory matching (easy case) ── */
    printf("TEST 6: Trajectory matching (last %d steps)\n", WINDOW);
    {
        float traj_enroll[WINDOW][HIDDEN_DIM];
        float traj_a[WINDOW][HIDDEN_DIM];
        float traj_b[WINDOW][HIDDEN_DIM];

        unsigned int s1 = 42, s2 = 200, s3 = 300;
        run_sequence_trajectory(80, 0.12f, 0.03f, &s1, 1, traj_enroll);
        run_sequence_trajectory(80, 0.12f, 0.03f, &s2, 1, traj_a);
        run_sequence_trajectory(80, 0.35f, 0.10f, &s3, 1, traj_b);

        float sim_a = trajectory_similarity(traj_enroll, traj_a);
        float sim_b = trajectory_similarity(traj_enroll, traj_b);
        printf("  Enrolled trajectory vs User A: %.4f\n", sim_a);
        printf("  Enrolled trajectory vs User B: %.4f\n", sim_b);
        printf("  Separation: %.4f\n\n", sim_a - sim_b);
    }

    /* ── Test 7: Print actual hidden states for visual inspection ── */
    printf("TEST 7: Hidden state samples (first 3 dims)\n");
    {
        int n_samples = 5;
        printf("  User A samples (0.12s mean):\n");
        for (int r = 0; r < n_samples; r++) {
            float h[HIDDEN_DIM];
            unsigned int seed = 1000 + r * 37;
            run_sequence(80, 0.12f, 0.03f, &seed, 0, h);
            printf("    run %d: [%.4f, %.4f, %.4f, %.4f, %.4f, %.4f, %.4f, %.4f]\n",
                   r, h[0], h[1], h[2], h[3], h[4], h[5], h[6], h[7]);
        }
        printf("  User B samples (0.35s mean):\n");
        for (int r = 0; r < n_samples; r++) {
            float h[HIDDEN_DIM];
            unsigned int seed = 5000 + r * 53;
            run_sequence(80, 0.35f, 0.10f, &seed, 0, h);
            printf("    run %d: [%.4f, %.4f, %.4f, %.4f, %.4f, %.4f, %.4f, %.4f]\n",
                   r, h[0], h[1], h[2], h[3], h[4], h[5], h[6], h[7]);
        }
    }

    printf("\n═══════════════════════════════════════════════════\n");
    printf("  Probe 2 complete.\n");
    printf("═══════════════════════════════════════════════════\n");
    return 0;
}
