/*
 * Keystroke Probe 3 — On-device linear discriminant from enrollment
 *
 * Tests the hypothesis: can we learn a readout during enrollment that
 * separates users, without backpropagation through the CfC?
 *
 * Approach:
 *   1. During enrollment, collect hidden state vectors h[t] from CfC
 *   2. Compute mean and covariance of enrolled hidden states
 *   3. Extract principal direction via power iteration
 *   4. For auth: project live hidden state onto enrolled principal subspace
 *   5. Score = how close the projection is to the enrolled distribution
 *
 * The idea: the enrolled user's hidden states occupy a specific region of
 * hidden space. The principal direction captures "the direction this user's
 * hidden states vary most." Projecting onto it gives a 1D fingerprint.
 * An impostor's hidden states project differently because they occupy a
 * different region.
 *
 * Compile:
 *   cc -O2 -I../include -I../include/chips keystroke_probe3.c -lm -o keystroke_probe3
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

/* Same weights as main demo */
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

static void cfc_step(float key_norm, float dt, float *h_state) {
    float input[INPUT_DIM] = { key_norm, dt };
    float h_new[HIDDEN_DIM];
    CFC_CELL_GENERIC(input, h_state, dt,
                     W_gate, b_gate, W_cand, b_cand,
                     tau, 0, INPUT_DIM, HIDDEN_DIM, h_new);
    memcpy(h_state, h_new, sizeof(float) * HIDDEN_DIM);
}

/* ═══════════════════════════════════════════════════════════════════════════
 * Linear Discriminant learned from enrollment
 *
 * During enrollment, we collect hidden state samples. From these we learn:
 *   - mean: center of the enrolled distribution
 *   - principal components (top K via power iteration): the directions
 *     of variation in the enrolled user's hidden states
 *   - projection stats: mean and std of projection onto each PC
 *
 * At auth time, we:
 *   - Project the live hidden state onto the enrolled PCs
 *   - Compare the projection to the enrolled distribution
 *   - Score based on Mahalanobis-like distance in the PC subspace
 *
 * This is a one-shot linear discriminant. No gradients. No optimizer.
 * Just linear algebra on the hidden states the CfC already produces.
 * ═══════════════════════════════════════════════════════════════════════════ */

#define MAX_SAMPLES 200
#define N_PCS       3    /* Number of principal components to extract */
#define POWER_ITERS 20   /* Iterations for power method */

typedef struct {
    /* Learned from enrollment */
    float mean[HIDDEN_DIM];
    float pcs[N_PCS][HIDDEN_DIM];     /* Principal components */
    float proj_mean[N_PCS];            /* Mean projection onto each PC */
    float proj_std[N_PCS];             /* Std of projection onto each PC */
    int valid;
} LinearDiscriminant;

/**
 * Dot product of two vectors
 */
static float dot(const float *a, const float *b, int n) {
    float s = 0;
    for (int i = 0; i < n; i++) s += a[i] * b[i];
    return s;
}

/**
 * Normalize a vector in place. Returns the norm.
 */
static float vec_normalize(float *v, int n) {
    float norm = 0;
    for (int i = 0; i < n; i++) norm += v[i] * v[i];
    norm = sqrtf(norm);
    if (norm > 1e-10f) {
        for (int i = 0; i < n; i++) v[i] /= norm;
    }
    return norm;
}

/**
 * Learn the linear discriminant from enrollment hidden state samples.
 *
 * @param samples   Hidden state samples [n_samples][HIDDEN_DIM]
 * @param n_samples Number of samples
 * @param ld        Output: learned discriminant
 */
static void learn_discriminant(
    const float samples[][HIDDEN_DIM],
    int n_samples,
    LinearDiscriminant *ld
) {
    memset(ld, 0, sizeof(*ld));

    if (n_samples < 5) { ld->valid = 0; return; }

    /* Step 1: Compute mean */
    for (int i = 0; i < HIDDEN_DIM; i++) {
        float s = 0;
        for (int t = 0; t < n_samples; t++) s += samples[t][i];
        ld->mean[i] = s / n_samples;
    }

    /* Step 2: Center the samples (in-place copy) */
    float centered[n_samples][HIDDEN_DIM];
    for (int t = 0; t < n_samples; t++)
        for (int i = 0; i < HIDDEN_DIM; i++)
            centered[t][i] = samples[t][i] - ld->mean[i];

    /* Step 3: Extract PCs via power iteration with deflation */
    for (int pc = 0; pc < N_PCS; pc++) {
        /* Initialize with first centered sample */
        float v[HIDDEN_DIM];
        for (int i = 0; i < HIDDEN_DIM; i++)
            v[i] = centered[0][i] + 0.01f * (i + 1);  /* break symmetry */
        vec_normalize(v, HIDDEN_DIM);

        for (int iter = 0; iter < POWER_ITERS; iter++) {
            /* v_new = C @ v, where C = (1/n) * X^T X (covariance) */
            float v_new[HIDDEN_DIM] = {0};
            for (int t = 0; t < n_samples; t++) {
                float proj = dot(centered[t], v, HIDDEN_DIM);
                for (int i = 0; i < HIDDEN_DIM; i++)
                    v_new[i] += proj * centered[t][i];
            }
            for (int i = 0; i < HIDDEN_DIM; i++)
                v_new[i] /= n_samples;

            memcpy(v, v_new, sizeof(v));
            vec_normalize(v, HIDDEN_DIM);
        }

        memcpy(ld->pcs[pc], v, sizeof(float) * HIDDEN_DIM);

        /* Compute projection statistics for this PC */
        float proj_sum = 0, proj_sum2 = 0;
        for (int t = 0; t < n_samples; t++) {
            float p = dot(centered[t], v, HIDDEN_DIM);
            proj_sum += p;
            proj_sum2 += p * p;
        }
        ld->proj_mean[pc] = proj_sum / n_samples;
        float var = proj_sum2 / n_samples - ld->proj_mean[pc] * ld->proj_mean[pc];
        ld->proj_std[pc] = sqrtf(var > 0 ? var : 1e-8f);

        /* Deflate: remove this PC from centered data */
        for (int t = 0; t < n_samples; t++) {
            float p = dot(centered[t], v, HIDDEN_DIM);
            for (int i = 0; i < HIDDEN_DIM; i++)
                centered[t][i] -= p * v[i];
        }
    }

    ld->valid = 1;
}

/**
 * Score a hidden state against the learned discriminant.
 *
 * Returns a score in [0, 1] where higher = more similar to enrollment.
 */
static float score_discriminant(
    const float *h_state,
    const LinearDiscriminant *ld
) {
    if (!ld->valid) return 0.5f;

    /* Center the hidden state */
    float centered[HIDDEN_DIM];
    for (int i = 0; i < HIDDEN_DIM; i++)
        centered[i] = h_state[i] - ld->mean[i];

    /* Project onto each PC and compute normalized distance */
    float total_dist = 0;
    for (int pc = 0; pc < N_PCS; pc++) {
        float proj = dot(centered, ld->pcs[pc], HIDDEN_DIM);
        float z = (proj - ld->proj_mean[pc]) / (ld->proj_std[pc] + 1e-8f);
        total_dist += z * z;
    }
    total_dist /= N_PCS;  /* Average squared z-score */

    /* Map distance to score: small distance = high score.
     * dist=0 -> score~1, dist=4 -> score~0.5, dist=16 -> score~0 */
    return SIGMOID_CHIP(2.0f - total_dist);
}

/* ═══════════════════════════════════════════════════════════════════════════
 * Enrollment: run CfC, collect hidden states, learn discriminant
 * ═══════════════════════════════════════════════════════════════════════════ */

typedef struct {
    LinearDiscriminant ld;
    int n_enrollment_keys;
} EnrolledModel;

static void enroll_user(
    int n_keys, float mean_dt, float jitter, unsigned int *seed,
    EnrolledModel *model
) {
    float h_state[HIDDEN_DIM] = {0};
    float samples[MAX_SAMPLES][HIDDEN_DIM];
    int n_samples = 0;

    /* Warmup: skip first 10 keystrokes */
    for (int k = 0; k < 10; k++) {
        cfc_step(sim_key(seed), sim_dt(mean_dt, jitter, seed), h_state);
    }

    /* Collect hidden state samples */
    for (int k = 0; k < n_keys && n_samples < MAX_SAMPLES; k++) {
        cfc_step(sim_key(seed), sim_dt(mean_dt, jitter, seed), h_state);
        memcpy(samples[n_samples], h_state, sizeof(float) * HIDDEN_DIM);
        n_samples++;
    }

    learn_discriminant(samples, n_samples, &model->ld);
    model->n_enrollment_keys = n_keys + 10;
}

/**
 * Authenticate: run CfC on test keystrokes, score against enrolled model.
 * Returns average score over last (n_keys - warmup) keystrokes.
 */
static float authenticate_user(
    int n_keys, float mean_dt, float jitter, unsigned int *seed,
    const EnrolledModel *model
) {
    float h_state[HIDDEN_DIM] = {0};
    int warmup = 10;

    /* Warmup */
    for (int k = 0; k < warmup; k++)
        cfc_step(sim_key(seed), sim_dt(mean_dt, jitter, seed), h_state);

    /* Score */
    float sum = 0;
    for (int k = 0; k < n_keys; k++) {
        cfc_step(sim_key(seed), sim_dt(mean_dt, jitter, seed), h_state);
        sum += score_discriminant(h_state, &model->ld);
    }
    return sum / n_keys;
}

/* ═══════════════════════════════════════════════════════════════════════════
 * Experiments
 * ═══════════════════════════════════════════════════════════════════════════ */

static void run_test(
    const char *label,
    float a_speed, float a_jitter,
    float b_speed, float b_jitter,
    int n_runs
) {
    printf("  %s\n", label);
    printf("    %-6s  %-10s  %-10s  %-10s\n", "Run", "User A", "User B", "Sep");

    float sum_a = 0, sum_b = 0;
    int a_wins = 0;

    for (int r = 0; r < n_runs; r++) {
        /* Enroll User A with a unique seed */
        unsigned int enroll_seed = 100 + r * 31;
        EnrolledModel model;
        memset(&model, 0, sizeof(model));
        enroll_user(80, a_speed, a_jitter, &enroll_seed, &model);

        /* Auth User A (independent seed) */
        unsigned int auth_a_seed = 3000 + r * 47;
        float score_a = authenticate_user(50, a_speed, a_jitter, &auth_a_seed, &model);

        /* Auth User B (independent seed) */
        unsigned int auth_b_seed = 7000 + r * 67;
        float score_b = authenticate_user(50, b_speed, b_jitter, &auth_b_seed, &model);

        sum_a += score_a;
        sum_b += score_b;
        if (score_a > score_b) a_wins++;

        printf("    %-6d  %-10.3f  %-10.3f  %+.3f\n",
               r + 1, score_a, score_b, score_a - score_b);
    }

    float avg_a = sum_a / n_runs, avg_b = sum_b / n_runs;
    printf("    ────────────────────────────────────\n");
    printf("    Avg:    %-10.3f  %-10.3f  %+.3f\n", avg_a, avg_b, avg_a - avg_b);
    printf("    A wins: %d/%d (%.0f%%)\n\n", a_wins, n_runs,
           100.0f * a_wins / n_runs);
}

int main(void) {
    printf("═══════════════════════════════════════════════════\n");
    printf("  Probe 3 — On-device Linear Discriminant\n");
    printf("═══════════════════════════════════════════════════\n\n");
    printf("  Approach: learn a PCA-based readout during enrollment.\n");
    printf("  No backprop. No optimizer. Just linear algebra on\n");
    printf("  the hidden states the CfC already produces.\n\n");

    int N = 20;

    /* Easy: 3x speed difference */
    run_test("TEST 1: Easy — 3x speed difference (A=0.12s, B=0.35s)",
             0.12f, 0.03f, 0.35f, 0.10f, N);

    /* Medium: 1.5x speed difference */
    run_test("TEST 2: Medium — 1.5x speed difference (A=0.12s, B=0.18s)",
             0.12f, 0.03f, 0.18f, 0.045f, N);

    /* Hard: same speed, different jitter */
    run_test("TEST 3: Hard — same speed, different jitter (A=0.15s/0.02j, B=0.15s/0.08j)",
             0.15f, 0.02f, 0.15f, 0.08f, N);

    /* Hardest: same speed, same jitter, just different random sequences */
    run_test("TEST 4: Hardest — same everything (A=0.15s/0.04j, B=0.15s/0.04j)",
             0.15f, 0.04f, 0.15f, 0.04f, N);

    /* Control: A vs A (should score similarly) */
    run_test("TEST 5: Control — A vs A (should be ~equal)",
             0.15f, 0.04f, 0.15f, 0.04f, N);

    printf("═══════════════════════════════════════════════════\n");
    printf("  Discriminant size: %lu bytes\n",
           sizeof(LinearDiscriminant));
    printf("  (mean: %lu, PCs: %lu, stats: %lu)\n",
           sizeof(float) * HIDDEN_DIM,
           sizeof(float) * HIDDEN_DIM * N_PCS,
           sizeof(float) * N_PCS * 2);
    printf("═══════════════════════════════════════════════════\n");

    return 0;
}
