/*
 * Keystroke Biometric — Yinsen Chip Stack Demo (v3)
 *
 * Demonstrates the CfC chip producing consistent temporal representations
 * of typing patterns, with a hybrid linear discriminant learned from
 * enrollment data alone — no backpropagation, no optimizer.
 *
 * Pipeline (v3 — hybrid linear discriminant):
 *   Enrollment: keystroke -> CfC -> collect hidden states -> learn mean + PCA
 *   Auth:       keystroke -> CfC -> hybrid score (mean distance + PCA distance)
 *
 * What changed from v2:
 *   - Replaced cosine template matching with hybrid linear discriminant
 *   - Enrollment now learns mean vector + 5 principal components via power iteration
 *   - Auth scoring combines mean distance (position) and PCA distance (trajectory shape)
 *   - Gaussian falloff for natural distance-to-score mapping
 *   - Discriminant: 268 bytes (mean + dim_std + 5 PCs + projection stats)
 *
 * What survived from v2 (confirmed by probes 1-4):
 *   - No Welford normalization in CfC input path (destroys consistency)
 *   - Independent random seeds in simulation (no seed artifacts)
 *   - Honest reporting of separation/non-separation
 *
 * Falsification record:
 *   - probe1: Proved v1's seed correlation artifact, distance penalty inversion
 *   - probe2: Proved Welford destroys hidden state consistency (0.00 vs 0.90 cosine)
 *   - probe3: Discovered linear discriminant (100% easy, 75% hard, 156 bytes)
 *   - probe4: Ablation showed mean does heavy lifting, PCA(5) is sweet spot
 *   - LMM: 4-phase analysis in journal/scratchpad/keystroke_lda_*.md
 *
 * Key numbers:
 *   - CfC hidden state consistency: 0.897 intra-class cosine (no norm)
 *   - Discriminant: 268 bytes (mean:32 + dim_std:32 + PCs:160 + stats:40 + flag:4)
 *   - Enrollment: ~76,800 FLOPs (under 1ms)
 *   - Execution: <200 ns per keystroke
 *
 * Chips used:
 *   - cfc_cell_chip.h    CfC temporal dynamics (hidden_dim=8)
 *   - norm_chip.h        Drift monitoring only (not in CfC input path)
 *   - activation_chip.h  Sigmoid for score mapping
 *
 * Compile:
 *   cc -O2 -I../include -I../include/chips keystroke_biometric.c -lm -o keystroke_biometric
 *   or from project root:
 *   cc -O2 -I include -I include/chips examples/keystroke_biometric.c -lm -o examples/keystroke_biometric
 *
 * Created by: Tripp + Manus
 * Date: February 2026
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* Platform-specific raw keyboard input */
#if defined(__APPLE__) || defined(__linux__)
#include <termios.h>
#include <unistd.h>
#include <sys/time.h>
#define HAS_RAW_INPUT 1
#else
#define HAS_RAW_INPUT 0
#endif

/* Chip includes */
#include "onnx_shapes.h"
#include "chips/cfc_cell_chip.h"
#include "chips/norm_chip.h"
#include "chips/activation_chip.h"

/* ═══════════════════════════════════════════════════════════════════════════
 * Network Dimensions
 *
 * input_dim=2:  (key_code normalized, dt)
 * hidden_dim=8: CfC hidden state
 * ═══════════════════════════════════════════════════════════════════════════ */
#define INPUT_DIM    2
#define HIDDEN_DIM   8
#define CONCAT_DIM   (INPUT_DIM + HIDDEN_DIM)  /* 10 */

/* ═══════════════════════════════════════════════════════════════════════════
 * Discriminant Parameters (from LMM synthesis)
 *
 * N_PCS=5: sweet spot for 8-dim hidden state (probe4 ablation)
 *   PCA(3): 12/20 hard. PCA(5): 17/20 hard. PCA(7): 14/20 hard.
 * WARMUP=10: sufficient for slowest tau (0.80). 85% initial state decay.
 * MAX_ENROLL_SAMPLES=200: more than enough for 80-keystroke enrollment.
 * POWER_ITERS=20: converged well before this in all probes.
 * ═══════════════════════════════════════════════════════════════════════════ */
#define KS_N_PCS           5
#define KS_WARMUP          10
#define KS_MAX_SAMPLES     200
#define KS_POWER_ITERS     20
#define KS_ENROLL_KEYS     80    /* recommended enrollment length */
#define KS_MIN_ENROLL      20    /* minimum for early ESC */

/* Hybrid score weights.
 * PCA is the stronger signal across all difficulty levels (probe4 ablation).
 * Mean adds a floor on easy/medium cases but hurts on the hard case.
 * Weight PCA higher: the PCA subspace implicitly captures mean position
 * when N_PCS is high enough (5 of 8 dims). */
#define KS_MEAN_WEIGHT     0.3f
#define KS_PCA_WEIGHT      0.7f

/* ═══════════════════════════════════════════════════════════════════════════
 * Pre-initialized Weights
 *
 * Structurally meaningful initialization designed to create reasonable CfC
 * dynamics for keystroke timing patterns. NOT trained — the CfC encodes
 * timing information but not yet discriminative features.
 *
 * In production: replace with Python-trained ternary weights.
 * ═══════════════════════════════════════════════════════════════════════════ */

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

/* Time constants: mixed fast/slow for multi-timescale sensitivity. */
static const float tau[HIDDEN_DIM] = {
    0.05f, 0.10f, 0.20f, 0.50f, 0.05f, 0.15f, 0.30f, 0.80f
};

/* ═══════════════════════════════════════════════════════════════════════════
 * Keystroke Discriminant — Hybrid Linear Discriminant
 *
 * Learned from enrollment hidden states. Two orthogonal signals:
 *   1. Mean distance: WHERE in hidden space the user lives (position)
 *   2. PCA distance: HOW the user moves through hidden space (trajectory shape)
 *
 * Combined via weighted Gaussian falloff. 240 bytes total.
 * ═══════════════════════════════════════════════════════════════════════════ */

typedef struct {
    float mean[HIDDEN_DIM];                     /* 32 bytes — enrollment centroid */
    float dim_std[HIDDEN_DIM];                  /* 32 bytes — per-dim std for mean scoring */
    float pcs[KS_N_PCS][HIDDEN_DIM];            /* 160 bytes — principal components */
    float pc_mean[KS_N_PCS];                    /* 20 bytes — mean projection per PC */
    float pc_std[KS_N_PCS];                     /* 20 bytes — std projection per PC */
    int valid;                                   /* 4 bytes — enrollment complete flag */
} KeystrokeDiscriminant;  /* Total: 268 bytes */

/* ═══════════════════════════════════════════════════════════════════════════
 * Drift Detection — monitoring only, NOT in scoring path
 *
 * Tracks input distribution for anomaly detection. The Welford stats
 * are NEVER used to normalize CfC inputs (probe2 proved that destroys
 * hidden state consistency: 0.00 vs 0.90 cosine).
 * ═══════════════════════════════════════════════════════════════════════════ */

typedef struct {
    RunningStats input_stats[INPUT_DIM];
    float enrollment_mean[INPUT_DIM];
    float enrollment_var[INPUT_DIM];
    int warmup_done;
} DriftDetector;

/* ═══════════════════════════════════════════════════════════════════════════
 * Math Helpers
 * ═══════════════════════════════════════════════════════════════════════════ */

static float dotf(const float *a, const float *b, int n) {
    float s = 0;
    for (int i = 0; i < n; i++) s += a[i] * b[i];
    return s;
}

static float vec_normalize(float *v, int n) {
    float norm = 0;
    for (int i = 0; i < n; i++) norm += v[i] * v[i];
    norm = sqrtf(norm);
    if (norm > 1e-10f) {
        for (int i = 0; i < n; i++) v[i] /= norm;
    }
    return norm;
}

/* ═══════════════════════════════════════════════════════════════════════════
 * Core Functions
 * ═══════════════════════════════════════════════════════════════════════════ */

static void drift_init(DriftDetector *d) {
    memset(d, 0, sizeof(*d));
    for (int i = 0; i < INPUT_DIM; i++)
        RUNNING_STATS_INIT(&d->input_stats[i]);
}

/**
 * Step the CfC cell on one keystroke. No normalization — raw input directly.
 * Returns the new hidden state in h_state (updated in place).
 */
static void cfc_step(
    float key_normalized,
    float dt,
    float *h_state
) {
    float input[INPUT_DIM] = { key_normalized, dt };
    float h_new[HIDDEN_DIM];

    CFC_CELL_GENERIC(
        input, h_state, dt,
        W_gate, b_gate, W_cand, b_cand,
        tau, 0,  /* per-neuron time constants */
        INPUT_DIM, HIDDEN_DIM,
        h_new
    );

    memcpy(h_state, h_new, HIDDEN_DIM * sizeof(float));
}

/**
 * Learn the discriminant from enrollment hidden state samples.
 *
 * Steps:
 *   1. Compute mean vector (enrollment centroid)
 *   2. Compute enrollment radius (mean distance from centroid)
 *   3. Center samples
 *   4. Extract top 5 PCs via power iteration with deflation
 *   5. Compute projection statistics (mean, std) for each PC
 *
 * @param samples   Hidden state samples [n_samples][HIDDEN_DIM]
 * @param n_samples Number of samples (minimum 5)
 * @param disc      Output: learned discriminant
 */
static void learn_discriminant(
    const float samples[][HIDDEN_DIM],
    int n_samples,
    KeystrokeDiscriminant *disc
) {
    memset(disc, 0, sizeof(*disc));

    if (n_samples < 5) { disc->valid = 0; return; }

    /* Step 1: Compute mean */
    for (int i = 0; i < HIDDEN_DIM; i++) {
        float s = 0;
        for (int t = 0; t < n_samples; t++) s += samples[t][i];
        disc->mean[i] = s / n_samples;
    }

    /* Step 2: Compute per-dimension std (for mean scoring z-scores) */
    for (int i = 0; i < HIDDEN_DIM; i++) {
        float s = 0;
        for (int t = 0; t < n_samples; t++) {
            float d = samples[t][i] - disc->mean[i];
            s += d * d;
        }
        disc->dim_std[i] = sqrtf(s / n_samples + 1e-8f);
    }

    /* Step 3: Center the samples */
    float centered[n_samples][HIDDEN_DIM];
    for (int t = 0; t < n_samples; t++)
        for (int i = 0; i < HIDDEN_DIM; i++)
            centered[t][i] = samples[t][i] - disc->mean[i];

    /* Step 4: Extract PCs via power iteration with deflation */
    for (int pc = 0; pc < KS_N_PCS; pc++) {
        /* Initialize with first centered sample + symmetry breaker */
        float v[HIDDEN_DIM];
        for (int i = 0; i < HIDDEN_DIM; i++)
            v[i] = centered[0][i] + 0.01f * (i + 1);
        vec_normalize(v, HIDDEN_DIM);

        for (int iter = 0; iter < KS_POWER_ITERS; iter++) {
            /* v_new = C @ v, where C = (1/n) * X^T X */
            float v_new[HIDDEN_DIM] = {0};
            for (int t = 0; t < n_samples; t++) {
                float proj = dotf(centered[t], v, HIDDEN_DIM);
                for (int i = 0; i < HIDDEN_DIM; i++)
                    v_new[i] += proj * centered[t][i];
            }
            for (int i = 0; i < HIDDEN_DIM; i++)
                v_new[i] /= n_samples;

            memcpy(v, v_new, sizeof(v));
            vec_normalize(v, HIDDEN_DIM);
        }

        memcpy(disc->pcs[pc], v, sizeof(float) * HIDDEN_DIM);

        /* Step 5: Compute projection statistics for this PC */
        float proj_sum = 0, proj_sum2 = 0;
        for (int t = 0; t < n_samples; t++) {
            float p = dotf(centered[t], v, HIDDEN_DIM);
            proj_sum += p;
            proj_sum2 += p * p;
        }
        disc->pc_mean[pc] = proj_sum / n_samples;
        float var = proj_sum2 / n_samples - disc->pc_mean[pc] * disc->pc_mean[pc];
        disc->pc_std[pc] = sqrtf(var > 0 ? var : 1e-8f);

        /* Deflate: remove this PC's contribution from centered data */
        for (int t = 0; t < n_samples; t++) {
            float p = dotf(centered[t], v, HIDDEN_DIM);
            for (int i = 0; i < HIDDEN_DIM; i++)
                centered[t][i] -= p * v[i];
        }
    }

    disc->valid = 1;
}

/**
 * Score a hidden state against the enrolled discriminant.
 *
 * Both components use the same proven scoring function from probe4:
 *   score = sigmoid(2 - avg_squared_z)
 *
 * Mean component:  per-dimension z-score (position in hidden space)
 * PCA component:   per-PC z-score (trajectory shape)
 * Hybrid:          weighted combination
 *
 * @param h_state  Current hidden state
 * @param disc     Enrolled discriminant
 * @return         Score in [0, 1], higher = more similar to enrollment
 */
static float score_hybrid(
    const float *h_state,
    const KeystrokeDiscriminant *disc
) {
    if (!disc->valid) return 0.5f;

    /* ── Mean distance component (per-dim z-score, same as probe4) ── */
    float mean_dist = 0;
    float centered[HIDDEN_DIM];
    for (int i = 0; i < HIDDEN_DIM; i++) {
        centered[i] = h_state[i] - disc->mean[i];
        float z = centered[i] / (disc->dim_std[i] + 1e-8f);
        mean_dist += z * z;
    }
    mean_dist /= HIDDEN_DIM;
    float mean_score = SIGMOID_CHIP(2.0f - mean_dist);

    /* ── PCA distance component (per-PC z-score, same as probe4) ── */
    float pca_dist = 0;
    for (int pc = 0; pc < KS_N_PCS; pc++) {
        float proj = dotf(centered, disc->pcs[pc], HIDDEN_DIM);
        float z = (proj - disc->pc_mean[pc]) / (disc->pc_std[pc] + 1e-8f);
        pca_dist += z * z;
    }
    pca_dist /= KS_N_PCS;
    float pca_score = SIGMOID_CHIP(2.0f - pca_dist);

    /* ── Hybrid ── */
    return KS_MEAN_WEIGHT * mean_score + KS_PCA_WEIGHT * pca_score;
}

/**
 * Score using mean-only (for comparison in simulation).
 * Per-dimension z-score with sigmoid mapping — identical to probe4.
 */
static float score_mean_only(
    const float *h_state,
    const KeystrokeDiscriminant *disc
) {
    if (!disc->valid) return 0.5f;

    float dist = 0;
    for (int i = 0; i < HIDDEN_DIM; i++) {
        float z = (h_state[i] - disc->mean[i]) / (disc->dim_std[i] + 1e-8f);
        dist += z * z;
    }
    dist /= HIDDEN_DIM;
    return SIGMOID_CHIP(2.0f - dist);
}

/**
 * Score using PCA-only (for comparison in simulation).
 * Per-PC z-score with sigmoid mapping — identical to probe4.
 */
static float score_pca_only(
    const float *h_state,
    const KeystrokeDiscriminant *disc
) {
    if (!disc->valid) return 0.5f;

    float centered[HIDDEN_DIM];
    for (int i = 0; i < HIDDEN_DIM; i++)
        centered[i] = h_state[i] - disc->mean[i];

    float dist = 0;
    for (int pc = 0; pc < KS_N_PCS; pc++) {
        float proj = dotf(centered, disc->pcs[pc], HIDDEN_DIM);
        float z = (proj - disc->pc_mean[pc]) / (disc->pc_std[pc] + 1e-8f);
        dist += z * z;
    }
    dist /= KS_N_PCS;
    return SIGMOID_CHIP(2.0f - dist);
}

/**
 * Finalize enrollment: learn discriminant from collected samples,
 * snapshot drift baseline.
 */
static void finalize_enrollment(
    const float samples[][HIDDEN_DIM],
    int n_samples,
    KeystrokeDiscriminant *disc,
    DriftDetector *drift
) {
    learn_discriminant(samples, n_samples, disc);

    /* Snapshot drift baseline */
    for (int i = 0; i < INPUT_DIM; i++) {
        drift->enrollment_mean[i] = drift->input_stats[i].mean;
        drift->enrollment_var[i] = RUNNING_STATS_VARIANCE(&drift->input_stats[i]);
    }
    drift->warmup_done = 1;
}

static int check_drift(const DriftDetector *drift) {
    if (!drift->warmup_done) return 0;
    for (int i = 0; i < INPUT_DIM; i++) {
        float cur_var = RUNNING_STATS_VARIANCE(&drift->input_stats[i]);
        float enr_std = sqrtf(drift->enrollment_var[i] + 1e-8f);
        float mean_d = fabsf(drift->input_stats[i].mean - drift->enrollment_mean[i]);
        if (mean_d > 2.0f * enr_std) return 1;
        if (drift->enrollment_var[i] > 1e-8f && cur_var / drift->enrollment_var[i] > 3.0f)
            return 1;
    }
    return 0;
}

/* ═══════════════════════════════════════════════════════════════════════════
 * Terminal Raw Input
 * ═══════════════════════════════════════════════════════════════════════════ */

#if HAS_RAW_INPUT

static struct termios orig_termios;

static void disable_raw_mode(void) {
    tcsetattr(STDIN_FILENO, TCSAFLUSH, &orig_termios);
}

static void enable_raw_mode(void) {
    tcgetattr(STDIN_FILENO, &orig_termios);
    atexit(disable_raw_mode);
    struct termios raw = orig_termios;
    raw.c_lflag &= ~(ECHO | ICANON);
    raw.c_cc[VMIN]  = 1;
    raw.c_cc[VTIME] = 0;
    tcsetattr(STDIN_FILENO, TCSAFLUSH, &raw);
}

static double get_time_sec(void) {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (double)tv.tv_sec + (double)tv.tv_usec / 1e6;
}

static void print_score_bar(float score, float mean_s, float pca_s, int warmup) {
    const int width = 30;
    int filled = (int)(score * width);
    if (filled < 0) filled = 0;
    if (filled > width) filled = width;

    printf("\r  [");
    for (int i = 0; i < width; i++) {
        if (i < filled) printf("=");
        else if (i == filled) printf(">");
        else printf(" ");
    }

    if (warmup) {
        printf("] ----  warming up");
    } else {
        printf("] %.2f (m:%.2f p:%.2f)", score, mean_s, pca_s);
        if (score > 0.7f) printf("  MATCH");
        else if (score > 0.4f) printf("  uncertain");
        else printf("  MISMATCH");
    }
    printf("      ");
    fflush(stdout);
}

static void run_interactive(void) {
    printf("═══════════════════════════════════════════════════════════\n");
    printf("  Keystroke Biometric v3 — Hybrid Linear Discriminant\n");
    printf("═══════════════════════════════════════════════════════════\n\n");
    printf("  Pipeline: keystroke -> CfC -> hybrid score (mean + PCA)\n");
    printf("  Discriminant learned from enrollment — no backprop.\n");
    printf("  Score = 0.6*mean_distance + 0.4*PCA_distance\n\n");

    float h_state[HIDDEN_DIM] = {0};
    KeystrokeDiscriminant disc;
    memset(&disc, 0, sizeof(disc));
    DriftDetector drift;
    drift_init(&drift);

    /* Enrollment sample collection */
    float enroll_samples[KS_MAX_SAMPLES][HIDDEN_DIM];
    int n_enroll_samples = 0;

    int phase = 0;  /* 0=enrollment, 1=auth */
    int total_keys = 0;
    int enroll_keys = 0;
    int auth_keys = 0;

    printf("PHASE 1: ENROLLMENT\n");
    printf("  Type naturally for ~%d keystrokes.\n", KS_ENROLL_KEYS);
    printf("  Press ESC to finish early (min %d) or auto-completes at %d.\n\n",
           KS_MIN_ENROLL, KS_ENROLL_KEYS);
    printf("  Start typing: ");
    fflush(stdout);

    enable_raw_mode();
    double last_time = get_time_sec();

    while (1) {
        char c;
        if (read(STDIN_FILENO, &c, 1) != 1) continue;
        if (c == 27) {  /* ESC */
            if (phase == 0 && enroll_keys >= KS_MIN_ENROLL) {
                finalize_enrollment(
                    (const float (*)[HIDDEN_DIM])enroll_samples,
                    n_enroll_samples, &disc, &drift);
                phase = 1;
                auth_keys = 0;
                printf("\n\nPHASE 2: AUTHENTICATION\n");
                printf("  Enrolled with %d keystrokes (%d hidden state samples).\n",
                       enroll_keys, n_enroll_samples);
                printf("  Discriminant: %lu bytes (mean + dim_std + %d PCs + stats)\n",
                       sizeof(KeystrokeDiscriminant), KS_N_PCS);
                printf("  Keep typing to see score. Hand keyboard to someone else.\n");
                printf("  Press ESC to exit.\n\n  ");
                fflush(stdout);
                memset(h_state, 0, sizeof(h_state));
                last_time = get_time_sec();
                continue;
            } else {
                break;
            }
        }
        if (c == 3) break;  /* Ctrl-C */

        double now = get_time_sec();
        float dt = (float)(now - last_time);
        last_time = now;
        if (dt > 5.0f) dt = 5.0f;
        if (total_keys == 0) dt = 0.2f;

        float key_norm = ((float)((unsigned char)c) - 32.0f) / 94.0f;
        if (key_norm < 0.0f) key_norm = 0.0f;
        if (key_norm > 1.0f) key_norm = 1.0f;

        /* Update drift monitor (not in CfC path) */
        float raw[INPUT_DIM] = { key_norm, dt };
        for (int i = 0; i < INPUT_DIM; i++)
            RUNNING_STATS_UPDATE(&drift.input_stats[i], raw[i]);

        /* Step CfC */
        cfc_step(key_norm, dt, h_state);
        total_keys++;

        if (phase == 0) {
            enroll_keys++;

            /* Collect samples after warmup */
            if (enroll_keys > KS_WARMUP && n_enroll_samples < KS_MAX_SAMPLES) {
                memcpy(enroll_samples[n_enroll_samples], h_state,
                       HIDDEN_DIM * sizeof(float));
                n_enroll_samples++;
            }

            int pct = (enroll_keys * 100) / KS_ENROLL_KEYS;
            if (pct > 100) pct = 100;
            printf("\r  Enrolling... %d/%d [%d%%]  (%d samples)   ",
                   enroll_keys, KS_ENROLL_KEYS, pct, n_enroll_samples);
            fflush(stdout);

            if (enroll_keys >= KS_ENROLL_KEYS) {
                finalize_enrollment(
                    (const float (*)[HIDDEN_DIM])enroll_samples,
                    n_enroll_samples, &disc, &drift);
                phase = 1;
                auth_keys = 0;
                printf("\n\nPHASE 2: AUTHENTICATION\n");
                printf("  Enrolled with %d keystrokes (%d hidden state samples).\n",
                       enroll_keys, n_enroll_samples);
                printf("  Discriminant: %lu bytes (mean + dim_std + %d PCs + stats)\n",
                       sizeof(KeystrokeDiscriminant), KS_N_PCS);
                printf("  Keep typing to see score. Hand keyboard to someone else.\n");
                printf("  Press ESC to exit.\n\n  ");
                fflush(stdout);
                memset(h_state, 0, sizeof(h_state));
                last_time = get_time_sec();
            }
        } else {
            auth_keys++;
            int in_warmup = (auth_keys <= KS_WARMUP);
            float score = score_hybrid(h_state, &disc);
            float ms = score_mean_only(h_state, &disc);
            float ps = score_pca_only(h_state, &disc);
            print_score_bar(score, ms, ps, in_warmup);

            if (check_drift(&drift)) printf(" [DRIFT]");
            fflush(stdout);
        }
    }

    disable_raw_mode();
    printf("\n\nSession complete. %d total keystrokes.\n", total_keys);
}

#endif /* HAS_RAW_INPUT */

/* ═══════════════════════════════════════════════════════════════════════════
 * Simulation Mode
 *
 * Runs 4 difficulty levels with 3 scoring methods (mean, PCA, hybrid).
 * Independent random seeds for enrollment and auth.
 * Reports honestly — no inflation.
 * ═══════════════════════════════════════════════════════════════════════════ */

static float sim_dt(float mean, float jitter, unsigned int *s) {
    float u = (float)rand_r(s) / (float)RAND_MAX;
    float v = (float)rand_r(s) / (float)RAND_MAX;
    float dt = mean + jitter * (u + v - 1.0f);
    return dt < 0.02f ? 0.02f : dt;
}

static float sim_key(unsigned int *s) {
    return ((float)(32 + rand_r(s) % 95) - 32.0f) / 94.0f;
}

/**
 * Collect hidden state samples from a simulated typing session.
 * Skips first KS_WARMUP keystrokes (cold start transient).
 */
static int sim_collect_samples(
    int n_keys, float mean_dt, float jitter, unsigned int *seed,
    float samples[][HIDDEN_DIM], int max_samples
) {
    float h_state[HIDDEN_DIM] = {0};

    /* Warmup */
    for (int k = 0; k < KS_WARMUP; k++)
        cfc_step(sim_key(seed), sim_dt(mean_dt, jitter, seed), h_state);

    /* Collect */
    int n = 0;
    for (int k = 0; k < n_keys && n < max_samples; k++) {
        cfc_step(sim_key(seed), sim_dt(mean_dt, jitter, seed), h_state);
        memcpy(samples[n], h_state, sizeof(float) * HIDDEN_DIM);
        n++;
    }
    return n;
}

/**
 * Get final hidden state from a simulated auth session.
 * Skips first KS_WARMUP keystrokes.
 */
static void sim_get_final_h(
    int n_keys, float mean_dt, float jitter, unsigned int *seed,
    float *h_out
) {
    float h_state[HIDDEN_DIM] = {0};
    for (int k = 0; k < KS_WARMUP; k++)
        cfc_step(sim_key(seed), sim_dt(mean_dt, jitter, seed), h_state);
    for (int k = 0; k < n_keys; k++)
        cfc_step(sim_key(seed), sim_dt(mean_dt, jitter, seed), h_state);
    memcpy(h_out, h_state, sizeof(float) * HIDDEN_DIM);
}

/**
 * Run one difficulty level across all three scoring methods.
 */
static void sim_test(
    const char *label,
    float a_speed, float a_jitter,
    float b_speed, float b_jitter,
    int n_runs
) {
    printf("  %s\n", label);
    printf("    %-14s  %-10s  %-10s  %-10s  %-8s\n",
           "Method", "Avg A", "Avg B", "Sep", "A wins");

    typedef float (*score_fn)(const float *, const KeystrokeDiscriminant *);
    score_fn fns[] = { score_mean_only, score_pca_only, score_hybrid };
    char hybrid_label[32];
    snprintf(hybrid_label, sizeof(hybrid_label), "hybrid(%.1f/%.1f)",
             KS_MEAN_WEIGHT, KS_PCA_WEIGHT);
    const char *names[] = { "mean-only", "PCA(5)-only", hybrid_label };

    for (int mi = 0; mi < 3; mi++) {
        float sum_a = 0, sum_b = 0;
        int a_wins = 0;

        for (int r = 0; r < n_runs; r++) {
            /* Enroll User A */
            unsigned int enroll_seed = 100 + r * 31;
            float samples[KS_MAX_SAMPLES][HIDDEN_DIM];
            int ns = sim_collect_samples(80, a_speed, a_jitter, &enroll_seed,
                                         samples, KS_MAX_SAMPLES);

            KeystrokeDiscriminant disc;
            learn_discriminant(samples, ns, &disc);

            /* Auth User A (independent seed) */
            unsigned int sa = 3000 + r * 47;
            float ha[HIDDEN_DIM];
            sim_get_final_h(50, a_speed, a_jitter, &sa, ha);

            /* Auth User B (independent seed) */
            unsigned int sb = 7000 + r * 67;
            float hb[HIDDEN_DIM];
            sim_get_final_h(50, b_speed, b_jitter, &sb, hb);

            float score_a = fns[mi](ha, &disc);
            float score_b = fns[mi](hb, &disc);

            sum_a += score_a;
            sum_b += score_b;
            if (score_a > score_b) a_wins++;
        }

        float avg_a = sum_a / n_runs, avg_b = sum_b / n_runs;
        printf("    %-14s  %-10.3f  %-10.3f  %+-10.3f  %d/%d\n",
               names[mi], avg_a, avg_b, avg_a - avg_b, a_wins, n_runs);
    }
    printf("\n");
}

static void run_simulation(void) {
    printf("═══════════════════════════════════════════════════════════\n");
    printf("  Keystroke Biometric v3 — Hybrid Linear Discriminant\n");
    printf("═══════════════════════════════════════════════════════════\n\n");

    printf("  Approach: learn mean + PCA readout during enrollment.\n");
    printf("  No backprop. No optimizer. Just linear algebra on CfC\n");
    printf("  hidden states. Weights are pre-initialized, NOT trained.\n\n");
    printf("  Hybrid score = %.1f * mean_distance + %.1f * PCA_distance\n",
           KS_MEAN_WEIGHT, KS_PCA_WEIGHT);
    printf("  PCs: %d (sweet spot for %d-dim hidden state)\n\n", KS_N_PCS, HIDDEN_DIM);

    int N = 20;

    /* ── Experiment 1: Consistency check ── */
    printf("EXPERIMENT 1: Hidden state consistency (same user, 10 runs)\n");
    {
        float h_runs[10][HIDDEN_DIM];
        for (int r = 0; r < 10; r++) {
            unsigned int seed = 1000 + r * 37;
            float samples[KS_MAX_SAMPLES][HIDDEN_DIM];
            int ns = sim_collect_samples(80, 0.12f, 0.03f, &seed, samples, KS_MAX_SAMPLES);
            /* Use final sample as representative hidden state */
            memcpy(h_runs[r], samples[ns - 1], sizeof(float) * HIDDEN_DIM);
        }

        float cos_sum = 0; int npairs = 0;
        for (int i = 0; i < 10; i++)
            for (int j = i + 1; j < 10; j++) {
                float d = dotf(h_runs[i], h_runs[j], HIDDEN_DIM);
                float ni = 0, nj = 0;
                for (int k = 0; k < HIDDEN_DIM; k++) {
                    ni += h_runs[i][k] * h_runs[i][k];
                    nj += h_runs[j][k] * h_runs[j][k];
                }
                float denom = sqrtf(ni) * sqrtf(nj);
                cos_sum += denom > 1e-10f ? d / denom : 0.0f;
                npairs++;
            }
        printf("  Mean intra-class cosine: %.4f (%d pairs)\n", cos_sum / npairs, npairs);
        printf("  (>0.85 = consistent representations)\n\n");
    }

    /* ── Experiment 2: Four difficulty levels, three methods ── */
    printf("EXPERIMENT 2: Separation across difficulty levels\n\n");

    sim_test("TEST 1: Easy — 3x speed (A=0.12s, B=0.35s)",
             0.12f, 0.03f, 0.35f, 0.10f, N);

    sim_test("TEST 2: Medium — 1.5x speed (A=0.12s, B=0.18s)",
             0.12f, 0.03f, 0.18f, 0.045f, N);

    sim_test("TEST 3: Hard — same speed, diff jitter (A=0.15s/0.02j, B=0.15s/0.08j)",
             0.15f, 0.02f, 0.15f, 0.08f, N);

    sim_test("TEST 4: Control — same everything (A=B=0.15s/0.04j)",
             0.15f, 0.04f, 0.15f, 0.04f, N);

    /* ── Experiment 3: Execution time ── */
    printf("EXPERIMENT 3: Execution time\n");
    {
        float h_state[HIDDEN_DIM] = {0};
        unsigned int seed = 77;
        int M = 10000;

        /* Warmup */
        for (int k = 0; k < 100; k++) {
            float dt = sim_dt(0.15f, 0.04f, &seed);
            cfc_step(sim_key(&seed), dt, h_state);
        }

        struct timeval t0, t1;
        gettimeofday(&t0, NULL);
        for (int k = 0; k < M; k++) {
            float dt = sim_dt(0.15f, 0.04f, &seed);
            cfc_step(sim_key(&seed), dt, h_state);
        }
        gettimeofday(&t1, NULL);
        double us_cfc = (double)(t1.tv_sec - t0.tv_sec) * 1e6 +
                        (double)(t1.tv_usec - t0.tv_usec);

        /* Time hybrid scoring */
        KeystrokeDiscriminant disc;
        {
            unsigned int es = 42;
            float samples[KS_MAX_SAMPLES][HIDDEN_DIM];
            int ns = sim_collect_samples(80, 0.15f, 0.04f, &es, samples, KS_MAX_SAMPLES);
            learn_discriminant(samples, ns, &disc);
        }

        gettimeofday(&t0, NULL);
        volatile float dummy = 0;
        for (int k = 0; k < M; k++) {
            dummy += score_hybrid(h_state, &disc);
        }
        gettimeofday(&t1, NULL);
        double us_score = (double)(t1.tv_sec - t0.tv_sec) * 1e6 +
                          (double)(t1.tv_usec - t0.tv_usec);
        (void)dummy;

        printf("  %d CfC steps in %.0f us (%.0f ns/step)\n", M, us_cfc, (us_cfc / M) * 1000);
        printf("  %d hybrid scores in %.0f us (%.0f ns/score)\n", M, us_score, (us_score / M) * 1000);
        printf("  Total per keystroke: %.0f ns (CfC + score)\n",
               ((us_cfc + us_score) / M) * 1000);
    }

    /* ── Summary ── */
    printf("\n═══════════════════════════════════════════════════════════\n");
    printf("  Summary:\n");
    printf("    Pipeline: keystroke -> CfC -> hybrid(mean + PCA) -> score\n");
    printf("    CfC weights: %lu bytes (pre-initialized, not trained)\n",
           sizeof(W_gate) + sizeof(b_gate) + sizeof(W_cand) + sizeof(b_cand) +
           sizeof(tau));
    printf("    Hidden state: %lu bytes (%d floats)\n",
           sizeof(float) * HIDDEN_DIM, HIDDEN_DIM);
    printf("    Discriminant: %lu bytes (mean + dim_std + %d PCs + stats)\n",
           sizeof(KeystrokeDiscriminant), KS_N_PCS);
    printf("    Hybrid score: %.1f*mean + %.1f*PCA (sigmoid z-score)\n",
           KS_MEAN_WEIGHT, KS_PCA_WEIGHT);
    printf("    Chips: cfc_cell_chip, norm_chip (drift only), activation_chip\n");
    printf("═══════════════════════════════════════════════════════════\n");
}

/* ═══════════════════════════════════════════════════════════════════════════ */

int main(int argc, char **argv) {
    int sim_mode = 0;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--sim") == 0 || strcmp(argv[i], "-s") == 0)
            sim_mode = 1;
        if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0) {
            printf("Usage: keystroke_biometric [--sim]\n\n");
            printf("  --sim, -s   Simulation mode (no keyboard needed)\n");
            printf("  (default)   Interactive mode — type to enroll and compare\n\n");
            printf("  v3: Hybrid linear discriminant (mean + PCA)\n");
            printf("  Discriminant learned from enrollment — no backprop.\n");
            return 0;
        }
    }

    if (sim_mode) {
        run_simulation();
    } else {
#if HAS_RAW_INPUT
        run_interactive();
#else
        printf("Interactive mode not available. Use --sim.\n");
        return 1;
#endif
    }
    return 0;
}
