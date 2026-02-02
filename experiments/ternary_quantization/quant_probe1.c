/*
 * Ternary Quantization Probe 1 — The Bridge Experiment
 *
 * THE critical experiment: does quantizing CfC weights from float32 to
 * ternary {-1, 0, +1} preserve enrollment-based detection quality?
 *
 * Method:
 *   1. Take the shared hand-initialized float32 weights (W_gate, W_cand)
 *   2. Quantize each weight: |w| > threshold → sign(w), else → 0
 *   3. Run identical ISS-style enrollment + anomaly detection pipeline
 *   4. Side-by-side: float CfC vs ternary CfC on same input stream
 *   5. Sweep threshold in [0.05, 0.1, 0.2, 0.3, 0.5]
 *
 * Success criteria (from real_gap_synth.md):
 *   - Discriminant convergence within 15% of float baseline
 *   - Detection latency within 2x of float baseline
 *   - At least one threshold works across all tests
 *
 * Kill criteria:
 *   - If NO threshold preserves >50% of discriminant score, ternary CfC
 *     is falsified for enrollment-based detection. Document honestly.
 *
 * Uses ISS simulation as test domain (most complex: 8 channels, orbital
 * cycling, 5 anomaly types). If ternary works here, it works everywhere.
 *
 * Compile:
 *   cc -O2 -I include -I include/chips experiments/ternary_quantization/quant_probe1.c -lm -o experiments/ternary_quantization/quant_probe1
 *
 * Created by: Tripp + Manus
 * Date: February 2026
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <sys/time.h>

/* Chip includes */
#include "onnx_shapes.h"
#include "chips/cfc_cell_chip.h"
#include "chips/norm_chip.h"
#include "chips/activation_chip.h"
#include "chips/ternary_dot_chip.h"
#include "trit_encoding.h"

/* ═══════════════════════════════════════════════════════════════════════════
 * Dimensions (same as all demos)
 * ═══════════════════════════════════════════════════════════════════════════ */
#define INPUT_DIM    2
#define HIDDEN_DIM   8
#define CONCAT_DIM   (INPUT_DIM + HIDDEN_DIM)  /* 10 */
#define N_CHANNELS   8

/* Discriminant */
#define N_PCS        5
#define WARMUP       20
#define MAX_SAMPLES  500
#define POWER_ITERS  20
#define ENROLL_SAMPLES 300
#define ANOMALY_THRESH 0.35f
#define MEAN_WEIGHT    0.3f
#define PCA_WEIGHT     0.7f

/* Simulation */
#define SIM_DT             10.0f
#define ORBITAL_PERIOD     5520.0f
#define CAL_ORBITS         1.0f
#define ENROLL_ORBITS      2.0f
#define ANOMALY_ORBITS     1.0f

/* Threshold sweep */
#define N_THRESHOLDS  5
static const float thresholds[N_THRESHOLDS] = {
    0.05f, 0.1f, 0.2f, 0.3f, 0.5f
};

/* ═══════════════════════════════════════════════════════════════════════════
 * Float Weights (canonical — shared across all demos)
 * ═══════════════════════════════════════════════════════════════════════════ */

static const float W_gate_float[HIDDEN_DIM * CONCAT_DIM] = {
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

static const float W_cand_float[HIDDEN_DIM * CONCAT_DIM] = {
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

/* ISS tau values */
static const float tau[HIDDEN_DIM] = {
    5.0f, 15.0f, 45.0f, 120.0f, 10.0f, 30.0f, 90.0f, 600.0f
};

/* ═══════════════════════════════════════════════════════════════════════════
 * Ternary Quantization
 *
 * Quantize float weight to ternary:
 *   |w| > threshold → sign(w) ∈ {-1, +1}
 *   |w| <= threshold → 0
 *
 * Then expand to float {-1.0, 0.0, +1.0} for the standard CfC cell,
 * AND pack to 2-bit for the ternary dot path.
 * ═══════════════════════════════════════════════════════════════════════════ */

static void quantize_weights(
    const float *src, float *dst_float, uint8_t *dst_packed,
    int n, float threshold,
    int *n_pos, int *n_neg, int *n_zero
) {
    *n_pos = 0; *n_neg = 0; *n_zero = 0;

    /* Pack: 4 trits per byte, LSB first */
    int n_bytes = (n + 3) / 4;
    memset(dst_packed, 0, n_bytes);

    for (int i = 0; i < n; i++) {
        float w = src[i];
        uint8_t trit;

        if (w > threshold) {
            dst_float[i] = 1.0f;
            trit = TRIT_POS;
            (*n_pos)++;
        } else if (w < -threshold) {
            dst_float[i] = -1.0f;
            trit = TRIT_NEG;
            (*n_neg)++;
        } else {
            dst_float[i] = 0.0f;
            trit = TRIT_ZERO;
            (*n_zero)++;
        }

        int byte_idx = i / 4;
        int bit_pos = (i % 4) * 2;
        dst_packed[byte_idx] |= (trit << bit_pos);
    }
}

/* ═══════════════════════════════════════════════════════════════════════════
 * CfC with Ternary Dot Product (using chip primitive)
 *
 * Same computation as CFC_CELL_GENERIC but uses TERNARY_MATVEC_BIAS_CHIP
 * for the W*concat products. Biases remain float. Hidden state remains
 * float. Only the weight multiply is ternary (add/subtract/skip).
 * ═══════════════════════════════════════════════════════════════════════════ */

static inline void CFC_CELL_TERNARY(
    const float *x,
    const float *h_prev,
    float dt,
    const uint8_t *W_gate_packed,  /* packed ternary */
    const float *b_gate_f,
    const uint8_t *W_cand_packed,  /* packed ternary */
    const float *b_cand_f,
    const float *tau_f,
    int input_dim,
    int hidden_dim,
    float *h_new
) {
    const int concat_dim = input_dim + hidden_dim;
    float concat[concat_dim];
    float gate_pre[hidden_dim];
    float gate[hidden_dim];
    float cand_pre[hidden_dim];
    float candidate[hidden_dim];
    float decay[hidden_dim];

    /* Concatenate [x; h_prev] */
    memcpy(concat, x, input_dim * sizeof(float));
    memcpy(concat + input_dim, h_prev, hidden_dim * sizeof(float));

    /* Gate = sigmoid(W_gate @ concat + b_gate) — TERNARY DOT */
    TERNARY_MATVEC_BIAS_CHIP(W_gate_packed, concat, b_gate_f,
                             gate_pre, hidden_dim, concat_dim);
    for (int i = 0; i < hidden_dim; i++)
        gate[i] = yinsen_sigmoid(gate_pre[i]);

    /* Candidate = tanh(W_cand @ concat + b_cand) — TERNARY DOT */
    TERNARY_MATVEC_BIAS_CHIP(W_cand_packed, concat, b_cand_f,
                             cand_pre, hidden_dim, concat_dim);
    for (int i = 0; i < hidden_dim; i++)
        candidate[i] = yinsen_tanh(cand_pre[i]);

    /* Decay = exp(-dt / tau) */
    for (int i = 0; i < hidden_dim; i++)
        decay[i] = expf(-dt / tau_f[i]);

    /* Mix: h_new = (1 - gate) * h_prev * decay + gate * candidate */
    for (int i = 0; i < hidden_dim; i++) {
        float retention = (1.0f - gate[i]) * h_prev[i] * decay[i];
        float update = gate[i] * candidate[i];
        h_new[i] = retention + update;
    }
}

/* ═══════════════════════════════════════════════════════════════════════════
 * Discriminant (identical to ISS/keystroke/seismic)
 * ═══════════════════════════════════════════════════════════════════════════ */

typedef struct {
    float mean[HIDDEN_DIM];
    float dim_std[HIDDEN_DIM];
    float pcs[N_PCS][HIDDEN_DIM];
    float pc_mean[N_PCS];
    float pc_std[N_PCS];
    int valid;
} Discriminant;

static float dotf(const float *a, const float *b, int n) {
    float s = 0;
    for (int i = 0; i < n; i++) s += a[i] * b[i];
    return s;
}

static float vec_normalize(float *v, int n) {
    float norm = 0;
    for (int i = 0; i < n; i++) norm += v[i] * v[i];
    norm = sqrtf(norm);
    if (norm > 1e-10f)
        for (int i = 0; i < n; i++) v[i] /= norm;
    return norm;
}

static void learn_discriminant(
    const float samples[][HIDDEN_DIM], int n_samples, Discriminant *disc
) {
    memset(disc, 0, sizeof(*disc));
    if (n_samples < 5) { disc->valid = 0; return; }

    for (int i = 0; i < HIDDEN_DIM; i++) {
        float s = 0;
        for (int t = 0; t < n_samples; t++) s += samples[t][i];
        disc->mean[i] = s / n_samples;
    }

    for (int i = 0; i < HIDDEN_DIM; i++) {
        float s = 0;
        for (int t = 0; t < n_samples; t++) {
            float d = samples[t][i] - disc->mean[i];
            s += d * d;
        }
        disc->dim_std[i] = sqrtf(s / n_samples + 1e-8f);
    }

    float centered[n_samples][HIDDEN_DIM];
    for (int t = 0; t < n_samples; t++)
        for (int i = 0; i < HIDDEN_DIM; i++)
            centered[t][i] = samples[t][i] - disc->mean[i];

    for (int pc = 0; pc < N_PCS; pc++) {
        float v[HIDDEN_DIM];
        for (int i = 0; i < HIDDEN_DIM; i++)
            v[i] = centered[0][i] + 0.01f * (i + 1);
        vec_normalize(v, HIDDEN_DIM);

        for (int iter = 0; iter < POWER_ITERS; iter++) {
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

        float proj_sum = 0, proj_sum2 = 0;
        for (int t = 0; t < n_samples; t++) {
            float p = dotf(centered[t], v, HIDDEN_DIM);
            proj_sum += p;
            proj_sum2 += p * p;
        }
        disc->pc_mean[pc] = proj_sum / n_samples;
        float var = proj_sum2 / n_samples - disc->pc_mean[pc] * disc->pc_mean[pc];
        disc->pc_std[pc] = sqrtf(var > 0 ? var : 1e-8f);

        for (int t = 0; t < n_samples; t++) {
            float p = dotf(centered[t], v, HIDDEN_DIM);
            for (int i = 0; i < HIDDEN_DIM; i++)
                centered[t][i] -= p * v[i];
        }
    }
    disc->valid = 1;
}

static float score_discriminant(
    const float *h_state, const Discriminant *disc
) {
    if (!disc->valid) return 0.5f;

    float centered[HIDDEN_DIM];
    float mean_dist = 0;
    for (int i = 0; i < HIDDEN_DIM; i++) {
        centered[i] = h_state[i] - disc->mean[i];
        float z = centered[i] / (disc->dim_std[i] + 1e-8f);
        mean_dist += z * z;
    }
    mean_dist /= HIDDEN_DIM;
    float mean_score = SIGMOID_CHIP(2.0f - mean_dist);

    float pca_dist = 0;
    for (int pc = 0; pc < N_PCS; pc++) {
        float proj = dotf(centered, disc->pcs[pc], HIDDEN_DIM);
        float z = (proj - disc->pc_mean[pc]) / (disc->pc_std[pc] + 1e-8f);
        pca_dist += z * z;
    }
    pca_dist /= N_PCS;
    float pca_score = SIGMOID_CHIP(2.0f - pca_dist);

    return MEAN_WEIGHT * mean_score + PCA_WEIGHT * pca_score;
}

/* ═══════════════════════════════════════════════════════════════════════════
 * Per-Channel State (one for float, one for ternary)
 * ═══════════════════════════════════════════════════════════════════════════ */

typedef struct {
    float h_state[HIDDEN_DIM];
    Discriminant disc;
    float enroll_buf[MAX_SAMPLES][HIDDEN_DIM];
    int n_enroll;
    int sample_count;
    float input_mean;
    float input_std;
} Channel;

/* ═══════════════════════════════════════════════════════════════════════════
 * Simulation — ISS-style with orbital cycling
 * ═══════════════════════════════════════════════════════════════════════════ */

static float randf(unsigned int *s) {
    *s = *s * 1103515245u + 12345u;
    return (float)(*s & 0x7FFFFFFFu) / (float)0x7FFFFFFFu;
}

static float gaussf(unsigned int *s) {
    float u1 = randf(s) + 1e-10f;
    float u2 = randf(s);
    return sqrtf(-2.0f * logf(u1)) * cosf(6.28318530718f * u2);
}

static float sim_value(float t, int ch, unsigned int *seed, int anomaly, float anom_start) {
    float phase = 2.0f * 3.14159265f * t / ORBITAL_PERIOD;
    float noise = gaussf(seed);
    float value = 0;

    switch (ch) {
    case 0: case 1: case 2: case 3:
        value = 0.001f + 0.0001f * ch + 0.0002f * sinf(phase) + 0.0002f * noise;
        break;
    case 4: case 5:
        value = 15.0f + (ch == 5 ? 2.0f : 0.0f) + 8.0f * sinf(phase) + 0.5f * noise;
        break;
    case 6:
        value = 101.3f + 0.02f * sinf(phase) + 0.05f * noise;
        break;
    case 7:
        value = 21.3f + 0.15f * (fmodf(t, 900.0f) / 900.0f - 0.5f) + 0.1f * noise;
        break;
    }

    /* Anomaly injection: step change */
    if (anomaly && ch == 0)  value += 0.005f;   /* CMG spike */
    if (anomaly && ch == 4)  value += 3.0f * sinf(2.0f * 3.14159265f * (t - anom_start) / 200.0f); /* oscillation */
    if (anomaly && ch == 6)  value -= 1.0f * fminf((t - anom_start) / 3000.0f, 1.0f);  /* slow leak */

    return value;
}

/* ═══════════════════════════════════════════════════════════════════════════
 * MAIN — Side-by-side float vs ternary at multiple thresholds
 * ═══════════════════════════════════════════════════════════════════════════ */

int main(void) {
    printf("═══════════════════════════════════════════════════════════════\n");
    printf("  TERNARY QUANTIZATION PROBE 1 — The Bridge Experiment\n");
    printf("═══════════════════════════════════════════════════════════════\n\n");

    int total_weights = 2 * HIDDEN_DIM * CONCAT_DIM;  /* W_gate + W_cand */
    printf("  Weight matrices: W_gate[%d×%d] + W_cand[%d×%d] = %d weights\n",
           HIDDEN_DIM, CONCAT_DIM, HIDDEN_DIM, CONCAT_DIM, total_weights);
    printf("  Float size:   %d bytes\n", total_weights * (int)sizeof(float));
    printf("  Ternary size: %d bytes (2-bit packed)\n",
           2 * ((HIDDEN_DIM * CONCAT_DIM + 3) / 4) * (int)sizeof(uint8_t));
    printf("  Biases: remain float32 (16 values, 64 bytes)\n\n");

    /* ── Quantization analysis ── */
    printf("QUANTIZATION ANALYSIS\n");
    printf("  Unique weight values in float matrices:\n");
    printf("    W_gate: {-0.1, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8}\n");
    printf("    W_cand: {-0.1, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6}\n\n");

    printf("  %-10s  %6s  %6s  %6s  %8s\n",
           "Threshold", "+1", "-1", "0", "Nonzero%");
    printf("  %-10s  %6s  %6s  %6s  %8s\n",
           "----------", "------", "------", "------", "--------");

    /* Storage for all threshold configs */
    float W_gate_tern[N_THRESHOLDS][HIDDEN_DIM * CONCAT_DIM];
    float W_cand_tern[N_THRESHOLDS][HIDDEN_DIM * CONCAT_DIM];
    uint8_t W_gate_packed[N_THRESHOLDS][(HIDDEN_DIM * CONCAT_DIM + 3) / 4];
    uint8_t W_cand_packed[N_THRESHOLDS][(HIDDEN_DIM * CONCAT_DIM + 3) / 4];

    for (int ti = 0; ti < N_THRESHOLDS; ti++) {
        int gp, gn, gz, cp, cn, cz;
        quantize_weights(W_gate_float, W_gate_tern[ti], W_gate_packed[ti],
                         HIDDEN_DIM * CONCAT_DIM, thresholds[ti], &gp, &gn, &gz);
        quantize_weights(W_cand_float, W_cand_tern[ti], W_cand_packed[ti],
                         HIDDEN_DIM * CONCAT_DIM, thresholds[ti], &cp, &cn, &cz);

        int total_pos = gp + cp;
        int total_neg = gn + cn;
        int total_zero = gz + cz;
        float nonzero_pct = 100.0f * (total_pos + total_neg) / total_weights;

        printf("  %-10.2f  %6d  %6d  %6d  %7.1f%%\n",
               thresholds[ti], total_pos, total_neg, total_zero, nonzero_pct);
    }

    /* ── Run side-by-side tests ── */
    printf("\n═══════════════════════════════════════════════════════════════\n");
    printf("  SIDE-BY-SIDE TEST: Float vs Ternary (ISS 8-channel sim)\n");
    printf("═══════════════════════════════════════════════════════════════\n\n");

    /* For each threshold, run full pipeline */
    float float_normal_scores[N_CHANNELS];
    float float_anomaly_scores[N_CHANNELS];
    int float_detections[3];  /* per anomaly test */
    int float_det_steps[3];

    float tern_normal_scores[N_THRESHOLDS][N_CHANNELS];
    float tern_anomaly_scores[N_THRESHOLDS][N_CHANNELS];
    int tern_detections[N_THRESHOLDS][3];
    int tern_det_steps[N_THRESHOLDS][3];

    /* Hidden state divergence tracking */
    float h_divergence[N_THRESHOLDS][N_CHANNELS];  /* mean |h_float - h_tern| */

    static const char *ch_names[N_CHANNELS] = {
        "CMG1", "CMG2", "CMG3", "CMG4", "CoolA", "CoolB", "CabinP", "O2"
    };

    /* Anomaly configs: (target_channel, description) */
    typedef struct { int channel; const char *desc; } AnomalyConfig;
    AnomalyConfig anomalies[3] = {
        {0, "CMG1 step (+5x vibration)"},
        {4, "CoolA oscillation shift"},
        {6, "CabinP slow leak"},
    };

    /* ── Calibration phase (shared across all configs) ── */
    printf("Phase 0: Calibration (1 orbit, %d steps)\n",
           (int)(ORBITAL_PERIOD / SIM_DT));

    double cal_sum[N_CHANNELS], cal_sum2[N_CHANNELS];
    int cal_n = 0;
    memset(cal_sum, 0, sizeof(cal_sum));
    memset(cal_sum2, 0, sizeof(cal_sum2));

    unsigned int seed_cal = 42;
    int cal_steps = (int)(CAL_ORBITS * ORBITAL_PERIOD / SIM_DT);
    for (int step = 0; step < cal_steps; step++) {
        float t = step * SIM_DT;
        for (int ch = 0; ch < N_CHANNELS; ch++) {
            float v = sim_value(t, ch, &seed_cal, 0, 0);
            cal_sum[ch] += v;
            cal_sum2[ch] += (double)v * v;
        }
        cal_n++;
    }

    float ch_mean[N_CHANNELS], ch_std[N_CHANNELS];
    for (int ch = 0; ch < N_CHANNELS; ch++) {
        ch_mean[ch] = (float)(cal_sum[ch] / cal_n);
        float var = (float)(cal_sum2[ch] / cal_n - (double)ch_mean[ch] * ch_mean[ch]);
        ch_std[ch] = sqrtf(var > 0 ? var : 1e-8f);
    }

    printf("  Calibration frozen.\n\n");

    /* ── For each config (float + N thresholds), run enrollment + detection ── */

    /* Run float baseline first */
    printf("Running FLOAT baseline...\n");
    {
        Channel channels[N_CHANNELS];
        memset(channels, 0, sizeof(channels));

        /* Enrollment */
        unsigned int seed_enr = 123;
        int enr_steps = (int)(ENROLL_ORBITS * ORBITAL_PERIOD / SIM_DT);
        float t_base = CAL_ORBITS * ORBITAL_PERIOD;

        for (int step = 0; step < enr_steps; step++) {
            float t = t_base + step * SIM_DT;
            for (int ch = 0; ch < N_CHANNELS; ch++) {
                float raw = sim_value(t, ch, &seed_enr, 0, 0);
                float scaled = (raw - ch_mean[ch]) / (ch_std[ch] + 1e-8f);
                float input[INPUT_DIM] = { scaled, SIM_DT / SIM_DT };
                float h_new[HIDDEN_DIM];

                CFC_CELL_GENERIC(input, channels[ch].h_state, SIM_DT,
                                 W_gate_float, b_gate, W_cand_float, b_cand,
                                 tau, 0, INPUT_DIM, HIDDEN_DIM, h_new);
                memcpy(channels[ch].h_state, h_new, sizeof(h_new));
                channels[ch].sample_count++;

                if (channels[ch].sample_count > WARMUP
                    && channels[ch].n_enroll < MAX_SAMPLES) {
                    memcpy(channels[ch].enroll_buf[channels[ch].n_enroll],
                           h_new, sizeof(h_new));
                    channels[ch].n_enroll++;
                }
            }
        }

        /* Learn discriminants */
        for (int ch = 0; ch < N_CHANNELS; ch++) {
            learn_discriminant(
                (const float (*)[HIDDEN_DIM])channels[ch].enroll_buf,
                channels[ch].n_enroll, &channels[ch].disc);
        }

        /* Normal scoring */
        for (int ch = 0; ch < N_CHANNELS; ch++) {
            memset(channels[ch].h_state, 0, sizeof(channels[ch].h_state));
            channels[ch].sample_count = 0;
        }

        unsigned int seed_norm = 456;
        int norm_steps = (int)(ORBITAL_PERIOD / SIM_DT);  /* 1 orbit */
        float t_norm_base = t_base + enr_steps * SIM_DT;
        float score_sums[N_CHANNELS] = {0};
        int scored = 0;

        for (int step = 0; step < norm_steps; step++) {
            float t = t_norm_base + step * SIM_DT;
            for (int ch = 0; ch < N_CHANNELS; ch++) {
                float raw = sim_value(t, ch, &seed_norm, 0, 0);
                float scaled = (raw - ch_mean[ch]) / (ch_std[ch] + 1e-8f);
                float input[INPUT_DIM] = { scaled, 1.0f };
                float h_new[HIDDEN_DIM];

                CFC_CELL_GENERIC(input, channels[ch].h_state, SIM_DT,
                                 W_gate_float, b_gate, W_cand_float, b_cand,
                                 tau, 0, INPUT_DIM, HIDDEN_DIM, h_new);
                memcpy(channels[ch].h_state, h_new, sizeof(h_new));
                channels[ch].sample_count++;
            }
            if (channels[0].sample_count > WARMUP) {
                for (int ch = 0; ch < N_CHANNELS; ch++)
                    score_sums[ch] += score_discriminant(channels[ch].h_state,
                                                         &channels[ch].disc);
                scored++;
            }
        }

        for (int ch = 0; ch < N_CHANNELS; ch++)
            float_normal_scores[ch] = score_sums[ch] / scored;

        /* Anomaly tests */
        for (int ai = 0; ai < 3; ai++) {
            for (int ch = 0; ch < N_CHANNELS; ch++) {
                memset(channels[ch].h_state, 0, sizeof(channels[ch].h_state));
                channels[ch].sample_count = 0;
            }

            unsigned int seed_anom = 789 + ai * 111;
            int anom_steps = (int)(ANOMALY_ORBITS * ORBITAL_PERIOD / SIM_DT);
            int inject_at = anom_steps / 2;
            float t_anom_base = t_norm_base + norm_steps * SIM_DT;
            float anom_start_time = t_anom_base + inject_at * SIM_DT;

            float_detections[ai] = 0;
            float_det_steps[ai] = -1;

            for (int step = 0; step < anom_steps; step++) {
                float t = t_anom_base + step * SIM_DT;
                int is_anom = (step >= inject_at);
                for (int ch = 0; ch < N_CHANNELS; ch++) {
                    float raw = sim_value(t, ch, &seed_anom, is_anom, anom_start_time);
                    float scaled = (raw - ch_mean[ch]) / (ch_std[ch] + 1e-8f);
                    float input[INPUT_DIM] = { scaled, 1.0f };
                    float h_new[HIDDEN_DIM];

                    CFC_CELL_GENERIC(input, channels[ch].h_state, SIM_DT,
                                     W_gate_float, b_gate, W_cand_float, b_cand,
                                     tau, 0, INPUT_DIM, HIDDEN_DIM, h_new);
                    memcpy(channels[ch].h_state, h_new, sizeof(h_new));
                    channels[ch].sample_count++;
                }

                if (channels[0].sample_count > WARMUP && step >= inject_at) {
                    int tgt = anomalies[ai].channel;
                    float s = score_discriminant(channels[tgt].h_state,
                                                 &channels[tgt].disc);
                    if (!float_detections[ai] && s < ANOMALY_THRESH) {
                        float_detections[ai] = 1;
                        float_det_steps[ai] = step - inject_at;
                    }
                }
            }
        }
    }

    printf("  Float baseline complete.\n\n");

    /* ── Run each ternary threshold ── */
    for (int ti = 0; ti < N_THRESHOLDS; ti++) {
        printf("Running TERNARY threshold=%.2f ...\n", thresholds[ti]);

        Channel channels[N_CHANNELS];
        memset(channels, 0, sizeof(channels));

        /* Enrollment — using ternary CfC with packed weights */
        unsigned int seed_enr = 123;  /* SAME seed as float */
        int enr_steps = (int)(ENROLL_ORBITS * ORBITAL_PERIOD / SIM_DT);
        float t_base = CAL_ORBITS * ORBITAL_PERIOD;

        /* Also run a float shadow for divergence tracking */
        float h_float_shadow[N_CHANNELS][HIDDEN_DIM];
        memset(h_float_shadow, 0, sizeof(h_float_shadow));
        double div_sum[N_CHANNELS];
        int div_n = 0;
        memset(div_sum, 0, sizeof(div_sum));

        for (int step = 0; step < enr_steps; step++) {
            float t = t_base + step * SIM_DT;
            unsigned int seed_copy = seed_enr;  /* both paths see same RNG */

            for (int ch = 0; ch < N_CHANNELS; ch++) {
                float raw = sim_value(t, ch, &seed_enr, 0, 0);
                float scaled = (raw - ch_mean[ch]) / (ch_std[ch] + 1e-8f);
                float input[INPUT_DIM] = { scaled, SIM_DT / SIM_DT };
                float h_new[HIDDEN_DIM];

                /* Ternary path */
                CFC_CELL_TERNARY(input, channels[ch].h_state, SIM_DT,
                                 W_gate_packed[ti], b_gate,
                                 W_cand_packed[ti], b_cand,
                                 tau, INPUT_DIM, HIDDEN_DIM, h_new);
                memcpy(channels[ch].h_state, h_new, sizeof(h_new));
                channels[ch].sample_count++;

                if (channels[ch].sample_count > WARMUP
                    && channels[ch].n_enroll < MAX_SAMPLES) {
                    memcpy(channels[ch].enroll_buf[channels[ch].n_enroll],
                           h_new, sizeof(h_new));
                    channels[ch].n_enroll++;
                }

                /* Float shadow for divergence */
                float h_float_new[HIDDEN_DIM];
                /* Need to regenerate same raw value — use saved RNG state */
                CFC_CELL_GENERIC(input, h_float_shadow[ch], SIM_DT,
                                 W_gate_float, b_gate, W_cand_float, b_cand,
                                 tau, 0, INPUT_DIM, HIDDEN_DIM, h_float_new);
                memcpy(h_float_shadow[ch], h_float_new, sizeof(h_float_new));
            }

            /* Track divergence after warmup */
            if (channels[0].sample_count > WARMUP) {
                for (int ch = 0; ch < N_CHANNELS; ch++) {
                    float d = 0;
                    for (int i = 0; i < HIDDEN_DIM; i++) {
                        float diff = channels[ch].h_state[i] - h_float_shadow[ch][i];
                        d += diff * diff;
                    }
                    div_sum[ch] += sqrtf(d);
                }
                div_n++;
            }
        }

        for (int ch = 0; ch < N_CHANNELS; ch++)
            h_divergence[ti][ch] = (div_n > 0) ? (float)(div_sum[ch] / div_n) : 0;

        /* Learn discriminants */
        for (int ch = 0; ch < N_CHANNELS; ch++) {
            learn_discriminant(
                (const float (*)[HIDDEN_DIM])channels[ch].enroll_buf,
                channels[ch].n_enroll, &channels[ch].disc);
        }

        /* Normal scoring */
        for (int ch = 0; ch < N_CHANNELS; ch++) {
            memset(channels[ch].h_state, 0, sizeof(channels[ch].h_state));
            channels[ch].sample_count = 0;
        }

        unsigned int seed_norm = 456;
        int norm_steps = (int)(ORBITAL_PERIOD / SIM_DT);
        float t_norm_base = t_base + enr_steps * SIM_DT;
        float score_sums[N_CHANNELS] = {0};
        int scored = 0;

        for (int step = 0; step < norm_steps; step++) {
            float t = t_norm_base + step * SIM_DT;
            for (int ch = 0; ch < N_CHANNELS; ch++) {
                float raw = sim_value(t, ch, &seed_norm, 0, 0);
                float scaled = (raw - ch_mean[ch]) / (ch_std[ch] + 1e-8f);
                float input[INPUT_DIM] = { scaled, 1.0f };
                float h_new[HIDDEN_DIM];

                CFC_CELL_TERNARY(input, channels[ch].h_state, SIM_DT,
                                 W_gate_packed[ti], b_gate,
                                 W_cand_packed[ti], b_cand,
                                 tau, INPUT_DIM, HIDDEN_DIM, h_new);
                memcpy(channels[ch].h_state, h_new, sizeof(h_new));
                channels[ch].sample_count++;
            }
            if (channels[0].sample_count > WARMUP) {
                for (int ch = 0; ch < N_CHANNELS; ch++)
                    score_sums[ch] += score_discriminant(channels[ch].h_state,
                                                         &channels[ch].disc);
                scored++;
            }
        }

        for (int ch = 0; ch < N_CHANNELS; ch++)
            tern_normal_scores[ti][ch] = score_sums[ch] / scored;

        /* Anomaly tests */
        for (int ai = 0; ai < 3; ai++) {
            for (int ch = 0; ch < N_CHANNELS; ch++) {
                memset(channels[ch].h_state, 0, sizeof(channels[ch].h_state));
                channels[ch].sample_count = 0;
            }

            unsigned int seed_anom = 789 + ai * 111;
            int anom_steps = (int)(ANOMALY_ORBITS * ORBITAL_PERIOD / SIM_DT);
            int inject_at = anom_steps / 2;
            float t_anom_base = t_norm_base + norm_steps * SIM_DT;
            float anom_start_time = t_anom_base + inject_at * SIM_DT;

            tern_detections[ti][ai] = 0;
            tern_det_steps[ti][ai] = -1;

            for (int step = 0; step < anom_steps; step++) {
                float t = t_anom_base + step * SIM_DT;
                int is_anom = (step >= inject_at);
                for (int ch = 0; ch < N_CHANNELS; ch++) {
                    float raw = sim_value(t, ch, &seed_anom, is_anom, anom_start_time);
                    float scaled = (raw - ch_mean[ch]) / (ch_std[ch] + 1e-8f);
                    float input[INPUT_DIM] = { scaled, 1.0f };
                    float h_new[HIDDEN_DIM];

                    CFC_CELL_TERNARY(input, channels[ch].h_state, SIM_DT,
                                     W_gate_packed[ti], b_gate,
                                     W_cand_packed[ti], b_cand,
                                     tau, INPUT_DIM, HIDDEN_DIM, h_new);
                    memcpy(channels[ch].h_state, h_new, sizeof(h_new));
                    channels[ch].sample_count++;
                }

                if (channels[0].sample_count > WARMUP && step >= inject_at) {
                    int tgt = anomalies[ai].channel;
                    float s = score_discriminant(channels[tgt].h_state,
                                                 &channels[tgt].disc);
                    if (!tern_detections[ti][ai] && s < ANOMALY_THRESH) {
                        tern_detections[ti][ai] = 1;
                        tern_det_steps[ti][ai] = step - inject_at;
                    }
                }
            }
        }

        printf("  Ternary threshold=%.2f complete.\n", thresholds[ti]);
    }

    /* ═══════════════════════════════════════════════════════════════════════
     * RESULTS
     * ═══════════════════════════════════════════════════════════════════════ */

    printf("\n═══════════════════════════════════════════════════════════════\n");
    printf("  RESULTS\n");
    printf("═══════════════════════════════════════════════════════════════\n\n");

    /* 1. Normal operation scores */
    printf("1. NORMAL OPERATION — Mean discriminant score per channel\n");
    printf("   (Higher = more normal. Float baseline = 100%%)\n\n");

    printf("   %-8s", "Channel");
    printf("  %-8s", "Float");
    for (int ti = 0; ti < N_THRESHOLDS; ti++)
        printf("  t=%-5.2f", thresholds[ti]);
    printf("\n");

    printf("   %-8s", "--------");
    printf("  %-8s", "--------");
    for (int ti = 0; ti < N_THRESHOLDS; ti++)
        printf("  %-8s", "--------");
    printf("\n");

    for (int ch = 0; ch < N_CHANNELS; ch++) {
        printf("   %-8s  %.4f", ch_names[ch], float_normal_scores[ch]);
        for (int ti = 0; ti < N_THRESHOLDS; ti++) {
            float pct = (float_normal_scores[ch] > 1e-6f)
                ? 100.0f * tern_normal_scores[ti][ch] / float_normal_scores[ch]
                : 0;
            printf("  %.4f", tern_normal_scores[ti][ch]);
        }
        printf("\n");
    }

    /* Averages */
    printf("   %-8s  ", "AVERAGE");
    float float_avg = 0;
    for (int ch = 0; ch < N_CHANNELS; ch++) float_avg += float_normal_scores[ch];
    float_avg /= N_CHANNELS;
    printf("%.4f", float_avg);

    for (int ti = 0; ti < N_THRESHOLDS; ti++) {
        float tern_avg = 0;
        for (int ch = 0; ch < N_CHANNELS; ch++) tern_avg += tern_normal_scores[ti][ch];
        tern_avg /= N_CHANNELS;
        printf("  %.4f", tern_avg);
    }
    printf("\n\n");

    /* Degradation summary */
    printf("   Degradation vs float (lower = more degraded):\n");
    printf("   %-10s", "Threshold");
    printf("  %-10s", "Avg Score");
    printf("  %-10s", "vs Float");
    printf("  %-8s\n", "Status");

    for (int ti = 0; ti < N_THRESHOLDS; ti++) {
        float tern_avg = 0;
        for (int ch = 0; ch < N_CHANNELS; ch++) tern_avg += tern_normal_scores[ti][ch];
        tern_avg /= N_CHANNELS;
        float pct = (float_avg > 1e-6f) ? 100.0f * tern_avg / float_avg : 0;

        const char *status;
        if (pct >= 85.0f) status = "PASS";
        else if (pct >= 50.0f) status = "MARGINAL";
        else status = "FAIL";

        printf("   %-10.2f  %-10.4f  %8.1f%%  %-8s\n",
               thresholds[ti], tern_avg, pct, status);
    }

    /* 2. Hidden state divergence */
    printf("\n2. HIDDEN STATE DIVERGENCE (mean L2 distance, float vs ternary)\n");
    printf("   %-8s", "Channel");
    for (int ti = 0; ti < N_THRESHOLDS; ti++)
        printf("  t=%-5.2f", thresholds[ti]);
    printf("\n");
    printf("   %-8s", "--------");
    for (int ti = 0; ti < N_THRESHOLDS; ti++)
        printf("  %-8s", "--------");
    printf("\n");
    for (int ch = 0; ch < N_CHANNELS; ch++) {
        printf("   %-8s", ch_names[ch]);
        for (int ti = 0; ti < N_THRESHOLDS; ti++)
            printf("  %.4f  ", h_divergence[ti][ch]);
        printf("\n");
    }

    /* 3. Anomaly detection */
    printf("\n3. ANOMALY DETECTION — Steps to detection (fewer = better)\n");
    printf("   (-1 = missed)\n\n");

    for (int ai = 0; ai < 3; ai++) {
        printf("   Test %d: %s\n", ai + 1, anomalies[ai].desc);
        printf("     Float: %s",
               float_detections[ai] ? "" : "MISSED");
        if (float_detections[ai])
            printf("step %d (%.0f sec)", float_det_steps[ai],
                   float_det_steps[ai] * SIM_DT);
        printf("\n");

        for (int ti = 0; ti < N_THRESHOLDS; ti++) {
            printf("     t=%.2f: %s", thresholds[ti],
                   tern_detections[ti][ai] ? "" : "MISSED");
            if (tern_detections[ti][ai])
                printf("step %d (%.0f sec)", tern_det_steps[ti][ai],
                       tern_det_steps[ti][ai] * SIM_DT);

            /* Compare to float */
            if (float_detections[ai] && tern_detections[ti][ai]) {
                int diff = tern_det_steps[ti][ai] - float_det_steps[ai];
                if (diff == 0)
                    printf("  [TIE]");
                else if (diff > 0)
                    printf("  [%.0fs slower]", diff * SIM_DT);
                else
                    printf("  [%.0fs faster]", -diff * SIM_DT);
            } else if (float_detections[ai] && !tern_detections[ti][ai]) {
                printf("  [REGRESSION]");
            } else if (!float_detections[ai] && tern_detections[ti][ai]) {
                printf("  [IMPROVEMENT]");
            }
            printf("\n");
        }
        printf("\n");
    }

    /* 4. Execution benchmark — THREE paths */
    printf("4. EXECUTION BENCHMARK — Three execution strategies\n\n");
    printf("   Strategy A: Original float weights (0.1, 0.8, etc.) via GEMM\n");
    printf("   Strategy B: Packed 2-bit ternary via branch-per-weight dot product\n");
    printf("   Strategy C: Ternary-as-float (+1.0, 0.0, -1.0) via GEMM (FPU path)\n");
    printf("               Store packed 2-bit, expand once at load, execute as float.\n");
    printf("               FPU doesn't care that weights are {-1,0,+1}.\n\n");
    {
        int M = 100000;
        float h_state[HIDDEN_DIM] = {0};
        float input[INPUT_DIM] = {0.5f, 1.0f};
        float h_new[HIDDEN_DIM];

        /* Strategy A: Original float GEMM */
        struct timeval t0, t1;
        gettimeofday(&t0, NULL);
        for (int k = 0; k < M; k++) {
            CFC_CELL_GENERIC(input, h_state, SIM_DT,
                             W_gate_float, b_gate, W_cand_float, b_cand,
                             tau, 0, INPUT_DIM, HIDDEN_DIM, h_new);
            memcpy(h_state, h_new, sizeof(h_new));
        }
        gettimeofday(&t1, NULL);
        double us_float = (double)(t1.tv_sec - t0.tv_sec) * 1e6
                        + (double)(t1.tv_usec - t0.tv_usec);

        /* Strategy B: Packed ternary branch-per-weight */
        memset(h_state, 0, sizeof(h_state));
        gettimeofday(&t0, NULL);
        for (int k = 0; k < M; k++) {
            CFC_CELL_TERNARY(input, h_state, SIM_DT,
                             W_gate_packed[0], b_gate,
                             W_cand_packed[0], b_cand,
                             tau, INPUT_DIM, HIDDEN_DIM, h_new);
            memcpy(h_state, h_new, sizeof(h_new));
        }
        gettimeofday(&t1, NULL);
        double us_branch = (double)(t1.tv_sec - t0.tv_sec) * 1e6
                         + (double)(t1.tv_usec - t0.tv_usec);

        /* Strategy C: Ternary-as-float via standard GEMM
         * The key insight: the FPU multiplies by 1.0 in the same cycle
         * as it multiplies by 0.3. The ternary constraint is on VALUES,
         * not INSTRUCTIONS. Store packed 2-bit for compression/auditability,
         * expand to float {-1.0, 0.0, +1.0} at load time, then use the
         * exact same CFC_CELL_GENERIC (standard GEMM) for execution. */
        memset(h_state, 0, sizeof(h_state));
        gettimeofday(&t0, NULL);
        for (int k = 0; k < M; k++) {
            CFC_CELL_GENERIC(input, h_state, SIM_DT,
                             W_gate_tern[1], b_gate,  /* t=0.10 expanded floats */
                             W_cand_tern[1], b_cand,
                             tau, 0, INPUT_DIM, HIDDEN_DIM, h_new);
            memcpy(h_state, h_new, sizeof(h_new));
        }
        gettimeofday(&t1, NULL);
        double us_ternfloat = (double)(t1.tv_sec - t0.tv_sec) * 1e6
                            + (double)(t1.tv_usec - t0.tv_usec);

        double ns_A = (us_float / M) * 1000.0;
        double ns_B = (us_branch / M) * 1000.0;
        double ns_C = (us_ternfloat / M) * 1000.0;

        printf("   Strategy A (float GEMM):       %6.0f ns/step\n", ns_A);
        printf("   Strategy B (ternary branch):   %6.0f ns/step\n", ns_B);
        printf("   Strategy C (ternary-as-float): %6.0f ns/step\n", ns_C);
        printf("\n");
        printf("   B vs A: %.2fx (%s)\n", ns_A / ns_B,
               ns_B < ns_A ? "faster" : "slower");
        printf("   C vs A: %.2fx (%s)\n", ns_A / ns_C,
               ns_C < ns_A ? "faster" : "slower");
        printf("   C vs B: %.2fx (%s)\n", ns_B / ns_C,
               ns_C < ns_B ? "faster" : "slower");
        printf("\n");

        int packed_bytes = 2 * ((HIDDEN_DIM * CONCAT_DIM + 3) / 4);
        int float_bytes = total_weights * (int)sizeof(float);
        printf("   Weight storage:\n");
        printf("     A (original float):    %4d bytes\n", float_bytes);
        printf("     B (packed 2-bit):      %4d bytes (%.0fx compression)\n",
               packed_bytes, (float)float_bytes / packed_bytes);
        printf("     C (packed 2-bit):      %4d bytes at rest, %d bytes expanded\n",
               packed_bytes, float_bytes);
        printf("     C key insight: compress for storage/transfer/audit,\n");
        printf("       expand once to float at load, run at FPU speed.\n");
        printf("\n");
        printf("   The architecture for platforms WITH FPU (Apple, Cortex-M4F):\n");
        printf("     Store: 2-bit packed (%d bytes) — auditable, compressible\n",
               packed_bytes);
        printf("     Load:  expand to float {-1.0, 0.0, +1.0}\n");
        printf("     Run:   standard GEMM at FPU speed\n");
        printf("\n");
        printf("   The architecture for platforms WITHOUT FPU (Cortex-M0):\n");
        printf("     Store: 2-bit packed (%d bytes)\n", packed_bytes);
        printf("     Run:   ternary_dot_chip (add/subtract/skip, no multiply)\n");
    }

    /* ═══════════════════════════════════════════════════════════════════════
     * VERDICT
     * ═══════════════════════════════════════════════════════════════════════ */

    printf("\n═══════════════════════════════════════════════════════════════\n");
    printf("  VERDICT\n");
    printf("═══════════════════════════════════════════════════════════════\n\n");

    /* Check success criteria for each threshold */
    int any_pass = 0;
    for (int ti = 0; ti < N_THRESHOLDS; ti++) {
        float tern_avg = 0;
        for (int ch = 0; ch < N_CHANNELS; ch++)
            tern_avg += tern_normal_scores[ti][ch];
        tern_avg /= N_CHANNELS;

        float pct = (float_avg > 1e-6f) ? 100.0f * tern_avg / float_avg : 0;

        int det_ok = 1;
        for (int ai = 0; ai < 3; ai++) {
            if (float_detections[ai] && !tern_detections[ti][ai])
                det_ok = 0;
            if (float_detections[ai] && tern_detections[ti][ai]
                && tern_det_steps[ti][ai] > 2 * float_det_steps[ai] + 10)
                det_ok = 0;
        }

        int pass = (pct >= 85.0f) && det_ok;

        printf("   Threshold %.2f: score=%5.1f%% of float, detection=%s → %s\n",
               thresholds[ti], pct, det_ok ? "OK" : "REGRESSED",
               pass ? "PASS" : "FAIL");

        if (pass) any_pass = 1;
    }

    printf("\n");
    if (any_pass) {
        printf("   *** TERNARY QUANTIZATION VALIDATED ***\n");
        printf("   At least one threshold preserves detection quality.\n");
        printf("   The bridge between verified ternary primitives and CfC detection\n");
        printf("   is viable. Proceed to Experiment 2 (full composition).\n");
    } else {
        /* Check kill criteria */
        int any_above_50 = 0;
        for (int ti = 0; ti < N_THRESHOLDS; ti++) {
            float tern_avg = 0;
            for (int ch = 0; ch < N_CHANNELS; ch++)
                tern_avg += tern_normal_scores[ti][ch];
            tern_avg /= N_CHANNELS;
            float pct = (float_avg > 1e-6f) ? 100.0f * tern_avg / float_avg : 0;
            if (pct >= 50.0f) any_above_50 = 1;
        }

        if (!any_above_50) {
            printf("   *** TERNARY QUANTIZATION FALSIFIED ***\n");
            printf("   Kill criteria hit: no threshold preserves >50%% quality.\n");
            printf("   Float CfC is the product. Document honestly.\n");
        } else {
            printf("   *** TERNARY QUANTIZATION MARGINAL ***\n");
            printf("   Some quality preserved but below 85%% target.\n");
            printf("   Consider: adjusted thresholds, mixed precision, or\n");
            printf("   accept the degradation if ternary auditability is worth it.\n");
        }
    }

    printf("\n");
    return 0;
}
