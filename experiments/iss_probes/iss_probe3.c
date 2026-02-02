/*
 * ISS Falsification Probe 3: Pre-Scaling Validation (v1 vs v2)
 *
 * Probes 1-2 found:
 *   - CMG hidden state DEGENERATE: H-Std=0.0007 (trapped at ~0.5014)
 *   - CMG null-discriminant scored 0.799 (discriminant can't distinguish)
 *   - Tau ablation FAILED under constant dt=10s
 *   - CfC showed NO advantage over simple 3-sigma threshold
 *
 * Root cause: Input scale problem. CMG at ~0.001g contributes 0.16%
 * of gate pre-activation. CfC can't see its inputs.
 *
 * v2 fix: Frozen per-channel pre-scaling from calibration phase.
 * Maps raw sensor range to CfC useful range (~[-3, +3]).
 *
 * This probe re-runs ALL critical tests from probes 1-2 with v2
 * pre-scaling, reporting v1 vs v2 side-by-side.
 *
 * Tests:
 *   1. HIDDEN STATE DYNAMICS: H-Std with/without pre-scaling
 *   2. NULL DISCRIMINANT: Does pre-scaling kill the 0.799 artifact?
 *   3. TAU ABLATION: Does pre-scaling make tau matter?
 *   4. MAGNITUDE SWEEP: Detection threshold with pre-scaling
 *   5. CfC vs 3-SIGMA: Honest head-to-head per anomaly type
 *   6. SAME-SUBSYSTEM: Can v2 tell CMG1 from CMG4?
 *
 * Compile:
 *   cc -O2 -I include -I include/chips experiments/iss_probes/iss_probe3.c -lm -o experiments/iss_probes/iss_probe3
 *
 * Created by: Tripp + Manus (v2 validation)
 * Date: February 2026
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "onnx_shapes.h"
#include "chips/cfc_cell_chip.h"
#include "chips/activation_chip.h"

#define INPUT_DIM    2
#define HIDDEN_DIM   8
#define CONCAT_DIM   (INPUT_DIM + HIDDEN_DIM)
#define N_PCS        5
#define WARMUP       20
#define MAX_SAMPLES  500
#define POWER_ITERS  20
#define MEAN_WEIGHT  0.3f
#define PCA_WEIGHT   0.7f
#define THRESH       0.35f

/* ═══════════════════════════════════════════════════════════════════════════
 * Weights — identical to iss_telemetry.c
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

/* Tau configurations for ablation */
static const float tau_iss[HIDDEN_DIM] = {
    5.0f, 15.0f, 45.0f, 120.0f, 10.0f, 30.0f, 90.0f, 600.0f
};
static const float tau_keystroke[HIDDEN_DIM] = {
    0.05f, 0.10f, 0.20f, 0.50f, 0.05f, 0.15f, 0.30f, 0.80f
};
static const float tau_constant[HIDDEN_DIM] = {
    30.0f, 30.0f, 30.0f, 30.0f, 30.0f, 30.0f, 30.0f, 30.0f
};

/* ═══════════════════════════════════════════════════════════════════════════
 * Pre-scaling calibration struct (matches iss_telemetry.c v2)
 * ═══════════════════════════════════════════════════════════════════════════ */

typedef struct {
    float input_mean;
    float input_std;
    float dt_mean;
    float sigma3_hi;
    float sigma3_lo;
    int calibrated;
} Calibration;

/* ═══════════════════════════════════════════════════════════════════════════ */

typedef struct {
    float mean[HIDDEN_DIM];
    float dim_std[HIDDEN_DIM];
    float pcs[N_PCS][HIDDEN_DIM];
    float pc_mean[N_PCS];
    float pc_std[N_PCS];
    int valid;
} Discriminant;

/* RNG */
static float randf(unsigned int *s) {
    *s = *s * 1103515245u + 12345u;
    return (float)(*s & 0x7FFFFFFFu) / (float)0x7FFFFFFFu;
}
static float gaussf(unsigned int *s) {
    float u1 = randf(s) + 1e-10f;
    float u2 = randf(s);
    return sqrtf(-2.0f * logf(u1)) * cosf(6.28318530718f * u2);
}

static float dotf(const float *a, const float *b, int n) {
    float s = 0;
    for (int i = 0; i < n; i++) s += a[i] * b[i];
    return s;
}
static float vec_normalize(float *v, int n) {
    float norm = 0;
    for (int i = 0; i < n; i++) norm += v[i] * v[i];
    norm = sqrtf(norm);
    if (norm > 1e-10f) for (int i = 0; i < n; i++) v[i] /= norm;
    return norm;
}

/* CfC step — v1 (no pre-scaling) */
static void cfc_step_v1(float value, float dt, float *h, const float *tau) {
    float input[INPUT_DIM] = { value, dt };
    float h_new[HIDDEN_DIM];
    CFC_CELL_GENERIC(input, h, dt, W_gate, b_gate, W_cand, b_cand,
                     tau, 0, INPUT_DIM, HIDDEN_DIM, h_new);
    memcpy(h, h_new, HIDDEN_DIM * sizeof(float));
}

/* CfC step — v2 (with pre-scaling) */
static void cfc_step_v2(float value, float dt, float *h, const float *tau,
                        const Calibration *cal) {
    float scaled_val = cal->calibrated
        ? (value - cal->input_mean) / (cal->input_std + 1e-8f)
        : value;
    float scaled_dt = (cal->calibrated && cal->dt_mean > 1e-6f)
        ? dt / cal->dt_mean
        : dt;
    float input[INPUT_DIM] = { scaled_val, scaled_dt };
    float h_new[HIDDEN_DIM];
    CFC_CELL_GENERIC(input, h, dt, W_gate, b_gate, W_cand, b_cand,
                     tau, 0, INPUT_DIM, HIDDEN_DIM, h_new);
    memcpy(h, h_new, HIDDEN_DIM * sizeof(float));
}

/* Learn discriminant (identical to probe1/probe2) */
static void learn_disc(const float samples[][HIDDEN_DIM], int n, Discriminant *d) {
    memset(d, 0, sizeof(*d));
    if (n < 5) { d->valid = 0; return; }

    for (int i = 0; i < HIDDEN_DIM; i++) {
        float s = 0;
        for (int t = 0; t < n; t++) s += samples[t][i];
        d->mean[i] = s / n;
    }
    for (int i = 0; i < HIDDEN_DIM; i++) {
        float s = 0;
        for (int t = 0; t < n; t++) {
            float dd = samples[t][i] - d->mean[i];
            s += dd * dd;
        }
        d->dim_std[i] = sqrtf(s / n + 1e-8f);
    }

    float centered[n][HIDDEN_DIM];
    for (int t = 0; t < n; t++)
        for (int i = 0; i < HIDDEN_DIM; i++)
            centered[t][i] = samples[t][i] - d->mean[i];

    for (int pc = 0; pc < N_PCS; pc++) {
        float v[HIDDEN_DIM];
        for (int i = 0; i < HIDDEN_DIM; i++)
            v[i] = centered[0][i] + 0.01f * (i + 1);
        vec_normalize(v, HIDDEN_DIM);

        for (int iter = 0; iter < POWER_ITERS; iter++) {
            float v_new[HIDDEN_DIM] = {0};
            for (int t = 0; t < n; t++) {
                float proj = dotf(centered[t], v, HIDDEN_DIM);
                for (int i = 0; i < HIDDEN_DIM; i++)
                    v_new[i] += proj * centered[t][i];
            }
            for (int i = 0; i < HIDDEN_DIM; i++) v_new[i] /= n;
            memcpy(v, v_new, sizeof(v));
            vec_normalize(v, HIDDEN_DIM);
        }
        memcpy(d->pcs[pc], v, sizeof(float) * HIDDEN_DIM);

        float ps = 0, ps2 = 0;
        for (int t = 0; t < n; t++) {
            float p = dotf(centered[t], v, HIDDEN_DIM);
            ps += p; ps2 += p * p;
        }
        d->pc_mean[pc] = ps / n;
        float var = ps2 / n - d->pc_mean[pc] * d->pc_mean[pc];
        d->pc_std[pc] = sqrtf(var > 0 ? var : 1e-8f);

        for (int t = 0; t < n; t++) {
            float p = dotf(centered[t], v, HIDDEN_DIM);
            for (int i = 0; i < HIDDEN_DIM; i++)
                centered[t][i] -= p * v[i];
        }
    }
    d->valid = 1;
}

/* Hybrid score (identical to probe1) */
static float score_hybrid(const float *h, const Discriminant *d) {
    if (!d->valid) return 0.5f;
    float md = 0, centered[HIDDEN_DIM];
    for (int i = 0; i < HIDDEN_DIM; i++) {
        centered[i] = h[i] - d->mean[i];
        float z = centered[i] / (d->dim_std[i] + 1e-8f);
        md += z * z;
    }
    md /= HIDDEN_DIM;
    float ms = SIGMOID_CHIP(2.0f - md);

    float pd = 0;
    for (int pc = 0; pc < N_PCS; pc++) {
        float proj = dotf(centered, d->pcs[pc], HIDDEN_DIM);
        float z = (proj - d->pc_mean[pc]) / (d->pc_std[pc] + 1e-8f);
        pd += z * z;
    }
    pd /= N_PCS;
    float ps = SIGMOID_CHIP(2.0f - pd);

    return MEAN_WEIGHT * ms + PCA_WEIGHT * ps;
}

/* ═══════════════════════════════════════════════════════════════════════════
 * Telemetry Generators (identical to probe1)
 * ═══════════════════════════════════════════════════════════════════════════ */

#define ORBITAL_PERIOD 5520.0f
#define DT 10.0f

static float gen_cmg(float t, int ch_offset, unsigned int *seed) {
    float phase = 6.28318530718f * t / ORBITAL_PERIOD;
    return 0.001f + 0.0001f * ch_offset + 0.0002f * sinf(phase)
         + 0.0002f * gaussf(seed);
}

static float gen_coolant(float t, int loop_b, unsigned int *seed) {
    float phase = 6.28318530718f * t / ORBITAL_PERIOD;
    return 15.0f + (loop_b ? 2.0f : 0.0f) + 8.0f * sinf(phase)
         + 0.5f * gaussf(seed);
}

static float gen_cabin_p(float t, unsigned int *seed) {
    float phase = 6.28318530718f * t / ORBITAL_PERIOD;
    return 101.3f + 0.02f * sinf(phase) + 0.05f * gaussf(seed);
}

static float gen_o2(float t, unsigned int *seed) {
    float cdra = fmodf(t, 900.0f) / 900.0f;
    return 21.3f + 0.15f * (cdra - 0.5f) + 0.1f * gaussf(seed);
}

static float gen_channel(int ch, float t, unsigned int *seed) {
    switch (ch) {
    case 0: return gen_cmg(t, 0, seed);
    case 1: return gen_cmg(t, 1, seed);
    case 2: return gen_cmg(t, 2, seed);
    case 3: return gen_cmg(t, 3, seed);
    case 4: return gen_coolant(t, 0, seed);
    case 5: return gen_coolant(t, 1, seed);
    case 6: return gen_cabin_p(t, seed);
    case 7: return gen_o2(t, seed);
    default: return gaussf(seed);
    }
}

/* Add anomaly with configurable magnitude (1.0 = probe1 levels) */
static float add_anomaly(float value, int ch, int type, float dt_anom, float mag) {
    float ramp = dt_anom / 3000.0f;
    if (ramp > 1.0f) ramp = 1.0f;

    switch (type) {
    case 0: /* step */
        if (ch < 4) return value + 0.005f * mag;
        if (ch < 6) return value + 5.0f * mag;
        if (ch == 6) return value - 0.5f * mag;
        return value - 1.0f * mag;
    case 1: /* ramp */
        if (ch < 4) return value + 0.008f * ramp * mag;
        if (ch < 6) return value + 8.0f * ramp * mag;
        if (ch == 6) return value - 1.0f * ramp * mag;
        return value - 2.0f * ramp * mag;
    case 2: /* oscillation shift */
        {
            float osc = sinf(6.28318530718f * dt_anom / 200.0f);
            if (ch < 4) return value + 0.003f * osc * mag;
            if (ch < 6) return value + 3.0f * osc * mag;
            if (ch == 6) return value + 0.3f * osc * mag;
            return value + 0.5f * osc * mag;
        }
    }
    return value;
}

/* ═══════════════════════════════════════════════════════════════════════════
 * Calibrate a channel: stream 1 orbit, compute mean/std
 * ═══════════════════════════════════════════════════════════════════════════ */
static Calibration calibrate_channel(int ch, unsigned int *seed) {
    Calibration cal;
    memset(&cal, 0, sizeof(cal));
    int cal_steps = (int)(ORBITAL_PERIOD / DT);
    double sum = 0, sum2 = 0;

    unsigned int seed_copy = *seed;
    for (int step = 0; step < cal_steps; step++) {
        float t = step * DT;
        float value = gen_channel(ch, t, seed);
        sum += (double)value;
        sum2 += (double)value * (double)value;
    }

    cal.input_mean = (float)(sum / cal_steps);
    double var = sum2 / cal_steps - (sum / cal_steps) * (sum / cal_steps);
    cal.input_std = sqrtf((float)(var > 0 ? var : 1e-8));
    cal.dt_mean = DT;
    cal.sigma3_hi = cal.input_mean + 3.0f * cal.input_std;
    cal.sigma3_lo = cal.input_mean - 3.0f * cal.input_std;
    cal.calibrated = 1;
    return cal;
}

/* ═══════════════════════════════════════════════════════════════════════════
 * Collect enrollment samples — v1 and v2 variants
 * ═══════════════════════════════════════════════════════════════════════════ */

static int collect_v1(int ch, int n_steps, unsigned int *seed,
                      const float *tau_vec,
                      float samples[][HIDDEN_DIM], int max_samples) {
    float h[HIDDEN_DIM] = {0};
    int n = 0;
    for (int step = 0; step < n_steps; step++) {
        float t = step * DT;
        float value = gen_channel(ch, t, seed);
        cfc_step_v1(value, DT, h, tau_vec);
        if (step > WARMUP && n < max_samples) {
            memcpy(samples[n], h, sizeof(float) * HIDDEN_DIM);
            n++;
        }
    }
    return n;
}

static int collect_v2(int ch, int n_steps, unsigned int *seed,
                      const float *tau_vec, const Calibration *cal,
                      float samples[][HIDDEN_DIM], int max_samples) {
    float h[HIDDEN_DIM] = {0};
    int n = 0;
    for (int step = 0; step < n_steps; step++) {
        float t = step * DT;
        float value = gen_channel(ch, t, seed);
        cfc_step_v2(value, DT, h, tau_vec, cal);
        if (step > WARMUP && n < max_samples) {
            memcpy(samples[n], h, sizeof(float) * HIDDEN_DIM);
            n++;
        }
    }
    return n;
}

/* Compute H-Std: average per-dimension std of hidden states */
static float compute_h_std(const float samples[][HIDDEN_DIM], int n) {
    if (n < 2) return 0;
    float total_std = 0;
    for (int d = 0; d < HIDDEN_DIM; d++) {
        double sum = 0, sum2 = 0;
        for (int t = 0; t < n; t++) {
            sum += (double)samples[t][d];
            sum2 += (double)samples[t][d] * (double)samples[t][d];
        }
        double mean = sum / n;
        double var = sum2 / n - mean * mean;
        total_std += sqrtf((float)(var > 0 ? var : 0));
    }
    return total_std / HIDDEN_DIM;
}

/* ═══════════════════════════════════════════════════════════════════════════ */

int main(void) {
    printf("================================================================\n");
    printf("  ISS Falsification Probe 3: Pre-Scaling Validation (v1 vs v2)\n");
    printf("================================================================\n\n");

    int enroll_steps = 1104;   /* 2 orbits */
    int test_steps = 552;      /* 1 orbit */
    int N_RUNS = 20;

    int probe_channels[] = {0, 4, 6, 7};
    const char *probe_names[] = {"CMG1-vib", "CoolA-T", "CabinP", "O2-PP"};

    /* Pre-compute calibrations for all channels */
    Calibration cals[8];
    for (int ch = 0; ch < 8; ch++) {
        unsigned int seed = 42;
        cals[ch] = calibrate_channel(ch, &seed);
    }

    printf("  Pre-scaling calibrations:\n");
    printf("    %-10s  %12s  %12s\n", "Channel", "Mean", "Std");
    const char *all_names[] = {"CMG1-vib", "CMG2-vib", "CMG3-vib", "CMG4-vib",
                               "CoolA-T", "CoolB-T", "CabinP", "O2-PP"};
    for (int ch = 0; ch < 8; ch++) {
        printf("    %-10s  %12.6f  %12.6f\n",
               all_names[ch], cals[ch].input_mean, cals[ch].input_std);
    }

    /* ══════════════════════════════════════════════════════════════════════
     * TEST 1: HIDDEN STATE DYNAMICS — v1 vs v2
     *
     * The core diagnostic. CMG H-Std went from 0.0007 (v1) to what?
     * ══════════════════════════════════════════════════════════════════════ */
    printf("\n");
    printf("TEST 1: HIDDEN STATE DYNAMICS (H-Std)\n");
    printf("  v1 CMG H-Std was 0.0007 (degenerate). v2 target: >0.01.\n\n");

    printf("  %-10s  %10s  %10s  %10s  %s\n",
           "Channel", "v1 H-Std", "v2 H-Std", "Ratio", "Status");
    printf("  %-10s  %10s  %10s  %10s  %s\n",
           "----------", "----------", "----------", "----------", "----------");

    for (int ci = 0; ci < 4; ci++) {
        int ch = probe_channels[ci];
        float v1_std = 0, v2_std = 0;

        for (int run = 0; run < N_RUNS; run++) {
            unsigned int seed1 = 42 + run * 97;
            unsigned int seed2 = 42 + run * 97;

            float s1[MAX_SAMPLES][HIDDEN_DIM], s2[MAX_SAMPLES][HIDDEN_DIM];
            int n1 = collect_v1(ch, enroll_steps, &seed1, tau_iss, s1, MAX_SAMPLES);
            int n2 = collect_v2(ch, enroll_steps, &seed2, tau_iss, &cals[ch],
                                s2, MAX_SAMPLES);
            v1_std += compute_h_std((const float (*)[HIDDEN_DIM])s1, n1);
            v2_std += compute_h_std((const float (*)[HIDDEN_DIM])s2, n2);
        }

        v1_std /= N_RUNS;
        v2_std /= N_RUNS;
        float ratio = (v1_std > 1e-10f) ? v2_std / v1_std : 0;

        const char *status;
        if (v2_std > 0.01f) status = "PASS (exercising)";
        else if (v2_std > 0.001f) status = "MARGINAL";
        else status = "FAIL (still degenerate)";

        printf("  %-10s  %10.6f  %10.6f  %10.1fx  %s\n",
               probe_names[ci], v1_std, v2_std, ratio, status);
    }

    /* ══════════════════════════════════════════════════════════════════════
     * TEST 2: NULL DISCRIMINANT — v1 vs v2
     *
     * Probe 1 found: noise-enrolled scored 0.799 on CMG (v1).
     * With pre-scaling, it should drop (noise patterns don't match
     * pre-scaled normal distribution).
     * ══════════════════════════════════════════════════════════════════════ */
    printf("\nTEST 2: NULL DISCRIMINANT (noise-enrolled scores on real data)\n");
    printf("  v1 CMG noise-disc scored 0.799. v2 target: <0.50.\n\n");

    printf("  %-10s  %12s  %12s  %12s  %12s\n",
           "Channel", "v1 real", "v1 noise", "v2 real", "v2 noise");
    printf("  %-10s  %12s  %12s  %12s  %12s\n",
           "----------", "------------", "------------", "------------", "------------");

    for (int ci = 0; ci < 4; ci++) {
        int ch = probe_channels[ci];
        float v1_real = 0, v1_noise = 0, v2_real = 0, v2_noise = 0;

        for (int run = 0; run < N_RUNS; run++) {
            /* v1 real-enrolled discriminant */
            Discriminant disc_v1_real;
            {
                unsigned int seed = 42 + run * 97;
                float samples[MAX_SAMPLES][HIDDEN_DIM];
                int ns = collect_v1(ch, enroll_steps, &seed, tau_iss,
                                    samples, MAX_SAMPLES);
                learn_disc(samples, ns, &disc_v1_real);
            }

            /* v1 noise-enrolled discriminant */
            Discriminant disc_v1_noise;
            {
                unsigned int seed = 77777 + run * 53;
                float h[HIDDEN_DIM] = {0};
                float samples[MAX_SAMPLES][HIDDEN_DIM];
                int n = 0;
                for (int step = 0; step < enroll_steps; step++) {
                    float value = gaussf(&seed);
                    cfc_step_v1(value, DT, h, tau_iss);
                    if (step > WARMUP && n < MAX_SAMPLES) {
                        memcpy(samples[n], h, sizeof(float) * HIDDEN_DIM);
                        n++;
                    }
                }
                learn_disc(samples, n, &disc_v1_noise);
            }

            /* v2 real-enrolled discriminant */
            Discriminant disc_v2_real;
            {
                unsigned int seed = 42 + run * 97;
                float samples[MAX_SAMPLES][HIDDEN_DIM];
                int ns = collect_v2(ch, enroll_steps, &seed, tau_iss, &cals[ch],
                                    samples, MAX_SAMPLES);
                learn_disc(samples, ns, &disc_v2_real);
            }

            /* v2 noise-enrolled discriminant (noise through pre-scaled CfC) */
            Discriminant disc_v2_noise;
            {
                unsigned int seed = 77777 + run * 53;
                float h[HIDDEN_DIM] = {0};
                float samples[MAX_SAMPLES][HIDDEN_DIM];
                int n = 0;
                for (int step = 0; step < enroll_steps; step++) {
                    float value = gaussf(&seed);
                    /* Noise gets pre-scaled too (which makes it huge) */
                    cfc_step_v2(value, DT, h, tau_iss, &cals[ch]);
                    if (step > WARMUP && n < MAX_SAMPLES) {
                        memcpy(samples[n], h, sizeof(float) * HIDDEN_DIM);
                        n++;
                    }
                }
                learn_disc(samples, n, &disc_v2_noise);
            }

            /* Score ISS test data against all 4 discriminants */
            unsigned int ts1 = 9999 + run * 131;
            unsigned int ts2 = ts1, ts3 = ts1, ts4 = ts1;

            float h1[HIDDEN_DIM]={0}, h2[HIDDEN_DIM]={0};
            float h3[HIDDEN_DIM]={0}, h4[HIDDEN_DIM]={0};
            float s1=0, s2=0, s3=0, s4=0;
            int scored = 0;

            for (int step = 0; step < test_steps; step++) {
                float t = step * DT;
                float val1 = gen_channel(ch, t, &ts1);
                float val2 = gen_channel(ch, t, &ts2);
                float val3 = gen_channel(ch, t, &ts3);
                float val4 = gen_channel(ch, t, &ts4);

                cfc_step_v1(val1, DT, h1, tau_iss);
                cfc_step_v1(val2, DT, h2, tau_iss);
                cfc_step_v2(val3, DT, h3, tau_iss, &cals[ch]);
                cfc_step_v2(val4, DT, h4, tau_iss, &cals[ch]);

                if (step > WARMUP) {
                    s1 += score_hybrid(h1, &disc_v1_real);
                    s2 += score_hybrid(h2, &disc_v1_noise);
                    s3 += score_hybrid(h3, &disc_v2_real);
                    s4 += score_hybrid(h4, &disc_v2_noise);
                    scored++;
                }
            }
            v1_real += s1/scored;
            v1_noise += s2/scored;
            v2_real += s3/scored;
            v2_noise += s4/scored;
        }

        printf("  %-10s  %12.4f  %12.4f  %12.4f  %12.4f\n",
               probe_names[ci],
               v1_real/N_RUNS, v1_noise/N_RUNS,
               v2_real/N_RUNS, v2_noise/N_RUNS);
    }

    /* ══════════════════════════════════════════════════════════════════════
     * TEST 3: TAU ABLATION WITH PRE-SCALING
     *
     * The original tau ablation failed because inputs were too small to
     * exercise any nonlinearity. With pre-scaling, does tau matter?
     * ══════════════════════════════════════════════════════════════════════ */
    printf("\nTEST 3: TAU ABLATION (v2 with pre-scaling)\n");
    printf("  Probe 1 found: all tau configs identical under v1.\n");
    printf("  With pre-scaling, ISS tau should outperform keystroke tau.\n\n");

    const float *tau_sets[] = {tau_iss, tau_keystroke, tau_constant};
    const char *tau_labels[] = {"ISS (5-600s)", "Keystroke (0.05-0.8s)", "Constant (30s)"};

    printf("  %-24s  %10s  %10s  %10s  %s\n",
           "Tau Config", "Normal", "Step", "Ramp", "Detected");
    printf("  %-24s  %10s  %10s  %10s  %s\n",
           "------------------------", "----------", "----------", "----------", "----------");

    for (int ti = 0; ti < 3; ti++) {
        const float *tau_vec = tau_sets[ti];
        float normal_sum = 0, step_sum = 0, ramp_sum = 0;
        int step_det = 0, ramp_det = 0;

        for (int run = 0; run < N_RUNS; run++) {
            /* Enroll with v2 */
            unsigned int enroll_seed = 42 + run * 97;
            float samples[MAX_SAMPLES][HIDDEN_DIM];
            int ns = collect_v2(0, enroll_steps, &enroll_seed, tau_vec,
                                &cals[0], samples, MAX_SAMPLES);
            Discriminant disc;
            learn_disc(samples, ns, &disc);

            /* Normal test */
            {
                unsigned int seed = 9999 + run * 131;
                float h[HIDDEN_DIM] = {0};
                float ss = 0; int scored = 0;
                for (int step = 0; step < test_steps; step++) {
                    float t = step * DT;
                    float v = gen_channel(0, t, &seed);
                    cfc_step_v2(v, DT, h, tau_vec, &cals[0]);
                    if (step > WARMUP) { ss += score_hybrid(h, &disc); scored++; }
                }
                normal_sum += ss / scored;
            }

            /* Step anomaly at midpoint */
            {
                unsigned int seed = 9999 + run * 131;
                float h[HIDDEN_DIM] = {0};
                float final_score = 0.5f;
                for (int step = 0; step < test_steps; step++) {
                    float t = step * DT;
                    float v = gen_channel(0, t, &seed);
                    if (step >= test_steps/2) {
                        float dt_anom = (step - test_steps/2) * DT;
                        v = add_anomaly(v, 0, 0, dt_anom, 1.0f);
                    }
                    cfc_step_v2(v, DT, h, tau_vec, &cals[0]);
                    final_score = score_hybrid(h, &disc);
                }
                step_sum += final_score;
                if (final_score < THRESH) step_det++;
            }

            /* Ramp anomaly at midpoint */
            {
                unsigned int seed = 9999 + run * 131;
                float h[HIDDEN_DIM] = {0};
                float final_score = 0.5f;
                for (int step = 0; step < test_steps; step++) {
                    float t = step * DT;
                    float v = gen_channel(0, t, &seed);
                    if (step >= test_steps/2) {
                        float dt_anom = (step - test_steps/2) * DT;
                        v = add_anomaly(v, 0, 1, dt_anom, 1.0f);
                    }
                    cfc_step_v2(v, DT, h, tau_vec, &cals[0]);
                    final_score = score_hybrid(h, &disc);
                }
                ramp_sum += final_score;
                if (final_score < THRESH) ramp_det++;
            }
        }

        printf("  %-24s  %10.4f  %10.4f  %10.4f  S:%d/%d R:%d/%d\n",
               tau_labels[ti],
               normal_sum/N_RUNS, step_sum/N_RUNS, ramp_sum/N_RUNS,
               step_det, N_RUNS, ramp_det, N_RUNS);
    }

    /* ══════════════════════════════════════════════════════════════════════
     * TEST 4: MAGNITUDE SWEEP (v2)
     *
     * At what anomaly magnitude does v2 detection fail?
     * Test CMG step anomaly at 100%, 50%, 20%, 10%, 5%, 2%, 1%.
     * Compare CfC detection rate vs 3-sigma detection rate.
     * ══════════════════════════════════════════════════════════════════════ */
    printf("\nTEST 4: MAGNITUDE SWEEP (CMG1 step, CfC vs 3-sigma)\n");
    printf("  At what magnitude does each method fail?\n\n");

    float magnitudes[] = {1.0f, 0.5f, 0.2f, 0.1f, 0.05f, 0.02f, 0.01f};
    int n_mags = 7;

    printf("  %-8s  %10s  %10s  %10s  %10s  %s\n",
           "Mag", "CfC score", "CfC det", "3sig det", "Winner", "Note");
    printf("  %-8s  %10s  %10s  %10s  %10s  %s\n",
           "--------", "----------", "----------", "----------", "----------", "----------");

    for (int mi = 0; mi < n_mags; mi++) {
        float mag = magnitudes[mi];
        int cfc_det = 0, sig_det = 0;
        float cfc_score_sum = 0;

        for (int run = 0; run < N_RUNS; run++) {
            /* Enroll v2 */
            unsigned int enroll_seed = 42 + run * 97;
            float samples[MAX_SAMPLES][HIDDEN_DIM];
            int ns = collect_v2(0, enroll_steps, &enroll_seed, tau_iss,
                                &cals[0], samples, MAX_SAMPLES);
            Discriminant disc;
            learn_disc(samples, ns, &disc);

            /* Test with anomaly — per-sample scoring */
            unsigned int seed = 9999 + run * 131;
            float h[HIDDEN_DIM] = {0};
            int cfc_found = 0, sig_found = 0;

            for (int step = 0; step < test_steps; step++) {
                float t = step * DT;
                float v = gen_channel(0, t, &seed);
                float raw_v = v;
                if (step >= test_steps/2) {
                    float dt_anom = (step - test_steps/2) * DT;
                    v = add_anomaly(v, 0, 0, dt_anom, mag);
                    raw_v = v;
                }
                cfc_step_v2(v, DT, h, tau_iss, &cals[0]);

                if (step > WARMUP && step >= test_steps/2) {
                    float s = score_hybrid(h, &disc);
                    cfc_score_sum += s;
                    if (s < THRESH && !cfc_found) cfc_found = 1;
                    /* 3-sigma check on raw value */
                    if ((raw_v > cals[0].sigma3_hi || raw_v < cals[0].sigma3_lo)
                        && !sig_found)
                        sig_found = 1;
                }
            }
            if (cfc_found) cfc_det++;
            if (sig_found) sig_det++;
        }

        const char *winner = "TIE";
        if (cfc_det > sig_det) winner = "CfC";
        else if (sig_det > cfc_det) winner = "3-sigma";

        printf("  %-8.0f%%  %10.4f  %7d/%d  %7d/%d  %10s\n",
               mag * 100,
               cfc_score_sum / (N_RUNS * (test_steps/2 - WARMUP)),
               cfc_det, N_RUNS, sig_det, N_RUNS, winner);
    }

    /* ══════════════════════════════════════════════════════════════════════
     * TEST 5: CfC vs 3-SIGMA HEAD-TO-HEAD (multiple anomaly types)
     *
     * For each anomaly type on multiple channels, compare detection.
     * ══════════════════════════════════════════════════════════════════════ */
    printf("\nTEST 5: CfC vs 3-SIGMA HEAD-TO-HEAD\n");
    printf("  Which anomaly types does CfC catch that 3-sigma misses?\n\n");

    typedef struct {
        const char *label;
        int ch;
        int type;
        float mag;
    } AnomalyCase;

    AnomalyCase cases[] = {
        {"CMG1 step 100%%",    0, 0, 1.0f},
        {"CMG1 step 20%%",     0, 0, 0.2f},
        {"CMG1 ramp 100%%",    0, 1, 1.0f},
        {"CMG1 ramp 20%%",     0, 1, 0.2f},
        {"CMG1 osc 100%%",     0, 2, 1.0f},
        {"CoolA step 100%%",   4, 0, 1.0f},
        {"CoolA osc 100%%",    4, 2, 1.0f},
        {"CoolA osc 20%%",     4, 2, 0.2f},
        {"CabinP ramp 100%%",  6, 1, 1.0f},
        {"O2 step 100%%",      7, 0, 1.0f},
    };
    int n_cases = (int)(sizeof(cases) / sizeof(cases[0]));

    printf("  %-22s  %8s  %8s  %s\n",
           "Anomaly", "CfC det", "3sig det", "Winner");
    printf("  %-22s  %8s  %8s  %s\n",
           "----------------------", "--------", "--------", "--------");

    for (int ci = 0; ci < n_cases; ci++) {
        int ch = cases[ci].ch;
        int type = cases[ci].type;
        float mag = cases[ci].mag;
        int cfc_det = 0, sig_det = 0;

        for (int run = 0; run < N_RUNS; run++) {
            unsigned int enroll_seed = 42 + run * 97;
            float samples[MAX_SAMPLES][HIDDEN_DIM];
            int ns = collect_v2(ch, enroll_steps, &enroll_seed, tau_iss,
                                &cals[ch], samples, MAX_SAMPLES);
            Discriminant disc;
            learn_disc(samples, ns, &disc);

            unsigned int seed = 9999 + run * 131;
            float h[HIDDEN_DIM] = {0};
            int cf = 0, sf = 0;

            for (int step = 0; step < test_steps; step++) {
                float t = step * DT;
                float v = gen_channel(ch, t, &seed);
                if (step >= test_steps/2) {
                    float dt_anom = (step - test_steps/2) * DT;
                    v = add_anomaly(v, ch, type, dt_anom, mag);
                }
                float raw = v;
                cfc_step_v2(v, DT, h, tau_iss, &cals[ch]);

                if (step > WARMUP && step >= test_steps/2) {
                    if (score_hybrid(h, &disc) < THRESH && !cf) cf = 1;
                    if ((raw > cals[ch].sigma3_hi || raw < cals[ch].sigma3_lo)
                        && !sf) sf = 1;
                }
            }
            if (cf) cfc_det++;
            if (sf) sig_det++;
        }

        const char *winner = "TIE";
        if (cfc_det > sig_det) winner = "CfC WINS";
        else if (sig_det > cfc_det) winner = "3sig WINS";

        printf("  %-22s  %5d/%d  %5d/%d  %s\n",
               cases[ci].label, cfc_det, N_RUNS, sig_det, N_RUNS, winner);
    }

    /* ══════════════════════════════════════════════════════════════════════
     * TEST 6: SAME-SUBSYSTEM DISCRIMINATION (v2)
     *
     * Can v2 tell apart CMG1 from CMG4? They differ by 0.0003g baseline.
     * Pre-scaling should amplify this difference.
     * ══════════════════════════════════════════════════════════════════════ */
    printf("\nTEST 6: SAME-SUBSYSTEM DISCRIMINATION (CMG1 vs CMG4)\n");
    printf("  v1 probe2 found delta=-0.077. Pre-scaling should increase it.\n\n");

    float v1_same = 0, v1_cross = 0, v2_same = 0, v2_cross = 0;

    for (int run = 0; run < N_RUNS; run++) {
        /* v1: enroll on CMG1, test on CMG1 and CMG4 */
        {
            unsigned int seed = 42 + run * 97;
            float samples[MAX_SAMPLES][HIDDEN_DIM];
            int ns = collect_v1(0, enroll_steps, &seed, tau_iss, samples, MAX_SAMPLES);
            Discriminant disc;
            learn_disc(samples, ns, &disc);

            /* Same channel */
            {
                unsigned int ts = 9999 + run * 131;
                float h[HIDDEN_DIM] = {0};
                float ss = 0; int scored = 0;
                for (int step = 0; step < test_steps; step++) {
                    float t = step * DT;
                    cfc_step_v1(gen_channel(0, t, &ts), DT, h, tau_iss);
                    if (step > WARMUP) { ss += score_hybrid(h, &disc); scored++; }
                }
                v1_same += ss / scored;
            }
            /* Cross channel (CMG4) */
            {
                unsigned int ts = 9999 + run * 131;
                float h[HIDDEN_DIM] = {0};
                float ss = 0; int scored = 0;
                for (int step = 0; step < test_steps; step++) {
                    float t = step * DT;
                    cfc_step_v1(gen_channel(3, t, &ts), DT, h, tau_iss);
                    if (step > WARMUP) { ss += score_hybrid(h, &disc); scored++; }
                }
                v1_cross += ss / scored;
            }
        }

        /* v2: enroll on CMG1, test on CMG1 and CMG4 */
        {
            unsigned int seed = 42 + run * 97;
            float samples[MAX_SAMPLES][HIDDEN_DIM];
            int ns = collect_v2(0, enroll_steps, &seed, tau_iss, &cals[0],
                                samples, MAX_SAMPLES);
            Discriminant disc;
            learn_disc(samples, ns, &disc);

            /* Same channel (with CMG1 calibration) */
            {
                unsigned int ts = 9999 + run * 131;
                float h[HIDDEN_DIM] = {0};
                float ss = 0; int scored = 0;
                for (int step = 0; step < test_steps; step++) {
                    float t = step * DT;
                    cfc_step_v2(gen_channel(0, t, &ts), DT, h, tau_iss, &cals[0]);
                    if (step > WARMUP) { ss += score_hybrid(h, &disc); scored++; }
                }
                v2_same += ss / scored;
            }
            /* Cross channel (CMG4 data, but CMG1 calibration) */
            {
                unsigned int ts = 9999 + run * 131;
                float h[HIDDEN_DIM] = {0};
                float ss = 0; int scored = 0;
                for (int step = 0; step < test_steps; step++) {
                    float t = step * DT;
                    /* Use CMG1's calibration on CMG4 data — the offset
                     * difference will be amplified by pre-scaling */
                    cfc_step_v2(gen_channel(3, t, &ts), DT, h, tau_iss, &cals[0]);
                    if (step > WARMUP) { ss += score_hybrid(h, &disc); scored++; }
                }
                v2_cross += ss / scored;
            }
        }
    }

    printf("  %-12s  %10s  %10s  %10s\n",
           "Version", "Same-CMG1", "Cross-CMG4", "Delta");
    printf("  %-12s  %10s  %10s  %10s\n",
           "------------", "----------", "----------", "----------");
    printf("  %-12s  %10.4f  %10.4f  %+10.4f\n",
           "v1 (raw)", v1_same/N_RUNS, v1_cross/N_RUNS,
           v1_cross/N_RUNS - v1_same/N_RUNS);
    printf("  %-12s  %10.4f  %10.4f  %+10.4f\n",
           "v2 (scaled)", v2_same/N_RUNS, v2_cross/N_RUNS,
           v2_cross/N_RUNS - v2_same/N_RUNS);

    /* ══════════════════════════════════════════════════════════════════════
     * SUMMARY
     * ══════════════════════════════════════════════════════════════════════ */
    printf("\n================================================================\n");
    printf("  SUCCESS CRITERIA (from LMM synthesis):\n");
    printf("  [ ] CMG hidden state H-Std > 0.01\n");
    printf("  [ ] Noise-enrolled CMG scores < 0.50\n");
    printf("  [ ] At least one tau config outperforms another\n");
    printf("  [ ] CfC detects at least one anomaly type 3-sigma misses\n");
    printf("  [ ] Same-subsystem delta larger in v2 than v1\n");
    printf("================================================================\n");

    return 0;
}
