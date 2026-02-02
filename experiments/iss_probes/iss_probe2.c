/*
 * ISS Falsification Probe 2: Subtle Anomalies + CMG Null-Disc Deep Dive
 *
 * Probe 1 found:
 *   - Tau ablation FAILED: all tau configs detect equally at current magnitudes
 *   - CMG null-discriminant scored 0.799 (suspiciously high)
 *
 * This probe answers:
 *
 * 1. ANOMALY MAGNITUDE SWEEP: At what anomaly magnitude does detection fail?
 *    Sweep from 100% (probe1 levels) down to 1% across all tau configs.
 *    If tau matters, it should show up at small anomaly magnitudes.
 *
 * 2. SLOW DRIFT vs FAST TRANSIENT: Does tau differentiate between:
 *    - A fast transient (sensor glitch, 1 sample) vs
 *    - A slow drift (bearing wear, over 1000s)?
 *    ISS tau should be better at slow drifts; keystroke tau at fast transients.
 *
 * 3. CMG ATTRACTOR ANALYSIS: Is the CMG hidden state distribution so narrow
 *    that any discriminant works? Compute hidden state variance and
 *    inter-class distance for CMG vs other channels.
 *
 * 4. SAME-SUBSYSTEM DISCRIMINATION: Can the discriminant tell apart
 *    CMG1 from CMG2? (They differ only by 0.0001g offset.) This tests
 *    whether the discriminant detects micro-differences or just "is CMG."
 *
 * Compile:
 *   cc -O2 -I include -I include/chips experiments/iss_probes/iss_probe2.c -lm -o experiments/iss_probes/iss_probe2
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

/* Weights — identical to iss_telemetry.c */
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

static const float tau_iss[HIDDEN_DIM] = {
    5.0f, 15.0f, 45.0f, 120.0f, 10.0f, 30.0f, 90.0f, 600.0f
};
static const float tau_keystroke[HIDDEN_DIM] = {
    0.05f, 0.10f, 0.20f, 0.50f, 0.05f, 0.15f, 0.30f, 0.80f
};
static const float tau_constant[HIDDEN_DIM] = {
    30.0f, 30.0f, 30.0f, 30.0f, 30.0f, 30.0f, 30.0f, 30.0f
};

typedef struct {
    float mean[HIDDEN_DIM];
    float dim_std[HIDDEN_DIM];
    float pcs[N_PCS][HIDDEN_DIM];
    float pc_mean[N_PCS];
    float pc_std[N_PCS];
    int valid;
} Discriminant;

static float randf_s(unsigned int *s) {
    *s = *s * 1103515245u + 12345u;
    return (float)(*s & 0x7FFFFFFFu) / (float)0x7FFFFFFFu;
}
static float gaussf(unsigned int *s) {
    float u1 = randf_s(s) + 1e-10f, u2 = randf_s(s);
    return sqrtf(-2.0f * logf(u1)) * cosf(6.28318530718f * u2);
}
static float dotf(const float *a, const float *b, int n) {
    float s = 0; for (int i = 0; i < n; i++) s += a[i] * b[i]; return s;
}
static float vec_normalize(float *v, int n) {
    float norm = 0; for (int i = 0; i < n; i++) norm += v[i]*v[i];
    norm = sqrtf(norm);
    if (norm > 1e-10f) for (int i = 0; i < n; i++) v[i] /= norm;
    return norm;
}

static void cfc_step(float value, float dt, float *h, const float *tau) {
    float input[INPUT_DIM] = {value, dt};
    float h_new[HIDDEN_DIM];
    CFC_CELL_GENERIC(input, h, dt, W_gate, b_gate, W_cand, b_cand,
                     tau, 0, INPUT_DIM, HIDDEN_DIM, h_new);
    memcpy(h, h_new, HIDDEN_DIM * sizeof(float));
}

static void learn_disc(const float samples[][HIDDEN_DIM], int n, Discriminant *d) {
    memset(d, 0, sizeof(*d));
    if (n < 5) { d->valid = 0; return; }
    for (int i = 0; i < HIDDEN_DIM; i++) {
        float s = 0; for (int t = 0; t < n; t++) s += samples[t][i];
        d->mean[i] = s / n;
    }
    for (int i = 0; i < HIDDEN_DIM; i++) {
        float s = 0; for (int t = 0; t < n; t++) {
            float dd = samples[t][i] - d->mean[i]; s += dd*dd;
        }
        d->dim_std[i] = sqrtf(s/n + 1e-8f);
    }
    float centered[n][HIDDEN_DIM];
    for (int t = 0; t < n; t++)
        for (int i = 0; i < HIDDEN_DIM; i++)
            centered[t][i] = samples[t][i] - d->mean[i];
    for (int pc = 0; pc < N_PCS; pc++) {
        float v[HIDDEN_DIM];
        for (int i = 0; i < HIDDEN_DIM; i++) v[i] = centered[0][i] + 0.01f*(i+1);
        vec_normalize(v, HIDDEN_DIM);
        for (int iter = 0; iter < POWER_ITERS; iter++) {
            float v_new[HIDDEN_DIM] = {0};
            for (int t = 0; t < n; t++) {
                float proj = dotf(centered[t], v, HIDDEN_DIM);
                for (int i = 0; i < HIDDEN_DIM; i++) v_new[i] += proj*centered[t][i];
            }
            for (int i = 0; i < HIDDEN_DIM; i++) v_new[i] /= n;
            memcpy(v, v_new, sizeof(v));
            vec_normalize(v, HIDDEN_DIM);
        }
        memcpy(d->pcs[pc], v, sizeof(float)*HIDDEN_DIM);
        float ps = 0, ps2 = 0;
        for (int t = 0; t < n; t++) {
            float p = dotf(centered[t], v, HIDDEN_DIM);
            ps += p; ps2 += p*p;
        }
        d->pc_mean[pc] = ps / n;
        float var = ps2/n - d->pc_mean[pc]*d->pc_mean[pc];
        d->pc_std[pc] = sqrtf(var > 0 ? var : 1e-8f);
        for (int t = 0; t < n; t++) {
            float p = dotf(centered[t], v, HIDDEN_DIM);
            for (int i = 0; i < HIDDEN_DIM; i++) centered[t][i] -= p*v[i];
        }
    }
    d->valid = 1;
}

static float score_hybrid(const float *h, const Discriminant *d) {
    if (!d->valid) return 0.5f;
    float md = 0, centered[HIDDEN_DIM];
    for (int i = 0; i < HIDDEN_DIM; i++) {
        centered[i] = h[i] - d->mean[i];
        float z = centered[i] / (d->dim_std[i] + 1e-8f);
        md += z*z;
    }
    md /= HIDDEN_DIM;
    float ms = SIGMOID_CHIP(2.0f - md);
    float pd = 0;
    for (int pc = 0; pc < N_PCS; pc++) {
        float proj = dotf(centered, d->pcs[pc], HIDDEN_DIM);
        float z = (proj - d->pc_mean[pc]) / (d->pc_std[pc] + 1e-8f);
        pd += z*z;
    }
    pd /= N_PCS;
    float ps = SIGMOID_CHIP(2.0f - pd);
    return MEAN_WEIGHT * ms + PCA_WEIGHT * ps;
}

/* Telemetry generators */
#define ORB_PERIOD 5520.0f

static float gen_cmg(float t, int offset, unsigned int *s) {
    float ph = 6.28318530718f * t / ORB_PERIOD;
    return 0.001f + 0.0001f*offset + 0.0002f*sinf(ph) + 0.0002f*gaussf(s);
}
static float gen_coolant(float t, int loop_b, unsigned int *s) {
    float ph = 6.28318530718f * t / ORB_PERIOD;
    return 15.0f + (loop_b ? 2.0f : 0.0f) + 8.0f*sinf(ph) + 0.5f*gaussf(s);
}
static float gen_cabin_p(float t, unsigned int *s) {
    float ph = 6.28318530718f * t / ORB_PERIOD;
    return 101.3f + 0.02f*sinf(ph) + 0.05f*gaussf(s);
}
static float gen_channel(int ch, float t, unsigned int *s) {
    switch (ch) {
    case 0: return gen_cmg(t, 0, s);
    case 1: return gen_cmg(t, 1, s);
    case 2: return gen_cmg(t, 2, s);
    case 3: return gen_cmg(t, 3, s);
    case 4: return gen_coolant(t, 0, s);
    case 5: return gen_coolant(t, 1, s);
    case 6: return gen_cabin_p(t, s);
    default: return gaussf(s);
    }
}

/* Collect enrollment samples */
static int collect_samples(
    int ch, float dt, int n_steps, unsigned int *seed,
    const float *tau_vec, float samples[][HIDDEN_DIM], int max_s
) {
    float h[HIDDEN_DIM] = {0};
    int n = 0;
    for (int step = 0; step < n_steps; step++) {
        float t = step * dt;
        float v = gen_channel(ch, t, seed);
        cfc_step(v, dt, h, tau_vec);
        if (step > WARMUP && n < max_s) {
            memcpy(samples[n], h, sizeof(float)*HIDDEN_DIM);
            n++;
        }
    }
    return n;
}

int main(void) {
    printf("================================================================\n");
    printf("  ISS Falsification Probe 2: Subtle Anomalies + CMG Deep Dive\n");
    printf("================================================================\n\n");

    float dt = 10.0f;
    int enroll_steps = 1104;
    int test_steps = 552;
    int N_RUNS = 20;

    /* ══════════════════════════════════════════════════════════════════════
     * TEST 1: ANOMALY MAGNITUDE SWEEP
     *
     * Original anomaly for CMG step: +0.005g on 0.001g baseline (5x jump).
     * Sweep: 100%, 50%, 20%, 10%, 5%, 2%, 1% of that magnitude.
     * At each level, test all three tau configs.
     * ══════════════════════════════════════════════════════════════════════ */
    printf("TEST 1: ANOMALY MAGNITUDE SWEEP (CMG1, step anomaly)\n");
    printf("  Baseline vibration: ~0.001g. Full anomaly: +0.005g.\n");
    printf("  Sweeping magnitude. Looking for where tau differentiation appears.\n\n");

    float magnitudes[] = {1.0f, 0.5f, 0.2f, 0.1f, 0.05f, 0.02f, 0.01f, 0.005f, 0.002f, 0.001f};
    int n_mags = 10;
    const float *tau_sets[] = {tau_iss, tau_keystroke, tau_constant};
    const char *tau_labels[] = {"ISS", "KS", "Const"};

    printf("  %-8s", "Mag%");
    for (int ti = 0; ti < 3; ti++)
        printf("  %-8s %-6s", tau_labels[ti], "Det");
    printf("\n");
    printf("  %-8s", "--------");
    for (int ti = 0; ti < 3; ti++)
        printf("  %-8s %-6s", "--------", "------");
    printf("\n");

    for (int mi = 0; mi < n_mags; mi++) {
        float mag = magnitudes[mi];
        float anomaly_offset = 0.005f * mag;  /* scaled step */

        printf("  %-8.1f", mag * 100.0f);

        for (int ti = 0; ti < 3; ti++) {
            const float *tau_vec = tau_sets[ti];
            float score_sum = 0;
            int detected = 0;

            for (int run = 0; run < N_RUNS; run++) {
                /* Enroll */
                unsigned int eseed = 42 + run * 97;
                float samples[MAX_SAMPLES][HIDDEN_DIM];
                int ns = collect_samples(0, dt, enroll_steps, &eseed,
                                         tau_vec, samples, MAX_SAMPLES);
                Discriminant disc;
                learn_disc(samples, ns, &disc);

                /* Test with anomaly injected at midpoint */
                unsigned int tseed = 9999 + run * 131;
                float h[HIDDEN_DIM] = {0};
                float final_score = 0;
                int scored = 0;

                for (int step = 0; step < test_steps; step++) {
                    float t = step * dt;
                    float v = gen_cmg(t, 0, &tseed);
                    if (step >= test_steps / 2)
                        v += anomaly_offset;
                    cfc_step(v, dt, h, tau_vec);
                    if (step > WARMUP) {
                        final_score = score_hybrid(h, &disc);
                        scored++;
                    }
                }
                score_sum += final_score;
                if (final_score < 0.35f) detected++;
            }

            printf("  %-8.4f %-2d/%-2d", score_sum / N_RUNS, detected, N_RUNS);
        }
        printf("\n");
    }

    /* ══════════════════════════════════════════════════════════════════════
     * TEST 2: SLOW DRIFT vs FAST TRANSIENT
     *
     * Two anomaly types on coolant (ch 4):
     *   (a) Single-sample spike (fast transient): +5°C for 1 sample
     *   (b) Slow drift: +0.001°C/sample accumulating over test period
     *
     * Theory: ISS tau (slow time constants) should catch slow drifts
     * better. Keystroke tau (fast time constants) should catch fast
     * transients better.
     * ══════════════════════════════════════════════════════════════════════ */
    printf("\nTEST 2: SLOW DRIFT vs FAST TRANSIENT (Coolant, ch 4)\n");
    printf("  Baseline: 15C +/- 8C orbital. Noise: 0.5C.\n\n");

    printf("  %-10s", "Anomaly");
    for (int ti = 0; ti < 3; ti++)
        printf("  %-8s %-6s", tau_labels[ti], "Det");
    printf("\n");
    printf("  %-10s", "----------");
    for (int ti = 0; ti < 3; ti++)
        printf("  %-8s %-6s", "--------", "------");
    printf("\n");

    /* Drift rates to test (°C per sample = °C per 10s) */
    float drift_rates[] = {0.001f, 0.002f, 0.005f, 0.01f, 0.02f};
    int n_drifts = 5;

    for (int di = 0; di < n_drifts; di++) {
        float drift_rate = drift_rates[di];
        printf("  drift %.3f", drift_rate);

        for (int ti = 0; ti < 3; ti++) {
            const float *tau_vec = tau_sets[ti];
            float score_sum = 0;
            int detected = 0;

            for (int run = 0; run < N_RUNS; run++) {
                unsigned int eseed = 42 + run * 97;
                float samples[MAX_SAMPLES][HIDDEN_DIM];
                int ns = collect_samples(4, dt, enroll_steps, &eseed,
                                         tau_vec, samples, MAX_SAMPLES);
                Discriminant disc;
                learn_disc(samples, ns, &disc);

                unsigned int tseed = 9999 + run * 131;
                float h[HIDDEN_DIM] = {0};
                float final_score = 0;

                for (int step = 0; step < test_steps; step++) {
                    float t = step * dt;
                    float v = gen_coolant(t, 0, &tseed);
                    if (step >= test_steps / 2) {
                        int steps_since = step - test_steps / 2;
                        v += drift_rate * steps_since;
                    }
                    cfc_step(v, dt, h, tau_vec);
                    if (step > WARMUP) {
                        final_score = score_hybrid(h, &disc);
                    }
                }
                score_sum += final_score;
                if (final_score < 0.35f) detected++;
            }

            printf("  %-8.4f %-2d/%-2d", score_sum / N_RUNS, detected, N_RUNS);
        }
        printf("\n");
    }

    /* Spike tests */
    printf("\n");
    float spike_mags[] = {1.0f, 2.0f, 5.0f, 10.0f, 20.0f};
    int n_spikes = 5;

    for (int si = 0; si < n_spikes; si++) {
        float spike = spike_mags[si];
        printf("  spike %4.0fC", spike);

        for (int ti = 0; ti < 3; ti++) {
            const float *tau_vec = tau_sets[ti];
            float score_sum = 0;
            int detected = 0;

            for (int run = 0; run < N_RUNS; run++) {
                unsigned int eseed = 42 + run * 97;
                float samples[MAX_SAMPLES][HIDDEN_DIM];
                int ns = collect_samples(4, dt, enroll_steps, &eseed,
                                         tau_vec, samples, MAX_SAMPLES);
                Discriminant disc;
                learn_disc(samples, ns, &disc);

                unsigned int tseed = 9999 + run * 131;
                float h[HIDDEN_DIM] = {0};
                float final_score = 0;

                /* Inject spike at 3/4 mark, measure score at end */
                int spike_step = test_steps * 3 / 4;
                for (int step = 0; step < test_steps; step++) {
                    float t = step * dt;
                    float v = gen_coolant(t, 0, &tseed);
                    if (step == spike_step) v += spike;
                    cfc_step(v, dt, h, tau_vec);
                    if (step > WARMUP) {
                        final_score = score_hybrid(h, &disc);
                    }
                }
                score_sum += final_score;
                if (final_score < 0.35f) detected++;
            }

            printf("  %-8.4f %-2d/%-2d", score_sum / N_RUNS, detected, N_RUNS);
        }
        printf("\n");
    }

    /* ══════════════════════════════════════════════════════════════════════
     * TEST 3: CMG ATTRACTOR ANALYSIS
     *
     * Compute hidden state statistics for CMG vs other channels.
     * Is CMG hidden state variance anomalously small?
     * ══════════════════════════════════════════════════════════════════════ */
    printf("\nTEST 3: HIDDEN STATE ANALYSIS\n");
    printf("  Compare hidden state spread across channels.\n");
    printf("  CMG inputs are tiny (~0.001). Are hidden states degenerate?\n\n");

    int analysis_channels[] = {0, 1, 4, 5, 6};
    const char *analysis_names[] = {"CMG1", "CMG2", "CoolA", "CoolB", "CabinP"};

    printf("  %-8s  %-10s  %-10s  %-10s  %-10s\n",
           "Channel", "H-Mean", "H-Std", "H-Range", "Input-Range");
    printf("  %-8s  %-10s  %-10s  %-10s  %-10s\n",
           "--------", "----------", "----------", "----------", "----------");

    for (int ci = 0; ci < 5; ci++) {
        int ch = analysis_channels[ci];
        unsigned int seed = 42;

        float h[HIDDEN_DIM] = {0};
        float h_min[HIDDEN_DIM], h_max[HIDDEN_DIM];
        float h_sum[HIDDEN_DIM] = {0}, h_sum2[HIDDEN_DIM] = {0};
        float v_min = 1e10f, v_max = -1e10f;
        int count = 0;

        for (int i = 0; i < HIDDEN_DIM; i++) {
            h_min[i] = 1e10f; h_max[i] = -1e10f;
        }

        for (int step = 0; step < enroll_steps; step++) {
            float t = step * dt;
            float v = gen_channel(ch, t, &seed);
            if (v < v_min) v_min = v;
            if (v > v_max) v_max = v;
            cfc_step(v, dt, h, tau_iss);

            if (step > WARMUP) {
                for (int i = 0; i < HIDDEN_DIM; i++) {
                    h_sum[i] += h[i];
                    h_sum2[i] += h[i] * h[i];
                    if (h[i] < h_min[i]) h_min[i] = h[i];
                    if (h[i] > h_max[i]) h_max[i] = h[i];
                }
                count++;
            }
        }

        /* Compute aggregate stats */
        float total_mean = 0, total_std = 0, total_range = 0;
        for (int i = 0; i < HIDDEN_DIM; i++) {
            float mean = h_sum[i] / count;
            float var = h_sum2[i] / count - mean * mean;
            total_mean += fabsf(mean);
            total_std += sqrtf(var > 0 ? var : 0);
            total_range += h_max[i] - h_min[i];
        }
        total_mean /= HIDDEN_DIM;
        total_std /= HIDDEN_DIM;
        total_range /= HIDDEN_DIM;

        printf("  %-8s  %-10.6f  %-10.6f  %-10.6f  %-10.4f\n",
               analysis_names[ci], total_mean, total_std, total_range,
               v_max - v_min);
    }

    /* ══════════════════════════════════════════════════════════════════════
     * TEST 4: SAME-SUBSYSTEM DISCRIMINATION
     *
     * Enroll on CMG1, test on CMG2 (differ by 0.0001g offset).
     * Enroll on CoolA, test on CoolB (differ by 2°C offset).
     * ══════════════════════════════════════════════════════════════════════ */
    printf("\nTEST 4: SAME-SUBSYSTEM DISCRIMINATION\n");
    printf("  Can discriminant tell apart units within same subsystem?\n\n");

    typedef struct { int enroll; int same; int other; const char *label; } SubTest;
    SubTest subtests[] = {
        {0, 0, 1, "CMG1 vs CMG2 (0.0001g diff)"},
        {0, 0, 2, "CMG1 vs CMG3 (0.0002g diff)"},
        {0, 0, 3, "CMG1 vs CMG4 (0.0003g diff)"},
        {4, 4, 5, "CoolA vs CoolB (2C diff)"},
    };
    int n_subtests = 4;

    printf("  %-30s  %-10s  %-10s  %-10s\n",
           "Test", "Same", "Other", "Delta");
    printf("  %-30s  %-10s  %-10s  %-10s\n",
           "------------------------------", "----------", "----------", "----------");

    for (int si = 0; si < n_subtests; si++) {
        SubTest *st = &subtests[si];
        float same_sum = 0, other_sum = 0;

        for (int run = 0; run < N_RUNS; run++) {
            unsigned int eseed = 42 + run * 97;
            float samples[MAX_SAMPLES][HIDDEN_DIM];
            int ns = collect_samples(st->enroll, dt, enroll_steps, &eseed,
                                     tau_iss, samples, MAX_SAMPLES);
            Discriminant disc;
            learn_disc(samples, ns, &disc);

            /* Same unit */
            {
                unsigned int seed = 9999 + run * 131;
                float h[HIDDEN_DIM] = {0};
                float ss = 0; int scored = 0;
                for (int step = 0; step < test_steps; step++) {
                    float t = step * dt;
                    float v = gen_channel(st->same, t, &seed);
                    cfc_step(v, dt, h, tau_iss);
                    if (step > WARMUP) { ss += score_hybrid(h, &disc); scored++; }
                }
                same_sum += ss / scored;
            }

            /* Other unit */
            {
                unsigned int seed = 9999 + run * 131;
                float h[HIDDEN_DIM] = {0};
                float ss = 0; int scored = 0;
                for (int step = 0; step < test_steps; step++) {
                    float t = step * dt;
                    float v = gen_channel(st->other, t, &seed);
                    cfc_step(v, dt, h, tau_iss);
                    if (step > WARMUP) { ss += score_hybrid(h, &disc); scored++; }
                }
                other_sum += ss / scored;
            }
        }

        printf("  %-30s  %-10.4f  %-10.4f  %+.4f\n",
               st->label, same_sum / N_RUNS, other_sum / N_RUNS,
               other_sum / N_RUNS - same_sum / N_RUNS);
    }

    /* ══════════════════════════════════════════════════════════════════════
     * TEST 5: BASELINE THRESHOLD — Is a simple threshold detector
     * as good as the CfC + discriminant for these anomalies?
     *
     * Compare CfC discriminant detection to: "flag if value > mean + 3*std"
     * If simple threshold matches CfC: the CfC adds no value.
     * ══════════════════════════════════════════════════════════════════════ */
    printf("\nTEST 5: CfC vs SIMPLE THRESHOLD (Coolant, drift anomaly)\n");
    printf("  Does CfC + discriminant outperform mean + 3*std?\n\n");

    printf("  %-12s  %-8s %-6s  %-8s %-6s\n",
           "Drift Rate", "CfC", "Det", "3-sigma", "Det");
    printf("  %-12s  %-8s %-6s  %-8s %-6s\n",
           "------------", "--------", "------", "--------", "------");

    for (int di = 0; di < n_drifts; di++) {
        float drift_rate = drift_rates[di];
        float cfc_score_sum = 0;
        int cfc_det = 0;
        int sigma_det = 0;

        for (int run = 0; run < N_RUNS; run++) {
            /* Enrollment: collect raw value stats AND CfC discriminant */
            unsigned int eseed = 42 + run * 97;
            unsigned int eseed2 = eseed;  /* same stream for both */
            float samples[MAX_SAMPLES][HIDDEN_DIM];
            int ns = collect_samples(4, dt, enroll_steps, &eseed,
                                     tau_iss, samples, MAX_SAMPLES);
            Discriminant disc;
            learn_disc(samples, ns, &disc);

            /* Collect raw value stats for threshold */
            float v_sum = 0, v_sum2 = 0;
            int v_count = 0;
            for (int step = 0; step < enroll_steps; step++) {
                float t = step * dt;
                float v = gen_coolant(t, 0, &eseed2);
                if (step > WARMUP) {
                    v_sum += v; v_sum2 += v*v; v_count++;
                }
            }
            float v_mean = v_sum / v_count;
            float v_var = v_sum2 / v_count - v_mean * v_mean;
            float v_std = sqrtf(v_var > 0 ? v_var : 1e-8f);
            float threshold_hi = v_mean + 3.0f * v_std;

            /* Test with drift */
            unsigned int tseed = 9999 + run * 131;
            unsigned int tseed2 = tseed;
            float h[HIDDEN_DIM] = {0};
            float final_cfc = 0;
            int sigma_flagged = 0;

            for (int step = 0; step < test_steps; step++) {
                float t = step * dt;
                float v = gen_coolant(t, 0, &tseed);
                float v2 = gen_coolant(t, 0, &tseed2);

                if (step >= test_steps / 2) {
                    int steps_since = step - test_steps / 2;
                    v += drift_rate * steps_since;
                    v2 += drift_rate * steps_since;
                }

                cfc_step(v, dt, h, tau_iss);
                if (step > WARMUP)
                    final_cfc = score_hybrid(h, &disc);

                if (v2 > threshold_hi)
                    sigma_flagged = 1;
            }

            cfc_score_sum += final_cfc;
            if (final_cfc < 0.35f) cfc_det++;
            if (sigma_flagged) sigma_det++;
        }

        printf("  drift %-5.3f  %-8.4f %-2d/%-2d  %-8s %-2d/%-2d\n",
               drift_rate, cfc_score_sum / N_RUNS, cfc_det, N_RUNS,
               "-", sigma_det, N_RUNS);
    }

    printf("\n================================================================\n");
    printf("  Key Questions:\n");
    printf("    1. Does tau differentiate at small anomaly magnitudes?\n");
    printf("    2. Does tau differentiate between slow drift vs fast spike?\n");
    printf("    3. Is CMG hidden state degenerate (tiny variance)?\n");
    printf("    4. Can discriminant separate CMG1 from CMG2?\n");
    printf("    5. Does CfC outperform simple 3-sigma threshold?\n");
    printf("================================================================\n");

    return 0;
}
