/*
 * ISS Falsification Probe 1: Seed Correlation + Null Discriminant + Tau Ablation
 *
 * Questions this probe answers:
 *
 * 1. SEED CORRELATION: The main sim uses seed=42 for both enrollment and
 *    test phases. The RNG state is continuous — enrollment ends at some
 *    state, test starts from that state. Does using fully independent
 *    seeds for enrollment vs test destroy the "normal" scores?
 *    If yes: our normal scores are an artifact, just like keystroke v1.
 *
 * 2. NULL DISCRIMINANT: If we enroll on pure Gaussian noise (no ISS
 *    structure at all), does the discriminant still produce high scores
 *    on normal ISS data? If yes: the discriminant isn't learning ISS
 *    patterns, it's just learning "anything CfC-shaped."
 *
 * 3. WRONG-CHANNEL TEST: If we enroll on CMG data but test with coolant
 *    data, does the score drop? If not: the discriminant isn't learning
 *    channel-specific patterns.
 *
 * 4. TAU ABLATION: Does replacing ISS tau values (5-600s) with keystroke
 *    tau values (0.05-0.80s) change detection performance? If not: the
 *    tau tuning claim is vacuous.
 *
 * 5. RANDOM DISCRIMINANT: Score normal data against a discriminant with
 *    random mean/PCs. If scores are similar to the real discriminant:
 *    the discriminant structure doesn't matter.
 *
 * 6. THRESHOLD ROC: Sweep thresholds and compute TPR/FPR to see if 0.35
 *    is cherry-picked or sits on a natural separation boundary.
 *
 * Compile:
 *   cc -O2 -I include -I include/chips experiments/iss_probes/iss_probe1.c -lm -o experiments/iss_probes/iss_probe1
 *
 * Created by: Tripp + Manus (falsification)
 * Date: February 2026
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <sys/time.h>

#include "onnx_shapes.h"
#include "chips/cfc_cell_chip.h"
#include "chips/norm_chip.h"
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

/* ═══════════════════════════════════════════════════════════════════════════
 * Weights — same as iss_telemetry.c
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

/* ISS tau — the values we're testing */
static const float tau_iss[HIDDEN_DIM] = {
    5.0f, 15.0f, 45.0f, 120.0f, 10.0f, 30.0f, 90.0f, 600.0f
};

/* Keystroke tau — wrong timescale, used for ablation */
static const float tau_keystroke[HIDDEN_DIM] = {
    0.05f, 0.10f, 0.20f, 0.50f, 0.05f, 0.15f, 0.30f, 0.80f
};

/* Constant tau — no multi-scale, used for ablation */
static const float tau_constant[HIDDEN_DIM] = {
    30.0f, 30.0f, 30.0f, 30.0f, 30.0f, 30.0f, 30.0f, 30.0f
};

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

/* CfC step with selectable tau */
static void cfc_step(float value, float dt, float *h_state, const float *tau) {
    float input[INPUT_DIM] = { value, dt };
    float h_new[HIDDEN_DIM];
    CFC_CELL_GENERIC(input, h_state, dt, W_gate, b_gate, W_cand, b_cand,
                     tau, 0, INPUT_DIM, HIDDEN_DIM, h_new);
    memcpy(h_state, h_new, HIDDEN_DIM * sizeof(float));
}

/* Learn discriminant */
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

/* Score */
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
 * Telemetry Generators
 * ═══════════════════════════════════════════════════════════════════════════ */

#define ORBITAL_PERIOD 5520.0f

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

static float gen_noise(unsigned int *seed) {
    return gaussf(seed);
}

/* Generate value for a channel */
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
    default: return gen_noise(seed);
    }
}

/* Add anomaly to value */
static float add_anomaly(float value, int ch, int type, float dt_anom) {
    float ramp = dt_anom / 3000.0f;
    if (ramp > 1.0f) ramp = 1.0f;

    switch (type) {
    case 0: /* step */
        if (ch < 4) return value + 0.005f;
        if (ch < 6) return value + 5.0f;
        if (ch == 6) return value - 0.5f;
        return value - 1.0f;
    case 1: /* ramp */
        if (ch < 4) return value + 0.008f * ramp;
        if (ch < 6) return value + 8.0f * ramp;
        if (ch == 6) return value - 1.0f * ramp;
        return value - 2.0f * ramp;
    }
    return value;
}

/* ═══════════════════════════════════════════════════════════════════════════
 * Collect enrollment samples for one channel
 * ═══════════════════════════════════════════════════════════════════════════ */
static int collect_samples(
    int ch, float dt, int n_steps, unsigned int *seed,
    const float *tau_vec,
    float samples[][HIDDEN_DIM], int max_samples
) {
    float h[HIDDEN_DIM] = {0};
    int n = 0;

    for (int step = 0; step < n_steps; step++) {
        float t = step * dt;
        float value = gen_channel(ch, t, seed);
        cfc_step(value, dt, h, tau_vec);

        if (step > WARMUP && n < max_samples) {
            memcpy(samples[n], h, sizeof(float) * HIDDEN_DIM);
            n++;
        }
    }
    return n;
}

/* Get final hidden state after processing a sequence */
static void get_final_h(
    int ch, float dt, int n_steps, unsigned int *seed,
    const float *tau_vec, float *h_out
) {
    float h[HIDDEN_DIM] = {0};
    for (int step = 0; step < n_steps; step++) {
        float t = step * dt;
        float value = gen_channel(ch, t, seed);
        cfc_step(value, dt, h, tau_vec);
    }
    memcpy(h_out, h, sizeof(float) * HIDDEN_DIM);
}

/* Get final h with anomaly injected partway through */
static void get_final_h_anomaly(
    int ch, float dt, int n_steps, int inject_at,
    int anomaly_type, unsigned int *seed,
    const float *tau_vec, float *h_out
) {
    float h[HIDDEN_DIM] = {0};
    for (int step = 0; step < n_steps; step++) {
        float t = step * dt;
        float value = gen_channel(ch, t, seed);
        if (step >= inject_at) {
            float dt_anom = (step - inject_at) * dt;
            value = add_anomaly(value, ch, anomaly_type, dt_anom);
        }
        cfc_step(value, dt, h, tau_vec);
    }
    memcpy(h_out, h, sizeof(float) * HIDDEN_DIM);
}

/* ═══════════════════════════════════════════════════════════════════════════ */

int main(void) {
    printf("================================================================\n");
    printf("  ISS Falsification Probe 1\n");
    printf("================================================================\n\n");

    float dt = 10.0f;
    int enroll_steps = 1104;  /* 2 orbits */
    int test_steps = 552;     /* 1 orbit */
    int N_RUNS = 20;

    /* ══════════════════════════════════════════════════════════════════════
     * TEST 1: Seed Correlation
     *
     * Compare: (a) continuous seed (enrollment → test, same RNG stream)
     *          (b) independent seeds (different RNG for enrollment vs test)
     * ══════════════════════════════════════════════════════════════════════ */
    printf("TEST 1: SEED CORRELATION\n");
    printf("  Does using independent seeds for enrollment vs test destroy\n");
    printf("  the normal-operation scores?\n\n");

    /* Test on CMG channel 0 and Coolant channel 4 */
    int probe_channels[] = {0, 4, 6, 7};
    const char *probe_names[] = {"CMG1-vib", "CoolA-T", "CabinP", "O2-PP"};

    printf("  %-10s  %-14s  %-14s  %-10s\n",
           "Channel", "Continuous", "Independent", "Delta");
    printf("  %-10s  %-14s  %-14s  %-10s\n",
           "----------", "--------------", "--------------", "----------");

    for (int ci = 0; ci < 4; ci++) {
        int ch = probe_channels[ci];
        float cont_sum = 0, indep_sum = 0;

        for (int run = 0; run < N_RUNS; run++) {
            /* (a) Continuous seed */
            {
                unsigned int seed = 42 + run * 97;
                unsigned int seed_copy = seed;

                float samples[MAX_SAMPLES][HIDDEN_DIM];
                int ns = collect_samples(ch, dt, enroll_steps, &seed,
                                         tau_iss, samples, MAX_SAMPLES);
                Discriminant disc;
                learn_disc(samples, ns, &disc);

                /* Test continues from same seed state */
                float h[HIDDEN_DIM] = {0};
                float score_sum = 0;
                int scored = 0;
                for (int step = 0; step < test_steps; step++) {
                    float t = step * dt;
                    float value = gen_channel(ch, t, &seed);
                    cfc_step(value, dt, h, tau_iss);
                    if (step > WARMUP) {
                        score_sum += score_hybrid(h, &disc);
                        scored++;
                    }
                }
                cont_sum += score_sum / scored;
            }

            /* (b) Independent seed */
            {
                unsigned int enroll_seed = 42 + run * 97;
                unsigned int test_seed = 9999 + run * 131;

                float samples[MAX_SAMPLES][HIDDEN_DIM];
                int ns = collect_samples(ch, dt, enroll_steps, &enroll_seed,
                                         tau_iss, samples, MAX_SAMPLES);
                Discriminant disc;
                learn_disc(samples, ns, &disc);

                /* Test uses completely different seed */
                float h[HIDDEN_DIM] = {0};
                float score_sum = 0;
                int scored = 0;
                for (int step = 0; step < test_steps; step++) {
                    float t = step * dt;
                    float value = gen_channel(ch, t, &test_seed);
                    cfc_step(value, dt, h, tau_iss);
                    if (step > WARMUP) {
                        score_sum += score_hybrid(h, &disc);
                        scored++;
                    }
                }
                indep_sum += score_sum / scored;
            }
        }

        float cont_avg = cont_sum / N_RUNS;
        float indep_avg = indep_sum / N_RUNS;
        printf("  %-10s  %-14.4f  %-14.4f  %+.4f\n",
               probe_names[ci], cont_avg, indep_avg, indep_avg - cont_avg);
    }

    /* ══════════════════════════════════════════════════════════════════════
     * TEST 2: NULL DISCRIMINANT — enroll on pure noise
     * ══════════════════════════════════════════════════════════════════════ */
    printf("\nTEST 2: NULL DISCRIMINANT (enrolled on pure Gaussian noise)\n");
    printf("  If noise-enrolled discriminant scores ISS data similarly to\n");
    printf("  ISS-enrolled: the discriminant isn't learning ISS structure.\n\n");

    printf("  %-10s  %-14s  %-14s  %-10s\n",
           "Channel", "ISS-enrolled", "Noise-enrolled", "Delta");
    printf("  %-10s  %-14s  %-14s  %-10s\n",
           "----------", "--------------", "--------------", "----------");

    for (int ci = 0; ci < 4; ci++) {
        int ch = probe_channels[ci];
        float iss_sum = 0, noise_sum = 0;

        for (int run = 0; run < N_RUNS; run++) {
            /* ISS-enrolled discriminant */
            Discriminant disc_iss;
            {
                unsigned int seed = 42 + run * 97;
                float samples[MAX_SAMPLES][HIDDEN_DIM];
                int ns = collect_samples(ch, dt, enroll_steps, &seed,
                                         tau_iss, samples, MAX_SAMPLES);
                learn_disc(samples, ns, &disc_iss);
            }

            /* Noise-enrolled discriminant — feed pure Gaussian noise */
            Discriminant disc_noise;
            {
                unsigned int seed = 77777 + run * 53;
                float h[HIDDEN_DIM] = {0};
                float samples[MAX_SAMPLES][HIDDEN_DIM];
                int n = 0;
                for (int step = 0; step < enroll_steps; step++) {
                    float value = gaussf(&seed);  /* pure noise, no ISS structure */
                    cfc_step(value, dt, h, tau_iss);
                    if (step > WARMUP && n < MAX_SAMPLES) {
                        memcpy(samples[n], h, sizeof(float) * HIDDEN_DIM);
                        n++;
                    }
                }
                learn_disc(samples, n, &disc_noise);
            }

            /* Score ISS test data against both discriminants */
            unsigned int test_seed = 9999 + run * 131;
            unsigned int test_seed2 = test_seed;  /* same test data */

            float h1[HIDDEN_DIM] = {0}, h2[HIDDEN_DIM] = {0};
            float s1 = 0, s2 = 0;
            int scored = 0;

            for (int step = 0; step < test_steps; step++) {
                float t = step * dt;
                float v1 = gen_channel(ch, t, &test_seed);
                float v2 = gen_channel(ch, t, &test_seed2);

                cfc_step(v1, dt, h1, tau_iss);
                cfc_step(v2, dt, h2, tau_iss);

                if (step > WARMUP) {
                    s1 += score_hybrid(h1, &disc_iss);
                    s2 += score_hybrid(h2, &disc_noise);
                    scored++;
                }
            }
            iss_sum += s1 / scored;
            noise_sum += s2 / scored;
        }

        printf("  %-10s  %-14.4f  %-14.4f  %+.4f\n",
               probe_names[ci], iss_sum / N_RUNS, noise_sum / N_RUNS,
               noise_sum / N_RUNS - iss_sum / N_RUNS);
    }

    /* ══════════════════════════════════════════════════════════════════════
     * TEST 3: WRONG-CHANNEL CROSS-TEST
     *
     * Enroll on channel X, test with channel Y data.
     * If scores are similar: the discriminant isn't channel-specific.
     * ══════════════════════════════════════════════════════════════════════ */
    printf("\nTEST 3: WRONG-CHANNEL CROSS-TEST\n");
    printf("  Enroll on one channel, test on a different channel.\n");
    printf("  If scores stay high: discriminant isn't channel-specific.\n\n");

    printf("  %-18s  %-10s  %-10s  %-10s\n",
           "Enroll -> Test", "Same-ch", "Wrong-ch", "Delta");
    printf("  %-18s  %-10s  %-10s  %-10s\n",
           "------------------", "----------", "----------", "----------");

    /* Test pairs: (enroll_ch, test_ch) */
    int pairs[][2] = {{0, 4}, {4, 0}, {0, 7}, {6, 7}, {4, 5}};
    const char *pair_labels[] = {
        "CMG1 -> CoolA", "CoolA -> CMG1", "CMG1 -> O2",
        "CabinP -> O2", "CoolA -> CoolB"
    };
    int n_pairs = 5;

    for (int pi = 0; pi < n_pairs; pi++) {
        int enroll_ch = pairs[pi][0];
        int test_ch = pairs[pi][1];
        float same_sum = 0, wrong_sum = 0;

        for (int run = 0; run < N_RUNS; run++) {
            unsigned int enroll_seed = 42 + run * 97;
            float samples[MAX_SAMPLES][HIDDEN_DIM];
            int ns = collect_samples(enroll_ch, dt, enroll_steps, &enroll_seed,
                                     tau_iss, samples, MAX_SAMPLES);
            Discriminant disc;
            learn_disc(samples, ns, &disc);

            /* Same channel test */
            {
                unsigned int seed = 9999 + run * 131;
                float h[HIDDEN_DIM] = {0};
                float ss = 0; int scored = 0;
                for (int step = 0; step < test_steps; step++) {
                    float t = step * dt;
                    float v = gen_channel(enroll_ch, t, &seed);
                    cfc_step(v, dt, h, tau_iss);
                    if (step > WARMUP) { ss += score_hybrid(h, &disc); scored++; }
                }
                same_sum += ss / scored;
            }

            /* Wrong channel test */
            {
                unsigned int seed = 9999 + run * 131;
                float h[HIDDEN_DIM] = {0};
                float ss = 0; int scored = 0;
                for (int step = 0; step < test_steps; step++) {
                    float t = step * dt;
                    float v = gen_channel(test_ch, t, &seed);
                    cfc_step(v, dt, h, tau_iss);
                    if (step > WARMUP) { ss += score_hybrid(h, &disc); scored++; }
                }
                wrong_sum += ss / scored;
            }
        }

        printf("  %-18s  %-10.4f  %-10.4f  %+.4f\n",
               pair_labels[pi],
               same_sum / N_RUNS, wrong_sum / N_RUNS,
               wrong_sum / N_RUNS - same_sum / N_RUNS);
    }

    /* ══════════════════════════════════════════════════════════════════════
     * TEST 4: TAU ABLATION
     *
     * Same test, three tau configurations:
     *   (a) ISS tau (5-600s) — the design values
     *   (b) Keystroke tau (0.05-0.80s) — wrong timescale
     *   (c) Constant tau (all 30s) — no multi-scale
     *
     * Measure: detection of ramp anomaly on CMG (step + ramp on ch 0)
     * ══════════════════════════════════════════════════════════════════════ */
    printf("\nTEST 4: TAU ABLATION\n");
    printf("  Does tau timescale matter? Test ramp anomaly detection.\n\n");

    const float *tau_sets[] = {tau_iss, tau_keystroke, tau_constant};
    const char *tau_names[] = {"ISS (5-600s)", "Keystroke (0.05-0.8s)", "Constant (30s)"};

    printf("  %-24s  %-10s  %-10s  %-10s  %-10s\n",
           "Tau Config", "Normal", "Step", "Ramp", "Detected");
    printf("  %-24s  %-10s  %-10s  %-10s  %-10s\n",
           "------------------------", "----------", "----------", "----------", "----------");

    for (int ti = 0; ti < 3; ti++) {
        const float *tau_vec = tau_sets[ti];
        float normal_sum = 0, step_sum = 0, ramp_sum = 0;
        int step_det = 0, ramp_det = 0;

        for (int run = 0; run < N_RUNS; run++) {
            /* Enroll */
            unsigned int enroll_seed = 42 + run * 97;
            float samples[MAX_SAMPLES][HIDDEN_DIM];
            int ns = collect_samples(0, dt, enroll_steps, &enroll_seed,
                                     tau_vec, samples, MAX_SAMPLES);
            Discriminant disc;
            learn_disc(samples, ns, &disc);

            /* Normal test */
            {
                unsigned int seed = 9999 + run * 131;
                float h[HIDDEN_DIM];
                get_final_h(0, dt, test_steps, &seed, tau_vec, h);
                normal_sum += score_hybrid(h, &disc);
            }

            /* Step anomaly — inject at midpoint */
            {
                unsigned int seed = 9999 + run * 131;
                float h[HIDDEN_DIM];
                get_final_h_anomaly(0, dt, test_steps, test_steps/2, 0,
                                    &seed, tau_vec, h);
                float s = score_hybrid(h, &disc);
                step_sum += s;
                if (s < 0.35f) step_det++;
            }

            /* Ramp anomaly — inject at midpoint */
            {
                unsigned int seed = 9999 + run * 131;
                float h[HIDDEN_DIM];
                get_final_h_anomaly(0, dt, test_steps, test_steps/2, 1,
                                    &seed, tau_vec, h);
                float s = score_hybrid(h, &disc);
                ramp_sum += s;
                if (s < 0.35f) ramp_det++;
            }
        }

        printf("  %-24s  %-10.4f  %-10.4f  %-10.4f  S:%d/%d R:%d/%d\n",
               tau_names[ti],
               normal_sum / N_RUNS, step_sum / N_RUNS, ramp_sum / N_RUNS,
               step_det, N_RUNS, ramp_det, N_RUNS);
    }

    /* ══════════════════════════════════════════════════════════════════════
     * TEST 5: RANDOM DISCRIMINANT
     *
     * Create a discriminant with random mean and PCs (not from any data).
     * Score normal ISS data against it. If scores are similar to real
     * discriminant → structure doesn't matter.
     * ══════════════════════════════════════════════════════════════════════ */
    printf("\nTEST 5: RANDOM DISCRIMINANT\n");
    printf("  Score normal data against random (non-enrolled) discriminant.\n");
    printf("  If similar to real: discriminant structure is meaningless.\n\n");

    printf("  %-10s  %-14s  %-14s  %-10s\n",
           "Channel", "Real-disc", "Random-disc", "Delta");
    printf("  %-10s  %-14s  %-14s  %-10s\n",
           "----------", "--------------", "--------------", "----------");

    for (int ci = 0; ci < 4; ci++) {
        int ch = probe_channels[ci];
        float real_sum = 0, rand_sum = 0;

        for (int run = 0; run < N_RUNS; run++) {
            /* Real discriminant */
            Discriminant disc_real;
            {
                unsigned int seed = 42 + run * 97;
                float samples[MAX_SAMPLES][HIDDEN_DIM];
                int ns = collect_samples(ch, dt, enroll_steps, &seed,
                                         tau_iss, samples, MAX_SAMPLES);
                learn_disc(samples, ns, &disc_real);
            }

            /* Random discriminant */
            Discriminant disc_rand;
            {
                unsigned int seed = 55555 + run * 37;
                for (int i = 0; i < HIDDEN_DIM; i++) {
                    disc_rand.mean[i] = gaussf(&seed) * 0.5f;
                    disc_rand.dim_std[i] = 0.1f + 0.5f * randf(&seed);
                }
                for (int pc = 0; pc < N_PCS; pc++) {
                    for (int i = 0; i < HIDDEN_DIM; i++)
                        disc_rand.pcs[pc][i] = gaussf(&seed);
                    vec_normalize(disc_rand.pcs[pc], HIDDEN_DIM);
                    disc_rand.pc_mean[pc] = gaussf(&seed) * 0.1f;
                    disc_rand.pc_std[pc] = 0.1f + 0.5f * randf(&seed);
                }
                disc_rand.valid = 1;
            }

            /* Score same test data against both */
            unsigned int test_seed = 9999 + run * 131;
            unsigned int test_seed2 = test_seed;

            float h1[HIDDEN_DIM] = {0}, h2[HIDDEN_DIM] = {0};
            float s1 = 0, s2 = 0;
            int scored = 0;

            for (int step = 0; step < test_steps; step++) {
                float t = step * dt;
                float v1 = gen_channel(ch, t, &test_seed);
                float v2 = gen_channel(ch, t, &test_seed2);
                cfc_step(v1, dt, h1, tau_iss);
                cfc_step(v2, dt, h2, tau_iss);
                if (step > WARMUP) {
                    s1 += score_hybrid(h1, &disc_real);
                    s2 += score_hybrid(h2, &disc_rand);
                    scored++;
                }
            }
            real_sum += s1 / scored;
            rand_sum += s2 / scored;
        }

        printf("  %-10s  %-14.4f  %-14.4f  %+.4f\n",
               probe_names[ci], real_sum / N_RUNS, rand_sum / N_RUNS,
               rand_sum / N_RUNS - real_sum / N_RUNS);
    }

    /* ══════════════════════════════════════════════════════════════════════
     * TEST 6: THRESHOLD ROC
     *
     * Sweep thresholds. For each: compute TPR (anomaly correctly flagged)
     * and FPR (normal incorrectly flagged). Is 0.35 special or arbitrary?
     * ══════════════════════════════════════════════════════════════════════ */
    printf("\nTEST 6: THRESHOLD ROC (CMG1, step anomaly, %d runs)\n\n", N_RUNS);

    /* Collect normal and anomaly final scores */
    float normal_scores[100], anomaly_scores[100];
    int n_collected = 0;

    for (int run = 0; run < N_RUNS && n_collected < 100; run++) {
        unsigned int enroll_seed = 42 + run * 97;
        float samples[MAX_SAMPLES][HIDDEN_DIM];
        int ns = collect_samples(0, dt, enroll_steps, &enroll_seed,
                                 tau_iss, samples, MAX_SAMPLES);
        Discriminant disc;
        learn_disc(samples, ns, &disc);

        /* Normal */
        {
            unsigned int seed = 9999 + run * 131;
            float h[HIDDEN_DIM];
            get_final_h(0, dt, test_steps, &seed, tau_iss, h);
            normal_scores[n_collected] = score_hybrid(h, &disc);
        }

        /* Anomaly (step, injected at midpoint) */
        {
            unsigned int seed = 9999 + run * 131;
            float h[HIDDEN_DIM];
            get_final_h_anomaly(0, dt, test_steps, test_steps/2, 0,
                                &seed, tau_iss, h);
            anomaly_scores[n_collected] = score_hybrid(h, &disc);
        }

        n_collected++;
    }

    printf("  Normal scores:  min=%.4f  max=%.4f  mean=%.4f\n",
           ({ float mn=1; for(int i=0;i<n_collected;i++) if(normal_scores[i]<mn) mn=normal_scores[i]; mn; }),
           ({ float mx=0; for(int i=0;i<n_collected;i++) if(normal_scores[i]>mx) mx=normal_scores[i]; mx; }),
           ({ float s=0; for(int i=0;i<n_collected;i++) s+=normal_scores[i]; s/n_collected; }));
    printf("  Anomaly scores: min=%.4f  max=%.4f  mean=%.4f\n\n",
           ({ float mn=1; for(int i=0;i<n_collected;i++) if(anomaly_scores[i]<mn) mn=anomaly_scores[i]; mn; }),
           ({ float mx=0; for(int i=0;i<n_collected;i++) if(anomaly_scores[i]>mx) mx=anomaly_scores[i]; mx; }),
           ({ float s=0; for(int i=0;i<n_collected;i++) s+=anomaly_scores[i]; s/n_collected; }));

    printf("  %-10s  %-8s  %-8s  %-8s\n", "Threshold", "TPR", "FPR", "Note");
    printf("  %-10s  %-8s  %-8s  %-8s\n", "----------", "--------", "--------", "--------");

    float thresholds[] = {0.10f, 0.20f, 0.30f, 0.35f, 0.40f, 0.50f, 0.60f, 0.70f, 0.80f};
    for (int ti = 0; ti < 9; ti++) {
        float thresh = thresholds[ti];
        int tp = 0, fp = 0;
        for (int i = 0; i < n_collected; i++) {
            if (anomaly_scores[i] < thresh) tp++;
            if (normal_scores[i] < thresh) fp++;
        }
        float tpr = (float)tp / n_collected;
        float fpr = (float)fp / n_collected;
        printf("  %-10.2f  %-8.2f  %-8.2f  %s\n",
               thresh, tpr, fpr,
               thresh == 0.35f ? "<-- CURRENT" : "");
    }

    printf("\n================================================================\n");
    printf("  VERDICT: Check each test for failures.\n");
    printf("  A healthy system should show:\n");
    printf("    Test 1: Independent seeds similar to continuous (<0.1 delta)\n");
    printf("    Test 2: Noise-enrolled scores LOWER than ISS-enrolled\n");
    printf("    Test 3: Wrong-channel scores LOWER than same-channel\n");
    printf("    Test 4: ISS tau outperforms keystroke tau on detection\n");
    printf("    Test 5: Random disc scores LOWER than real disc\n");
    printf("    Test 6: Clean separation in ROC, 0.35 not cherry-picked\n");
    printf("================================================================\n");

    return 0;
}
