/*
 * ISS Telemetry Anomaly Detection — Yinsen Chip Stack Demo
 *
 * Multi-channel CfC processor for real-time ISS telemetry anomaly detection.
 * Each telemetry parameter gets its own CfC cell + discriminant. Cross-channel
 * correlation detects compound anomalies (e.g., CMG vibration spike + bearing
 * temp rise = real problem, not noise).
 *
 * Pipeline:
 *   Enrollment: stream "normal" data -> per-channel CfC -> collect hidden states
 *               -> learn discriminant (mean + PCA) per channel
 *   Execution:  stream live data -> per-channel CfC -> score against discriminant
 *               -> cross-channel aggregate -> anomaly flag
 *
 * Two modes:
 *   --sim    Synthetic ISS telemetry (orbital thermal cycling, CMG bearings,
 *            cabin atmosphere) with injected anomalies. No network needed.
 *   --stdin  Reads (channel_id, timestamp, value) from stdin. Pipe from
 *            scripts/iss_websocket.py for live ISS data via Lightstreamer.
 *
 * Channels (simulation):
 *   0-3: CMG 1-4 vibration (g-force, bearing health)
 *   4-5: Coolant Loop A/B temperature (C)
 *   6:   Cabin pressure (kPa)
 *   7:   O2 partial pressure (kPa)
 *
 * Key numbers:
 *   Per-channel state:  324 bytes (hidden + discriminant + stats)
 *   Shared weights:     736 bytes
 *   8-channel total:    3,328 bytes = 5.1% of 64KB L1 cache
 *   Per-sample latency: ~55ns CfC + ~11ns scoring per channel
 *   Discriminant:       268 bytes per channel
 *
 * FALSIFICATION RECORD:
 *
 *   Probes 1-2 (v1, in experiments/iss_probes/):
 *     PASSED: Seed independence, channel specificity, random discriminant, ROC.
 *     FAILED: Tau ablation (identical under constant dt), CMG hidden state
 *       degenerate (H-Std=0.0007), CfC no advantage over 3-sigma.
 *
 *   Probe 3 (v2 with pre-scaling, in experiments/iss_probes/):
 *     PASSED (4/5 criteria):
 *       - CMG H-Std: 0.0007 -> 0.078 (4,970x improvement)
 *       - Noise-enrolled CMG: 0.799 -> 0.041 (below 0.50 target)
 *       - CfC beats 3-sigma: CoolA 20/20 vs 0/20, CMG 5% mag 20/20 vs 9/20
 *       - Same-subsystem delta: -0.077 -> -0.270 (3.5x improvement)
 *     STILL FAILED:
 *       - Tau ablation under constant dt=10s (all configs identical)
 *       - Prediction: variable dt from real ISS data should differentiate tau
 *
 *   REAL ISS TELEMETRY (live Lightstreamer, 2026-02-01):
 *     Connected to push.lightstreamer.com/ISSLIVE, 8 channels, variable dt.
 *     Calibration on real data: CMG wheel speed (744.7), spin current (17.3),
 *       ETCS temps (238-322°F), cabin pressure (743 mmHg), CO2 (2.26).
 *     Variable dt confirmed: 1.0-4.7s across channels (tau test viable).
 *     CabinP CfC converged to 0.84 on real data (genuine discriminant).
 *     CMG channels show active CfC dynamics (0.00-0.82 range).
 *     ETCS/ppCO2 need longer calibration (>1 orbit for orbital cycling).
 *     384 detection samples captured over 540s from the ISS.
 *
 * Chips used:
 *   - cfc_cell_chip.h    CfC temporal dynamics (hidden_dim=8)
 *   - norm_chip.h        Drift monitoring (NOT in CfC input path)
 *   - activation_chip.h  Sigmoid for score mapping
 *   - fft_chip.h         Spectral features (available, not yet integrated)
 *
 * Compile:
 *   cc -O2 -I include -I include/chips examples/iss_telemetry.c -lm -o examples/iss_telemetry
 *
 * Usage:
 *   ./examples/iss_telemetry --sim              # synthetic telemetry with anomalies
 *   ./examples/iss_telemetry --sim --verbose     # show per-channel detail
 *   python scripts/iss_websocket.py | ./examples/iss_telemetry --stdin
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
#include "chips/fft_chip.h"

/* ═══════════════════════════════════════════════════════════════════════════
 * Network Dimensions — same as keystroke biometric
 *
 * input_dim=2:  (value, dt)
 * hidden_dim=8: CfC hidden state
 * ═══════════════════════════════════════════════════════════════════════════ */
#define INPUT_DIM    2
#define HIDDEN_DIM   8
#define CONCAT_DIM   (INPUT_DIM + HIDDEN_DIM)  /* 10 */

/* ═══════════════════════════════════════════════════════════════════════════
 * Channel Configuration
 * ═══════════════════════════════════════════════════════════════════════════ */
#define MAX_CHANNELS     16
#define SIM_CHANNELS     8

/* Discriminant parameters (same sweet spot as keystroke) */
#define ISS_N_PCS            5
#define ISS_WARMUP           20     /* more warmup for slower telemetry */
#define ISS_MAX_SAMPLES      500    /* longer enrollment for orbital cycles */
#define ISS_POWER_ITERS      20
#define ISS_ENROLL_SAMPLES   300    /* target enrollment length */

/* Scoring */
#define ISS_MEAN_WEIGHT      0.3f
#define ISS_PCA_WEIGHT       0.7f
#define ISS_ANOMALY_THRESH   0.35f  /* below this = anomaly */
#define ISS_CROSS_THRESH     3      /* N channels anomalous = compound anomaly */

/* ═══════════════════════════════════════════════════════════════════════════
 * Simulation Parameters
 *
 * ISS orbital period: ~92 minutes = 5520 seconds
 * Telemetry sample rate: varies, but typically 1-10 Hz
 * Simulation dt: 10 seconds (0.1 Hz — conservative for demo)
 * ═══════════════════════════════════════════════════════════════════════════ */
#define SIM_DT               10.0f     /* seconds between samples */
#define SIM_ORBITAL_PERIOD   5520.0f   /* seconds (~92 min) */
#define SIM_ENROLL_ORBITS    2.0f      /* 2 full orbits for enrollment */
#define SIM_TEST_ORBITS      1.0f      /* 1 orbit for normal test */
#define SIM_ANOMALY_ORBITS   1.0f      /* 1 orbit with injected anomalies */

/* ═══════════════════════════════════════════════════════════════════════════
 * Pre-initialized Weights
 *
 * Same weight structure as keystroke, but with tau values tuned for
 * ISS telemetry timescales (seconds to minutes, not milliseconds).
 *
 * In production: train on actual ISS telemetry. For demo: structurally
 * meaningful initialization that creates reasonable CfC dynamics.
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

/* Time constants designed for ISS telemetry timescales.
 * ISS data arrives at ~0.1-10 Hz. Orbital period is ~92 min.
 *
 * FALSIFICATION NOTE (probe 1, test 4): tau values do NOT affect
 * detection performance at current anomaly magnitudes. ISS tau
 * (5-600s), keystroke tau (0.05-0.8s), and constant tau (30s) all
 * produce identical results. At 20% CMG anomaly, ISS tau shows
 * slight advantage (20/20 vs 8/20) — inconclusive.
 *
 * UPDATE (seismic_detector.c tau ablation, Feb 2026): tau tuning IS
 * validated on seismic data where decay dynamic range matches signal
 * temporal structure. Seismic tau detects M2.0 events 2.2x faster
 * than ISS/constant tau. The key metric is R = max(decay)/min(decay):
 * ISS R=7x (insufficient), seismic R=2700x (sufficient). Variable dt
 * from real ISS data remains untested for tau differentiation.
 */
static const float tau[HIDDEN_DIM] = {
    5.0f, 15.0f, 45.0f, 120.0f, 10.0f, 30.0f, 90.0f, 600.0f
};

/* ═══════════════════════════════════════════════════════════════════════════
 * Discriminant — same structure as keystroke, per channel
 * ═══════════════════════════════════════════════════════════════════════════ */

typedef struct {
    float mean[HIDDEN_DIM];                     /* 32 bytes */
    float dim_std[HIDDEN_DIM];                  /* 32 bytes */
    float pcs[ISS_N_PCS][HIDDEN_DIM];           /* 160 bytes */
    float pc_mean[ISS_N_PCS];                   /* 20 bytes */
    float pc_std[ISS_N_PCS];                    /* 20 bytes */
    int valid;                                   /* 4 bytes */
} TelemetryDiscriminant;  /* 268 bytes */

/* ═══════════════════════════════════════════════════════════════════════════
 * Per-Channel Calibration (v2 — LMM synthesis)
 *
 * Frozen affine transform learned during calibration phase.
 * Maps raw sensor values into the CfC's useful input range (~[-3,+3]).
 *
 * The keystroke lesson: FIXED transforms are safe. MOVING transforms
 * (Welford online) destroy hidden state consistency (probe2: 0.00 vs
 * 0.90 cosine). These parameters are computed once and never change.
 * ═══════════════════════════════════════════════════════════════════════════ */

typedef struct {
    float input_mean;       /* 4 bytes — calibration mean */
    float input_std;        /* 4 bytes — calibration std */
    float dt_mean;          /* 4 bytes — mean update interval */
    int calibrated;         /* 4 bytes — calibration complete? */
    int n_samples;          /* 4 bytes — samples seen during calibration */
    /* 3-sigma baseline (for honest comparison) */
    float sigma3_hi;        /* 4 bytes — mean + 3*std of raw values */
    float sigma3_lo;        /* 4 bytes — mean - 3*std of raw values */
} ChannelCalibration;       /* 28 bytes */

/* ═══════════════════════════════════════════════════════════════════════════
 * Per-Channel State
 *
 * Everything one channel needs for execution.
 * Weights are shared across all channels (same CfC architecture).
 *   hidden:        32 bytes
 *   discriminant: 268 bytes
 *   calibration:   28 bytes
 *   Welford:       24 bytes (2 channels * 12 bytes)
 *   Total:        352 bytes per channel (plus shared 736 bytes weights)
 * ═══════════════════════════════════════════════════════════════════════════ */

typedef struct {
    char name[32];                   /* human-readable label */
    float h_state[HIDDEN_DIM];       /* CfC hidden state */
    TelemetryDiscriminant disc;      /* learned discriminant */
    ChannelCalibration cal;          /* frozen pre-scaling parameters */
    RunningStats input_stats[INPUT_DIM]; /* drift monitoring */
    float last_score;                /* most recent anomaly score */
    float last_3sigma;               /* most recent 3-sigma flag (0 or 1) */
    int sample_count;                /* total samples processed */
    int enrolled;                    /* enrollment complete? */

    /* Enrollment sample buffer */
    float enroll_buf[ISS_MAX_SAMPLES][HIDDEN_DIM];
    int n_enroll;
} TelemetryChannel;

/* ═══════════════════════════════════════════════════════════════════════════
 * Math Helpers (same as keystroke)
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
    if (norm > 1e-10f)
        for (int i = 0; i < n; i++) v[i] /= norm;
    return norm;
}

/* ═══════════════════════════════════════════════════════════════════════════
 * Core Functions
 * ═══════════════════════════════════════════════════════════════════════════ */

static void channel_init(TelemetryChannel *ch, const char *name) {
    memset(ch, 0, sizeof(*ch));
    strncpy(ch->name, name, sizeof(ch->name) - 1);
    for (int i = 0; i < INPUT_DIM; i++)
        RUNNING_STATS_INIT(&ch->input_stats[i]);
}

/**
 * Pre-scale a raw sensor value using frozen calibration parameters.
 * Maps the channel's operating range to roughly [-3, +3].
 * Safe: these are constants, not online statistics (keystroke lesson).
 */
static float prescale_value(float raw, const ChannelCalibration *cal) {
    if (!cal->calibrated) return raw;
    return (raw - cal->input_mean) / (cal->input_std + 1e-8f);
}

static float prescale_dt(float dt, const ChannelCalibration *cal) {
    if (!cal->calibrated || cal->dt_mean < 1e-6f) return dt;
    return dt / cal->dt_mean;
}

/**
 * Step one channel's CfC on a new telemetry reading.
 * v2: applies frozen pre-scaling from calibration phase.
 */
static void channel_step(TelemetryChannel *ch, float value, float dt) {
    float scaled_val = prescale_value(value, &ch->cal);
    float scaled_dt = prescale_dt(dt, &ch->cal);
    float input[INPUT_DIM] = { scaled_val, scaled_dt };
    float h_new[HIDDEN_DIM];

    CFC_CELL_GENERIC(
        input, ch->h_state, dt,  /* real dt for decay computation */
        W_gate, b_gate, W_cand, b_cand,
        tau, 0,
        INPUT_DIM, HIDDEN_DIM,
        h_new
    );

    memcpy(ch->h_state, h_new, HIDDEN_DIM * sizeof(float));
    ch->sample_count++;

    /* Update drift monitor (not in CfC path) */
    RUNNING_STATS_UPDATE(&ch->input_stats[0], value);
    RUNNING_STATS_UPDATE(&ch->input_stats[1], dt);

    /* 3-sigma baseline check (for honest comparison) */
    if (ch->cal.calibrated) {
        ch->last_3sigma = (value > ch->cal.sigma3_hi || value < ch->cal.sigma3_lo)
                          ? 1.0f : 0.0f;
    }

    /* Collect enrollment samples after warmup */
    if (!ch->enrolled && ch->sample_count > ISS_WARMUP
        && ch->n_enroll < ISS_MAX_SAMPLES) {
        memcpy(ch->enroll_buf[ch->n_enroll], ch->h_state,
               HIDDEN_DIM * sizeof(float));
        ch->n_enroll++;
    }
}

/**
 * Learn discriminant from enrollment samples — identical to keystroke.
 */
static void learn_discriminant(
    const float samples[][HIDDEN_DIM],
    int n_samples,
    TelemetryDiscriminant *disc
) {
    memset(disc, 0, sizeof(*disc));
    if (n_samples < 5) { disc->valid = 0; return; }

    /* Mean */
    for (int i = 0; i < HIDDEN_DIM; i++) {
        float s = 0;
        for (int t = 0; t < n_samples; t++) s += samples[t][i];
        disc->mean[i] = s / n_samples;
    }

    /* Per-dim std */
    for (int i = 0; i < HIDDEN_DIM; i++) {
        float s = 0;
        for (int t = 0; t < n_samples; t++) {
            float d = samples[t][i] - disc->mean[i];
            s += d * d;
        }
        disc->dim_std[i] = sqrtf(s / n_samples + 1e-8f);
    }

    /* Center */
    float centered[n_samples][HIDDEN_DIM];
    for (int t = 0; t < n_samples; t++)
        for (int i = 0; i < HIDDEN_DIM; i++)
            centered[t][i] = samples[t][i] - disc->mean[i];

    /* Extract PCs via power iteration with deflation */
    for (int pc = 0; pc < ISS_N_PCS; pc++) {
        float v[HIDDEN_DIM];
        for (int i = 0; i < HIDDEN_DIM; i++)
            v[i] = centered[0][i] + 0.01f * (i + 1);
        vec_normalize(v, HIDDEN_DIM);

        for (int iter = 0; iter < ISS_POWER_ITERS; iter++) {
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

        /* Projection stats */
        float proj_sum = 0, proj_sum2 = 0;
        for (int t = 0; t < n_samples; t++) {
            float p = dotf(centered[t], v, HIDDEN_DIM);
            proj_sum += p;
            proj_sum2 += p * p;
        }
        disc->pc_mean[pc] = proj_sum / n_samples;
        float var = proj_sum2 / n_samples - disc->pc_mean[pc] * disc->pc_mean[pc];
        disc->pc_std[pc] = sqrtf(var > 0 ? var : 1e-8f);

        /* Deflate */
        for (int t = 0; t < n_samples; t++) {
            float p = dotf(centered[t], v, HIDDEN_DIM);
            for (int i = 0; i < HIDDEN_DIM; i++)
                centered[t][i] -= p * v[i];
        }
    }

    disc->valid = 1;
}

/**
 * Finalize enrollment for a channel.
 */
static void channel_finalize_enrollment(TelemetryChannel *ch) {
    learn_discriminant(
        (const float (*)[HIDDEN_DIM])ch->enroll_buf,
        ch->n_enroll, &ch->disc
    );
    ch->enrolled = 1;

    /* Reset hidden state for clean execution phase */
    memset(ch->h_state, 0, sizeof(ch->h_state));
    ch->sample_count = 0;
}

/**
 * Score current hidden state against discriminant.
 * Same hybrid scoring as keystroke v3.
 */
static float channel_score(const TelemetryChannel *ch) {
    if (!ch->disc.valid) return 0.5f;

    /* Mean distance */
    float mean_dist = 0;
    float centered[HIDDEN_DIM];
    for (int i = 0; i < HIDDEN_DIM; i++) {
        centered[i] = ch->h_state[i] - ch->disc.mean[i];
        float z = centered[i] / (ch->disc.dim_std[i] + 1e-8f);
        mean_dist += z * z;
    }
    mean_dist /= HIDDEN_DIM;
    float mean_score = SIGMOID_CHIP(2.0f - mean_dist);

    /* PCA distance */
    float pca_dist = 0;
    for (int pc = 0; pc < ISS_N_PCS; pc++) {
        float proj = dotf(centered, ch->disc.pcs[pc], HIDDEN_DIM);
        float z = (proj - ch->disc.pc_mean[pc]) / (ch->disc.pc_std[pc] + 1e-8f);
        pca_dist += z * z;
    }
    pca_dist /= ISS_N_PCS;
    float pca_score = SIGMOID_CHIP(2.0f - pca_dist);

    return ISS_MEAN_WEIGHT * mean_score + ISS_PCA_WEIGHT * pca_score;
}

/* ═══════════════════════════════════════════════════════════════════════════
 * Cross-Channel Correlation
 *
 * The real value of multi-channel monitoring: correlated anomalies across
 * related subsystems are far more diagnostic than single-channel deviations.
 *
 * Example: CMG1 vibration drops + CMG1 bearing temp rises = bearing issue
 *          vs. CMG1 vibration spike alone = maybe just a micrometeorite bump
 * ═══════════════════════════════════════════════════════════════════════════ */

typedef struct {
    int n_channels;
    int n_anomalous;           /* channels below threshold this step */
    int compound_anomaly;      /* n_anomalous >= ISS_CROSS_THRESH */
    float aggregate_score;     /* mean score across all channels */
    float min_score;           /* worst channel */
    int min_channel;           /* which channel is worst */
} CrossChannelResult;

static CrossChannelResult cross_channel_score(
    TelemetryChannel *channels, int n_channels
) {
    CrossChannelResult r;
    memset(&r, 0, sizeof(r));
    r.n_channels = n_channels;
    r.min_score = 1.0f;

    float sum = 0;
    for (int i = 0; i < n_channels; i++) {
        float s = channels[i].last_score;
        sum += s;
        if (s < r.min_score) {
            r.min_score = s;
            r.min_channel = i;
        }
        if (s < ISS_ANOMALY_THRESH) {
            r.n_anomalous++;
        }
    }

    r.aggregate_score = sum / n_channels;
    r.compound_anomaly = (r.n_anomalous >= ISS_CROSS_THRESH);
    return r;
}

/* ═══════════════════════════════════════════════════════════════════════════
 * Simulation: Synthetic ISS Telemetry
 *
 * Generates physically-motivated telemetry patterns:
 *
 * CMG vibration (ch 0-3):
 *   Baseline: 0.001g with orbital modulation (thermal expansion)
 *   Noise: Gaussian ~0.0002g
 *   Anomaly: bearing wear → slow vibration increase over time
 *
 * Coolant temperature (ch 4-5):
 *   Baseline: 15°C with orbital cycling (sun/eclipse)
 *   Amplitude: ±8°C orbital, ±0.5°C noise
 *   Anomaly: coolant leak → steady temperature rise
 *
 * Cabin pressure (ch 6):
 *   Baseline: 101.3 kPa (1 atm)
 *   Noise: ±0.05 kPa (pressure regulation)
 *   Anomaly: slow leak → linear pressure drop
 *
 * O2 partial pressure (ch 7):
 *   Baseline: 21.3 kPa (standard O2)
 *   Noise: ±0.1 kPa (CDRA/OGS cycling)
 *   Anomaly: scrubber degradation → O2 drop + CO2 proxy rise
 * ═══════════════════════════════════════════════════════════════════════════ */

/* Simple Gaussian from uniform (Box-Muller) */
static float randf(unsigned int *s) {
    *s = *s * 1103515245u + 12345u;
    return (float)(*s & 0x7FFFFFFFu) / (float)0x7FFFFFFFu;
}

static float gaussf(unsigned int *s) {
    float u1 = randf(s) + 1e-10f;
    float u2 = randf(s);
    return sqrtf(-2.0f * logf(u1)) * cosf(6.28318530718f * u2);
}

typedef struct {
    float time;              /* mission elapsed time (seconds) */
    int anomaly_active;      /* is an anomaly currently injected? */
    float anomaly_start;     /* when the anomaly began */
    int anomaly_channel;     /* which channel is anomalous (-1 = none) */
    int anomaly_type;        /* 0=step, 1=ramp, 2=oscillation shift */
    unsigned int seed;
} SimState;

/**
 * Generate one telemetry reading for a given channel at a given time.
 * Returns the raw sensor value in physical units.
 */
static float sim_generate(SimState *sim, int channel) {
    float t = sim->time;
    float orbital_phase = 2.0f * 3.14159265f * t / SIM_ORBITAL_PERIOD;
    float noise = gaussf(&sim->seed);
    float value = 0;

    switch (channel) {
    case 0: case 1: case 2: case 3: {
        /* CMG vibration — g-force on bearing housing */
        float base = 0.001f;  /* 1 milli-g baseline */
        float orbital_mod = 0.0002f * sinf(orbital_phase);  /* thermal */
        float unit_noise = 0.0002f * noise;
        /* Each CMG has slightly different baseline */
        float cmg_offset = 0.0001f * (float)channel;
        value = base + cmg_offset + orbital_mod + unit_noise;
        break;
    }
    case 4: case 5: {
        /* Coolant loop temperature — Celsius */
        float base = 15.0f;
        float orbital_mod = 8.0f * sinf(orbital_phase);  /* sun/eclipse */
        /* Loop B runs slightly warmer */
        float loop_offset = (channel == 5) ? 2.0f : 0.0f;
        float unit_noise = 0.5f * noise;
        value = base + loop_offset + orbital_mod + unit_noise;
        break;
    }
    case 6: {
        /* Cabin pressure — kPa */
        float base = 101.3f;
        float unit_noise = 0.05f * noise;
        /* Slight orbital pressure variation from thermal expansion */
        float orbital_mod = 0.02f * sinf(orbital_phase);
        value = base + orbital_mod + unit_noise;
        break;
    }
    case 7: {
        /* O2 partial pressure — kPa */
        float base = 21.3f;
        float unit_noise = 0.1f * noise;
        /* OGS/CDRA cycling creates a slow sawtooth */
        float cdra_phase = fmodf(t, 900.0f) / 900.0f;  /* 15-min cycle */
        float cdra_mod = 0.15f * (cdra_phase - 0.5f);
        value = base + cdra_mod + unit_noise;
        break;
    }
    default:
        value = noise;
    }

    /* Inject anomaly if active and targeting this channel */
    if (sim->anomaly_active && sim->anomaly_channel == channel) {
        float dt_anom = t - sim->anomaly_start;

        switch (sim->anomaly_type) {
        case 0: /* Step change — sudden offset */
            switch (channel) {
            case 0: case 1: case 2: case 3:
                value += 0.005f;  /* 5x vibration spike */
                break;
            case 4: case 5:
                value += 5.0f;    /* 5°C jump */
                break;
            case 6:
                value -= 0.5f;    /* pressure drop */
                break;
            case 7:
                value -= 1.0f;    /* O2 drop */
                break;
            }
            break;

        case 1: /* Ramp — gradual degradation */
            {
                float ramp = dt_anom / 3000.0f;  /* ramp over ~50 min */
                if (ramp > 1.0f) ramp = 1.0f;
                switch (channel) {
                case 0: case 1: case 2: case 3:
                    value += 0.008f * ramp;  /* bearing degradation */
                    break;
                case 4: case 5:
                    value += 8.0f * ramp;    /* coolant loss */
                    break;
                case 6:
                    value -= 1.0f * ramp;    /* slow leak */
                    break;
                case 7:
                    value -= 2.0f * ramp;    /* scrubber failure */
                    break;
                }
            }
            break;

        case 2: /* Oscillation shift — frequency/amplitude change */
            {
                float anomaly_osc = sinf(2.0f * 3.14159265f * dt_anom / 200.0f);
                switch (channel) {
                case 0: case 1: case 2: case 3:
                    value += 0.003f * anomaly_osc;  /* new vibration mode */
                    break;
                case 4: case 5:
                    value += 3.0f * anomaly_osc;    /* thermal oscillation */
                    break;
                case 6:
                    value += 0.3f * anomaly_osc;    /* pressure oscillation */
                    break;
                case 7:
                    value += 0.5f * anomaly_osc;    /* O2 oscillation */
                    break;
                }
            }
            break;
        }
    }

    return value;
}

/* ═══════════════════════════════════════════════════════════════════════════
 * Print Helpers
 * ═══════════════════════════════════════════════════════════════════════════ */

static const char* channel_names[SIM_CHANNELS] = {
    "CMG1-vib", "CMG2-vib", "CMG3-vib", "CMG4-vib",
    "CoolA-T",  "CoolB-T",  "CabinP",   "O2-PP"
};

static const char* channel_units[SIM_CHANNELS] = {
    "g", "g", "g", "g", "C", "C", "kPa", "kPa"
};

static void print_score_bar(float score, int width) {
    int filled = (int)(score * width);
    if (filled < 0) filled = 0;
    if (filled > width) filled = width;

    printf("[");
    for (int i = 0; i < width; i++) {
        if (i < filled) printf("=");
        else if (i == filled) printf(">");
        else printf(" ");
    }
    printf("]");
}

static void print_header(void) {
    printf("  %-9s ", "Channel");
    printf("%-8s ", "Score");
    printf("%-22s ", "Bar");
    printf("%-8s", "Status");
    printf("\n");
    printf("  %-9s ", "---------");
    printf("%-8s ", "--------");
    printf("%-22s ", "----------------------");
    printf("%-8s", "--------");
    printf("\n");
}

/* ═══════════════════════════════════════════════════════════════════════════
 * Simulation Mode
 * ═══════════════════════════════════════════════════════════════════════════ */

static void run_simulation(int verbose) {
    printf("=================================================================\n");
    printf("  ISS Telemetry Anomaly Detection v2 — Yinsen Chip Stack\n");
    printf("=================================================================\n\n");

    printf("  %d independent CfC channels, each with its own discriminant.\n", SIM_CHANNELS);
    printf("  v2: frozen pre-scaling from calibration phase (LMM fix).\n");
    printf("  Maps raw sensor range to CfC useful input range (~[-3,+3]).\n\n");

    printf("  Channels:\n");
    for (int i = 0; i < SIM_CHANNELS; i++)
        printf("    %d: %-10s (%s)\n", i, channel_names[i], channel_units[i]);
    printf("\n");

    /* Initialize channels */
    TelemetryChannel channels[SIM_CHANNELS];
    for (int i = 0; i < SIM_CHANNELS; i++)
        channel_init(&channels[i], channel_names[i]);

    SimState sim;
    memset(&sim, 0, sizeof(sim));
    sim.seed = 42;
    sim.anomaly_channel = -1;

    /* ── Phase 0: CALIBRATION (v2 — compute per-channel input statistics) ──
     *
     * Stream 1 orbit of normal telemetry, collecting raw values per channel.
     * Compute mean and std, freeze as calibration constants.
     * These are NEVER updated again (Welford lesson from keystroke).
     */
    printf("PHASE 0: CALIBRATION (1 orbit = %.0f samples)\n",
           SIM_ORBITAL_PERIOD / SIM_DT);

    float cal_duration = SIM_ORBITAL_PERIOD;  /* 1 full orbit */
    int cal_steps = (int)(cal_duration / SIM_DT);

    /* Accumulate raw statistics per channel (Welford online) */
    double cal_sum[SIM_CHANNELS], cal_sum2[SIM_CHANNELS];
    int cal_n[SIM_CHANNELS];
    memset(cal_sum, 0, sizeof(cal_sum));
    memset(cal_sum2, 0, sizeof(cal_sum2));
    memset(cal_n, 0, sizeof(cal_n));

    for (int step = 0; step < cal_steps; step++) {
        sim.time = step * SIM_DT;
        for (int ch = 0; ch < SIM_CHANNELS; ch++) {
            float value = sim_generate(&sim, ch);
            cal_sum[ch] += (double)value;
            cal_sum2[ch] += (double)value * (double)value;
            cal_n[ch]++;
        }
    }

    /* Freeze calibration parameters */
    for (int ch = 0; ch < SIM_CHANNELS; ch++) {
        ChannelCalibration *cal = &channels[ch].cal;
        cal->input_mean = (float)(cal_sum[ch] / cal_n[ch]);
        float var = (float)(cal_sum2[ch] / cal_n[ch]
                           - (cal_sum[ch] / cal_n[ch]) * (cal_sum[ch] / cal_n[ch]));
        cal->input_std = sqrtf(var > 0 ? var : 1e-8f);
        cal->dt_mean = SIM_DT;  /* constant dt in simulation */
        cal->n_samples = cal_n[ch];
        cal->sigma3_hi = cal->input_mean + 3.0f * cal->input_std;
        cal->sigma3_lo = cal->input_mean - 3.0f * cal->input_std;
        cal->calibrated = 1;
    }

    printf("  Calibration results (frozen per-channel pre-scaling):\n");
    printf("    %-10s  %12s  %12s  %12s  %12s\n",
           "Channel", "Mean", "Std", "3sig_lo", "3sig_hi");
    for (int ch = 0; ch < SIM_CHANNELS; ch++) {
        ChannelCalibration *cal = &channels[ch].cal;
        printf("    %-10s  %12.6f  %12.6f  %12.6f  %12.6f\n",
               channel_names[ch], cal->input_mean, cal->input_std,
               cal->sigma3_lo, cal->sigma3_hi);
    }

    /* ── Phase 1: Enrollment (with pre-scaling active) ── */
    printf("\nPHASE 1: ENROLLMENT (%.0f orbits = %.0f samples, pre-scaling ON)\n",
           SIM_ENROLL_ORBITS,
           SIM_ENROLL_ORBITS * SIM_ORBITAL_PERIOD / SIM_DT);

    /* Re-init channel state (keep calibration) */
    for (int i = 0; i < SIM_CHANNELS; i++) {
        ChannelCalibration saved_cal = channels[i].cal;
        channel_init(&channels[i], channel_names[i]);
        channels[i].cal = saved_cal;
    }

    float enroll_duration = SIM_ENROLL_ORBITS * SIM_ORBITAL_PERIOD;
    int enroll_steps = (int)(enroll_duration / SIM_DT);

    /* Track hidden state statistics during enrollment for diagnostics */
    double h_sum[SIM_CHANNELS][HIDDEN_DIM], h_sum2[SIM_CHANNELS][HIDDEN_DIM];
    int h_n[SIM_CHANNELS];
    memset(h_sum, 0, sizeof(h_sum));
    memset(h_sum2, 0, sizeof(h_sum2));
    memset(h_n, 0, sizeof(h_n));

    for (int step = 0; step < enroll_steps; step++) {
        sim.time = cal_duration + step * SIM_DT;
        for (int ch = 0; ch < SIM_CHANNELS; ch++) {
            float value = sim_generate(&sim, ch);
            channel_step(&channels[ch], value, SIM_DT);

            /* Collect hidden state stats after warmup */
            if (step > ISS_WARMUP) {
                for (int d = 0; d < HIDDEN_DIM; d++) {
                    h_sum[ch][d] += (double)channels[ch].h_state[d];
                    h_sum2[ch][d] += (double)channels[ch].h_state[d]
                                   * (double)channels[ch].h_state[d];
                }
                h_n[ch]++;
            }
        }
    }

    /* Finalize enrollment */
    for (int ch = 0; ch < SIM_CHANNELS; ch++)
        channel_finalize_enrollment(&channels[ch]);

    printf("  Enrolled all %d channels.\n", SIM_CHANNELS);
    for (int ch = 0; ch < SIM_CHANNELS; ch++) {
        printf("    %-10s: %d samples, discriminant %s\n",
               channels[ch].name, channels[ch].n_enroll,
               channels[ch].disc.valid ? "VALID" : "INVALID");
    }

    /* ── Hidden state diagnostics (v2 key metric) ──
     * CMG H-Std MUST increase from v1's 0.0007 to >0.01 for the fix to work.
     * If it doesn't, pre-scaling failed to exercise the CfC nonlinearity.
     */
    printf("\n  Hidden state diagnostics (H-Std per channel, post-enrollment):\n");
    printf("    %-10s  %10s  %10s  %s\n", "Channel", "H-Std", "H-Range", "Status");
    for (int ch = 0; ch < SIM_CHANNELS; ch++) {
        /* Compute per-dimension std, then average across dimensions */
        float h_std_avg = 0;
        float h_range = 0;
        for (int d = 0; d < HIDDEN_DIM; d++) {
            double mean_d = h_sum[ch][d] / h_n[ch];
            double var_d = h_sum2[ch][d] / h_n[ch] - mean_d * mean_d;
            float std_d = sqrtf((float)(var_d > 0 ? var_d : 0));
            h_std_avg += std_d;
            if (std_d > h_range) h_range = std_d;
        }
        h_std_avg /= HIDDEN_DIM;

        const char *status;
        if (h_std_avg > 0.01f)
            status = "OK (exercising nonlinearity)";
        else if (h_std_avg > 0.001f)
            status = "MARGINAL (weak dynamics)";
        else
            status = "DEGENERATE (trapped)";

        printf("    %-10s  %10.6f  %10.6f  %s\n",
               channel_names[ch], h_std_avg, h_range, status);
    }

    /* Memory footprint (now includes calibration struct) */
    size_t per_ch = sizeof(float) * HIDDEN_DIM          /* h_state */
                  + sizeof(TelemetryDiscriminant)        /* disc */
                  + sizeof(ChannelCalibration)            /* calibration */
                  + sizeof(RunningStats) * INPUT_DIM;    /* stats */
    size_t shared = sizeof(W_gate) + sizeof(b_gate)
                  + sizeof(W_cand) + sizeof(b_cand) + sizeof(tau);
    size_t total = shared + per_ch * SIM_CHANNELS;

    printf("\n  Memory footprint (execution phase — enrollment buffers freed):\n");
    printf("    Shared weights:     %4zu bytes\n", shared);
    printf("    Per-channel state:  %4zu bytes (incl %zu cal)\n",
           per_ch, sizeof(ChannelCalibration));
    printf("    Total (%d channels): %4zu bytes (%.1f%% of 64KB L1)\n",
           SIM_CHANNELS, total, 100.0f * total / 65536.0f);

    /* ── Phase 2: Normal operation ── */
    printf("\n");
    printf("PHASE 2: NORMAL OPERATION (%.0f orbit = %.0f samples)\n",
           SIM_TEST_ORBITS,
           SIM_TEST_ORBITS * SIM_ORBITAL_PERIOD / SIM_DT);

    float test_start = cal_duration + enroll_duration;
    float test_duration = SIM_TEST_ORBITS * SIM_ORBITAL_PERIOD;
    int test_steps = (int)(test_duration / SIM_DT);

    /* Track scores across normal operation */
    float normal_min[SIM_CHANNELS], normal_max[SIM_CHANNELS];
    float normal_sum[SIM_CHANNELS];
    int normal_3sig_flags[SIM_CHANNELS];  /* 3-sigma false positives */
    for (int ch = 0; ch < SIM_CHANNELS; ch++) {
        normal_min[ch] = 1.0f;
        normal_max[ch] = 0.0f;
        normal_sum[ch] = 0.0f;
        normal_3sig_flags[ch] = 0;
    }
    int normal_scored = 0;

    for (int step = 0; step < test_steps; step++) {
        sim.time = test_start + step * SIM_DT;
        for (int ch = 0; ch < SIM_CHANNELS; ch++) {
            float value = sim_generate(&sim, ch);
            channel_step(&channels[ch], value, SIM_DT);
        }

        /* Score after warmup */
        if (step > ISS_WARMUP) {
            for (int ch = 0; ch < SIM_CHANNELS; ch++) {
                float s = channel_score(&channels[ch]);
                channels[ch].last_score = s;
                if (s < normal_min[ch]) normal_min[ch] = s;
                if (s > normal_max[ch]) normal_max[ch] = s;
                normal_sum[ch] += s;
                if (channels[ch].last_3sigma > 0.5f) normal_3sig_flags[ch]++;
            }
            normal_scored++;
        }
    }

    printf("\n  Normal operation scores (higher = more normal):\n");
    printf("  %-9s  %-8s  %-22s  %-8s  %s\n",
           "Channel", "Score", "Bar", "Status", "3sig FP");
    printf("  %-9s  %-8s  %-22s  %-8s  %s\n",
           "---------", "--------", "----------------------", "--------", "--------");
    for (int ch = 0; ch < SIM_CHANNELS; ch++) {
        float avg = normal_sum[ch] / normal_scored;
        printf("  %-9s  %.3f    ", channel_names[ch], avg);
        print_score_bar(avg, 20);
        printf("  NORMAL  %d/%d",
               normal_3sig_flags[ch], normal_scored);
        printf("\n");
    }

    CrossChannelResult normal_xc = cross_channel_score(channels, SIM_CHANNELS);
    printf("\n  Cross-channel: aggregate=%.3f, min=%.3f (%s), anomalous=%d/%d\n",
           normal_xc.aggregate_score, normal_xc.min_score,
           channel_names[normal_xc.min_channel],
           normal_xc.n_anomalous, SIM_CHANNELS);

    /* ── Phase 3: Anomaly injection ── */
    printf("\n");
    printf("PHASE 3: ANOMALY INJECTION (%.0f orbit, 3 anomaly types)\n",
           SIM_ANOMALY_ORBITS);

    /* Test each anomaly type on different channels */
    typedef struct {
        const char *label;
        int channel;
        int type;   /* 0=step, 1=ramp, 2=oscillation */
    } AnomalyTest;

    AnomalyTest tests[] = {
        { "Step change on CMG1 vibration",       0, 0 },
        { "Ramp (bearing wear) on CMG3 vibration", 2, 1 },
        { "Oscillation shift on Coolant Loop A", 4, 2 },
        { "Slow leak on Cabin pressure",         6, 1 },
        { "Step drop on O2 partial pressure",    7, 0 },
    };
    int n_tests = (int)(sizeof(tests) / sizeof(tests[0]));

    for (int ti = 0; ti < n_tests; ti++) {
        printf("\n  TEST %d: %s\n", ti + 1, tests[ti].label);

        /* Reset channels for clean test */
        for (int ch = 0; ch < SIM_CHANNELS; ch++) {
            memset(channels[ch].h_state, 0, sizeof(channels[ch].h_state));
            channels[ch].sample_count = 0;
            channels[ch].last_score = 0.5f;
        }

        /* Configure anomaly */
        float anomaly_start_time = test_start + test_duration;
        float anomaly_duration = SIM_ANOMALY_ORBITS * SIM_ORBITAL_PERIOD;
        int anomaly_steps = (int)(anomaly_duration / SIM_DT);

        /* Run half the test normal, then inject anomaly */
        int inject_at = anomaly_steps / 2;

        sim.anomaly_active = 0;
        sim.anomaly_channel = tests[ti].channel;
        sim.anomaly_type = tests[ti].type;

        /* Track pre/post anomaly scores for the target channel */
        float pre_scores[256], post_scores[256];
        int n_pre = 0, n_post = 0;

        /* 3-sigma tracking (honest baseline comparison) */
        int sigma3_detected = 0;
        int sigma3_detection_step = -1;
        int sigma3_post_flags = 0;  /* how many post-injection samples flagged */
        int sigma3_pre_flags = 0;   /* false positives before injection */
        int sigma3_pre_total = 0;
        int sigma3_post_total = 0;

        /* Track all channels' scores at end for cross-channel display */
        int detected = 0;
        int detection_step = -1;

        for (int step = 0; step < anomaly_steps; step++) {
            sim.time = anomaly_start_time + step * SIM_DT;

            if (step == inject_at) {
                sim.anomaly_active = 1;
                sim.anomaly_start = sim.time;
            }

            for (int ch = 0; ch < SIM_CHANNELS; ch++) {
                float value = sim_generate(&sim, ch);
                channel_step(&channels[ch], value, SIM_DT);
            }

            if (step > ISS_WARMUP) {
                for (int ch = 0; ch < SIM_CHANNELS; ch++) {
                    channels[ch].last_score = channel_score(&channels[ch]);
                }

                float target_score = channels[tests[ti].channel].last_score;
                float target_3sig = channels[tests[ti].channel].last_3sigma;

                if (step < inject_at && n_pre < 256) {
                    pre_scores[n_pre++] = target_score;
                    sigma3_pre_total++;
                    if (target_3sig > 0.5f) sigma3_pre_flags++;
                }
                if (step >= inject_at && n_post < 256) {
                    post_scores[n_post++] = target_score;
                    sigma3_post_total++;
                    if (target_3sig > 0.5f) {
                        sigma3_post_flags++;
                        if (!sigma3_detected) {
                            sigma3_detected = 1;
                            sigma3_detection_step = step - inject_at;
                        }
                    }
                }

                /* Check for CfC detection */
                if (step >= inject_at && !detected
                    && target_score < ISS_ANOMALY_THRESH) {
                    detected = 1;
                    detection_step = step - inject_at;
                }
            }
        }

        sim.anomaly_active = 0;

        /* Compute pre/post averages */
        float pre_avg = 0, post_avg = 0;
        for (int i = 0; i < n_pre; i++) pre_avg += pre_scores[i];
        pre_avg /= (n_pre > 0 ? n_pre : 1);
        for (int i = 0; i < n_post; i++) post_avg += post_scores[i];
        post_avg /= (n_post > 0 ? n_post : 1);

        printf("    Target: %-10s  Pre-anomaly: %.3f  Post-anomaly: %.3f  Drop: %+.3f\n",
               channel_names[tests[ti].channel], pre_avg, post_avg, post_avg - pre_avg);

        /* CfC result */
        if (detected) {
            printf("    CfC:    DETECTED at step %d (%.0f sec after injection)\n",
                   detection_step, detection_step * SIM_DT);
        } else {
            printf("    CfC:    NOT DETECTED (score stayed above threshold %.2f)\n",
                   ISS_ANOMALY_THRESH);
        }

        /* 3-sigma result (honest comparison) */
        if (sigma3_detected) {
            printf("    3-sigma: DETECTED at step %d (%.0f sec after injection), "
                   "%d/%d post-injection flagged",
                   sigma3_detection_step, sigma3_detection_step * SIM_DT,
                   sigma3_post_flags, sigma3_post_total);
        } else {
            printf("    3-sigma: NOT DETECTED (%d/%d post-injection flagged)",
                   sigma3_post_flags, sigma3_post_total);
        }
        if (sigma3_pre_flags > 0) {
            printf(" [%d/%d false positives pre-injection]",
                   sigma3_pre_flags, sigma3_pre_total);
        }
        printf("\n");

        /* Verdict: who wins? */
        if (detected && sigma3_detected) {
            if (detection_step < sigma3_detection_step)
                printf("    VERDICT: CfC faster by %d steps (%.0f sec)\n",
                       sigma3_detection_step - detection_step,
                       (sigma3_detection_step - detection_step) * SIM_DT);
            else if (sigma3_detection_step < detection_step)
                printf("    VERDICT: 3-sigma faster by %d steps (%.0f sec)\n",
                       detection_step - sigma3_detection_step,
                       (detection_step - sigma3_detection_step) * SIM_DT);
            else
                printf("    VERDICT: TIE (same detection step)\n");
        } else if (detected && !sigma3_detected) {
            printf("    VERDICT: CfC WINS (3-sigma missed it)\n");
        } else if (!detected && sigma3_detected) {
            printf("    VERDICT: 3-sigma WINS (CfC missed it)\n");
        } else {
            printf("    VERDICT: BOTH MISSED\n");
        }

        /* Cross-channel view */
        CrossChannelResult xc = cross_channel_score(channels, SIM_CHANNELS);
        if (verbose) {
            printf("    Cross-channel at end of test:\n");
            for (int ch = 0; ch < SIM_CHANNELS; ch++) {
                printf("      %-10s CfC=%.3f 3sig=%s ",
                       channel_names[ch],
                       channels[ch].last_score,
                       channels[ch].last_3sigma > 0.5f ? "FLAG" : " ok ");
                print_score_bar(channels[ch].last_score, 15);
                if (channels[ch].last_score < ISS_ANOMALY_THRESH)
                    printf(" ANOMALY");
                printf("\n");
            }
        }
        if (xc.compound_anomaly) {
            printf("    COMPOUND ANOMALY: %d/%d channels flagged\n",
                   xc.n_anomalous, SIM_CHANNELS);
        }
    }

    /* ── Phase 4: Execution benchmarks ── */
    printf("\n");
    printf("PHASE 4: EXECUTION BENCHMARKS\n");
    {
        int M = 10000;

        /* Benchmark CfC step */
        struct timeval t0, t1;
        gettimeofday(&t0, NULL);
        for (int k = 0; k < M; k++) {
            for (int ch = 0; ch < SIM_CHANNELS; ch++) {
                float value = 0.001f * (k % 100);
                channel_step(&channels[ch], value, SIM_DT);
            }
        }
        gettimeofday(&t1, NULL);
        double us_step = (double)(t1.tv_sec - t0.tv_sec) * 1e6
                       + (double)(t1.tv_usec - t0.tv_usec);

        /* Benchmark scoring */
        gettimeofday(&t0, NULL);
        volatile float dummy = 0;
        for (int k = 0; k < M; k++) {
            for (int ch = 0; ch < SIM_CHANNELS; ch++) {
                dummy += channel_score(&channels[ch]);
            }
        }
        gettimeofday(&t1, NULL);
        double us_score = (double)(t1.tv_sec - t0.tv_sec) * 1e6
                        + (double)(t1.tv_usec - t0.tv_usec);
        (void)dummy;

        /* Benchmark cross-channel */
        gettimeofday(&t0, NULL);
        for (int k = 0; k < M; k++) {
            CrossChannelResult xc = cross_channel_score(channels, SIM_CHANNELS);
            (void)xc;
        }
        gettimeofday(&t1, NULL);
        double us_cross = (double)(t1.tv_sec - t0.tv_sec) * 1e6
                        + (double)(t1.tv_usec - t0.tv_usec);

        printf("  %d iterations x %d channels:\n", M, SIM_CHANNELS);
        printf("    CfC step:        %.0f us total, %.0f ns/channel/step\n",
               us_step, (us_step / M / SIM_CHANNELS) * 1000.0);
        printf("    Scoring:         %.0f us total, %.0f ns/channel/score\n",
               us_score, (us_score / M / SIM_CHANNELS) * 1000.0);
        printf("    Cross-channel:   %.0f us total, %.0f ns/evaluation\n",
               us_cross, (us_cross / M) * 1000.0);
        printf("    Total per sample: %.0f ns (%d channels: step + score + cross)\n",
               ((us_step + us_score + us_cross) / M) * 1000.0, SIM_CHANNELS);
    }

    /* ── Summary ── */
    printf("\n=================================================================\n");
    printf("  Summary:\n");
    printf("    Pipeline: telemetry -> %d x CfC -> hybrid(mean+PCA) -> cross-channel\n",
           SIM_CHANNELS);
    printf("    CfC weights (shared):  %zu bytes\n", shared);
    printf("    Per-channel state:     %zu bytes\n", per_ch);
    printf("    Total footprint:       %zu bytes (%.1f%% of 64KB L1)\n",
           total, 100.0f * total / 65536.0f);
    printf("    Discriminant/channel:  %zu bytes\n", sizeof(TelemetryDiscriminant));
    printf("    Scoring: %.1f*mean + %.1f*PCA (sigmoid z-score)\n",
           ISS_MEAN_WEIGHT, ISS_PCA_WEIGHT);
    printf("    Anomaly threshold:     %.2f\n", ISS_ANOMALY_THRESH);
    printf("    v2 fix: frozen pre-scaling from calibration phase\n");
    printf("    Chips: cfc_cell, norm (drift), activation, fft (future)\n");
    printf("=================================================================\n");
}

/* ═══════════════════════════════════════════════════════════════════════════
 * Stdin Streaming Mode
 *
 * Reads lines: channel_id,timestamp,value
 * Pipes from scripts/iss_websocket.py for live ISS data.
 *
 * Protocol (v2 — three-phase):
 *   CALIBRATE,<n_samples> — start calibration for n samples (Phase 0)
 *   ENROLL,<n_samples>    — start enrollment for n samples (Phase 1)
 *   FINALIZE              — end enrollment, learn discriminants
 *   <int>,<float>,<float> — channel_id, timestamp, value
 *
 * Flow: send CALIBRATE,N → stream N*n_channels data lines →
 *       send ENROLL,M → stream M*n_channels data lines →
 *       send FINALIZE → stream data → get scores on stdout
 *
 * Output (after FINALIZE):
 *   timestamp,ch0_score,ch1_score,...,ch0_3sig,...,aggregate,n_anomalous,compound
 * ═══════════════════════════════════════════════════════════════════════════ */

static void run_stdin(void) {
    TelemetryChannel channels[MAX_CHANNELS];
    int n_channels = 0;
    int calibrating = 0;
    int cal_target = 0;
    int cal_count = 0;
    int enrolling = 0;
    int enroll_target = 0;
    int enroll_count = 0;
    int finalized = 0;
    float last_timestamps[MAX_CHANNELS];
    memset(last_timestamps, 0, sizeof(last_timestamps));

    /* Calibration accumulators (double precision for stability) */
    double cal_sum[MAX_CHANNELS], cal_sum2[MAX_CHANNELS];
    double cal_dt_sum[MAX_CHANNELS];
    int cal_n[MAX_CHANNELS], cal_dt_n[MAX_CHANNELS];
    memset(cal_sum, 0, sizeof(cal_sum));
    memset(cal_sum2, 0, sizeof(cal_sum2));
    memset(cal_dt_sum, 0, sizeof(cal_dt_sum));
    memset(cal_n, 0, sizeof(cal_n));
    memset(cal_dt_n, 0, sizeof(cal_dt_n));

    char line[256];

    fprintf(stderr, "ISS Telemetry Processor v2 — stdin mode\n");
    fprintf(stderr, "Protocol: CALIBRATE,N -> data -> ENROLL,M -> data -> FINALIZE -> data\n");
    fprintf(stderr, "Waiting for CALIBRATE,<n> or ENROLL,<n>...\n");

    while (fgets(line, sizeof(line), stdin)) {
        /* Strip newline */
        int len = (int)strlen(line);
        while (len > 0 && (line[len-1] == '\n' || line[len-1] == '\r'))
            line[--len] = '\0';

        if (len == 0) continue;

        /* Command: CALIBRATE,N (v2 Phase 0) */
        if (strncmp(line, "CALIBRATE,", 10) == 0) {
            cal_target = atoi(line + 10);
            if (cal_target < 20) cal_target = 20;
            calibrating = 1;
            cal_count = 0;
            /* Reset accumulators */
            memset(cal_sum, 0, sizeof(cal_sum));
            memset(cal_sum2, 0, sizeof(cal_sum2));
            memset(cal_dt_sum, 0, sizeof(cal_dt_sum));
            memset(cal_n, 0, sizeof(cal_n));
            memset(cal_dt_n, 0, sizeof(cal_dt_n));
            fprintf(stderr, "Calibration started (target: %d samples/channel)\n",
                    cal_target);
            continue;
        }

        /* Command: ENROLL,N (Phase 1 — requires calibration first for v2) */
        if (strncmp(line, "ENROLL,", 7) == 0) {
            /* If we were calibrating, freeze calibration now */
            if (calibrating) {
                for (int ch = 0; ch < n_channels; ch++) {
                    if (cal_n[ch] > 0) {
                        ChannelCalibration *cal = &channels[ch].cal;
                        cal->input_mean = (float)(cal_sum[ch] / cal_n[ch]);
                        double var = cal_sum2[ch] / cal_n[ch]
                                   - (cal_sum[ch] / cal_n[ch])
                                     * (cal_sum[ch] / cal_n[ch]);
                        cal->input_std = sqrtf((float)(var > 0 ? var : 1e-8));
                        cal->dt_mean = (cal_dt_n[ch] > 0)
                                     ? (float)(cal_dt_sum[ch] / cal_dt_n[ch])
                                     : 1.0f;
                        cal->sigma3_hi = cal->input_mean + 3.0f * cal->input_std;
                        cal->sigma3_lo = cal->input_mean - 3.0f * cal->input_std;
                        cal->n_samples = cal_n[ch];
                        cal->calibrated = 1;
                        fprintf(stderr, "  ch%d calibrated: mean=%.4f std=%.4f dt=%.2f\n",
                                ch, cal->input_mean, cal->input_std, cal->dt_mean);
                    }
                }
                calibrating = 0;
                /* Re-init channel CfC state (keep calibration) */
                for (int ch = 0; ch < n_channels; ch++) {
                    ChannelCalibration saved = channels[ch].cal;
                    char saved_name[32];
                    strncpy(saved_name, channels[ch].name, sizeof(saved_name));
                    channel_init(&channels[ch], saved_name);
                    channels[ch].cal = saved;
                }
                memset(last_timestamps, 0, sizeof(last_timestamps));
            }

            enroll_target = atoi(line + 7);
            if (enroll_target < 50) enroll_target = 50;
            enrolling = 1;
            enroll_count = 0;
            finalized = 0;
            fprintf(stderr, "Enrollment started (target: %d samples, "
                    "pre-scaling %s)\n",
                    enroll_target,
                    channels[0].cal.calibrated ? "ON" : "OFF");
            continue;
        }

        /* Command: FINALIZE */
        if (strcmp(line, "FINALIZE") == 0) {
            for (int ch = 0; ch < n_channels; ch++)
                channel_finalize_enrollment(&channels[ch]);
            finalized = 1;
            enrolling = 0;
            fprintf(stderr, "Enrollment finalized. %d channels enrolled.\n",
                    n_channels);
            /* Print header */
            printf("timestamp");
            for (int ch = 0; ch < n_channels; ch++)
                printf(",%s_cfc,%s_3sig", channels[ch].name, channels[ch].name);
            printf(",aggregate,n_anomalous,compound\n");
            fflush(stdout);
            continue;
        }

        /* Data: channel_id,timestamp,value */
        int ch_id;
        float timestamp, value;
        if (sscanf(line, "%d,%f,%f", &ch_id, &timestamp, &value) != 3) {
            fprintf(stderr, "Parse error: %s\n", line);
            continue;
        }

        if (ch_id < 0 || ch_id >= MAX_CHANNELS) {
            fprintf(stderr, "Channel %d out of range\n", ch_id);
            continue;
        }

        /* Auto-initialize new channels */
        if (ch_id >= n_channels) {
            for (int i = n_channels; i <= ch_id; i++) {
                char name[32];
                snprintf(name, sizeof(name), "ch%d", i);
                channel_init(&channels[i], name);
            }
            n_channels = ch_id + 1;
        }

        /* Compute dt */
        float dt = 1.0f;  /* default for first sample */
        if (last_timestamps[ch_id] > 0)
            dt = timestamp - last_timestamps[ch_id];
        if (dt <= 0) dt = 1.0f;
        if (dt > 10000.0f) dt = 10000.0f;
        last_timestamps[ch_id] = timestamp;

        /* During calibration: accumulate raw statistics, don't step CfC */
        if (calibrating) {
            cal_sum[ch_id] += (double)value;
            cal_sum2[ch_id] += (double)value * (double)value;
            cal_n[ch_id]++;
            if (dt < 9999.0f && last_timestamps[ch_id] > 0) {
                cal_dt_sum[ch_id] += (double)dt;
                cal_dt_n[ch_id]++;
            }
            cal_count++;
            if (cal_count % 100 == 0)
                fprintf(stderr, "  Calibrating: %d samples collected\n",
                        cal_count);
            continue;
        }

        /* Step the channel (pre-scaling applied inside if calibrated) */
        channel_step(&channels[ch_id], value, dt);

        if (enrolling) {
            enroll_count++;
            if (enroll_count % 100 == 0)
                fprintf(stderr, "  Enrolled %d/%d samples\n",
                        enroll_count, enroll_target * n_channels);
        }

        /* Score and output (only after finalization) */
        if (finalized) {
            channels[ch_id].last_score = channel_score(&channels[ch_id]);

            /* Output CfC score + 3-sigma flag for every incoming sample */
            printf("%.1f", timestamp);
            for (int ch = 0; ch < n_channels; ch++)
                printf(",%.4f,%d", channels[ch].last_score,
                       channels[ch].last_3sigma > 0.5f ? 1 : 0);

            CrossChannelResult xc = cross_channel_score(channels, n_channels);
            printf(",%.4f,%d,%d\n",
                   xc.aggregate_score, xc.n_anomalous,
                   xc.compound_anomaly);
            fflush(stdout);
        }
    }

    fprintf(stderr, "EOF on stdin. Processed %d channels.\n", n_channels);
}

/* ═══════════════════════════════════════════════════════════════════════════ */

int main(int argc, char **argv) {
    int sim_mode = 0;
    int stdin_mode = 0;
    int verbose = 0;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--sim") == 0 || strcmp(argv[i], "-s") == 0)
            sim_mode = 1;
        if (strcmp(argv[i], "--stdin") == 0)
            stdin_mode = 1;
        if (strcmp(argv[i], "--verbose") == 0 || strcmp(argv[i], "-v") == 0)
            verbose = 1;
        if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0) {
            printf("Usage: iss_telemetry [--sim] [--stdin] [--verbose]\n\n");
            printf("  --sim, -s      Simulation mode (synthetic ISS telemetry)\n");
            printf("  --stdin        Read channel_id,timestamp,value from stdin\n");
            printf("  --verbose, -v  Show per-channel detail in simulation\n\n");
            printf("  Multi-channel CfC anomaly detection for ISS telemetry.\n");
            printf("  Enrollment learns normal patterns. Execution scores deviation.\n");
            printf("  Cross-channel correlation detects compound anomalies.\n\n");
            printf("  Pipe live data: python scripts/iss_websocket.py | ./iss_telemetry --stdin\n");
            return 0;
        }
    }

    if (stdin_mode) {
        run_stdin();
    } else if (sim_mode) {
        run_simulation(verbose);
    } else {
        printf("Usage: iss_telemetry [--sim] [--stdin] [--verbose]\n");
        printf("  --sim for synthetic ISS telemetry demo\n");
        printf("  --stdin for live data from scripts/iss_websocket.py\n");
        return 1;
    }

    return 0;
}
