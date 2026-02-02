/*
 * Seismic Anomaly Detection — Yinsen Chip Stack
 *
 * 3-channel CfC processor for real-time seismic waveform anomaly detection.
 * Designed to answer the open tau question: does CfC with properly-tuned
 * time constants outperform constant-tau CfC and STA/LTA?
 *
 * Seismic signals have genuine multi-timescale structure:
 *   P-wave:    0.01–0.1s  (body wave, compressional, arrives first)
 *   S-wave:    0.1–1s     (body wave, shear, arrives second)
 *   Surface:   1–30s      (Rayleigh/Love, arrives last, largest amplitude)
 *   Noise:     0.1–0.3 Hz (ocean microseism, always present)
 *
 * This is the strongest possible test for tau. At 100 Hz with 3 orders
 * of magnitude in temporal structure, properly-tuned tau should differentiate
 * P from surface waves. If it doesn't, tau is decorative.
 *
 * Pipeline:
 *   Phase 0: Calibrate — stream quiet background, freeze pre-scaling
 *   Phase 1: Enroll — stream background through CfC, learn discriminant
 *   Phase 2: Normal test — score against discriminant (should stay high)
 *   Phase 3: Detection test — inject synthetic earthquakes, compare:
 *            - CfC with seismic tau (0.01–30s)
 *            - CfC with ISS tau (5–600s)  [WRONG for seismic]
 *            - CfC with constant tau (1.0s)
 *            - STA/LTA (seismology standard baseline)
 *
 * Two modes:
 *   --sim    Synthetic seismic waveforms with injected earthquakes
 *   --stdin  Reads (channel_id, timestamp, value) from stdin. Pipe from
 *            scripts/seismic_seedlink.py for live GFZ data.
 *
 * Channels:
 *   0: HHZ — vertical component (P-wave dominant)
 *   1: HHN — north-south horizontal
 *   2: HHE — east-west horizontal
 *
 * Key numbers:
 *   Per-channel state:  ~352 bytes (hidden + discriminant + calibration)
 *   Shared weights:     736 bytes
 *   3-channel total:    ~1,792 bytes (2.7% of 64KB L1)
 *   Sample rate:        100 Hz (dt=0.01s)
 *
 * FALSIFICATION RECORD:
 *   ISS probe 1 test 4: tau ablation FAILED under constant dt=10s.
 *   ISS probe 3: tau ablation STILL FAILED under constant dt=10s.
 *   Prediction: variable dt OR matched-timescale tau should differentiate.
 *   This detector tests the second hypothesis: matched-timescale tau at
 *   fixed dt=0.01s. If tau still doesn't matter here, the story is dead.
 *
 * Compile:
 *   cc -O2 -I include -I include/chips examples/seismic_detector.c -lm -o examples/seismic_detector
 *
 * Usage:
 *   ./examples/seismic_detector --sim               # synthetic earthquake demo
 *   ./examples/seismic_detector --sim --verbose      # show per-sample detail
 *   python scripts/seismic_seedlink.py | ./examples/seismic_detector --stdin
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

/* ═══════════════════════════════════════════════════════════════════════════
 * Network Dimensions
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
#define SIM_CHANNELS     3    /* HHZ, HHN, HHE */

/* Discriminant parameters */
#define SEIS_N_PCS            5
#define SEIS_WARMUP           200    /* 2s at 100 Hz */
#define SEIS_MAX_SAMPLES      2000   /* 20s of enrollment */
#define SEIS_POWER_ITERS      20
#define SEIS_ENROLL_SAMPLES   1500   /* 15s of enrollment data */

/* Scoring */
#define SEIS_MEAN_WEIGHT      0.3f
#define SEIS_PCA_WEIGHT       0.7f
#define SEIS_ANOMALY_THRESH   0.35f

/* ═══════════════════════════════════════════════════════════════════════════
 * Simulation Parameters
 *
 * 100 Hz seismic data. Background = ocean microseism (0.1–0.3 Hz).
 * Earthquakes injected with P-wave, S-wave, surface wave arrivals.
 * ═══════════════════════════════════════════════════════════════════════════ */
#define SIM_DT               0.01f    /* 100 Hz */
#define SIM_SR               100.0f   /* samples/sec */

/* STA/LTA — the seismology baseline (Earle & Shearer, 1994) */
#define STA_LEN_SEC          1.0f     /* short-term average: 1 second */
#define LTA_LEN_SEC          30.0f    /* long-term average: 30 seconds */
#define STA_LTA_TRIGGER      3.0f     /* standard trigger ratio */
#define STA_LEN              100    /* STA_LEN_SEC * SIM_SR */
#define LTA_LEN              3000   /* LTA_LEN_SEC * SIM_SR */

/* ═══════════════════════════════════════════════════════════════════════════
 * Tau Configurations — THE EXPERIMENT
 *
 * Three tau arrays tested on identical data. If properly-matched tau
 * outperforms the others, the CfC time constant story is validated.
 * If not, tau is decorative and should be documented as such.
 * ═══════════════════════════════════════════════════════════════════════════ */

/* Config A: SEISMIC — matched to seismic timescales
 *   Neurons 0-1: P-wave band (0.01–0.1s)
 *   Neurons 2-3: S-wave band (0.1–1s)
 *   Neurons 4-5: Surface wave band (1–10s)
 *   Neurons 6-7: Long-period (10–30s)
 */
static const float tau_seismic[HIDDEN_DIM] = {
    0.01f, 0.05f, 0.2f, 0.5f, 2.0f, 5.0f, 15.0f, 30.0f
};

/* Config B: ISS — wrong timescale (5–600s). Control for mismatched tau. */
static const float tau_iss[HIDDEN_DIM] = {
    5.0f, 15.0f, 45.0f, 120.0f, 10.0f, 30.0f, 90.0f, 600.0f
};

/* Config C: CONSTANT — single tau=1.0s. Null hypothesis: tau doesn't matter. */
static const float tau_constant[HIDDEN_DIM] = {
    1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f
};

/* Active tau for main pipeline (seismic by default) */
static const float *active_tau = tau_seismic;

/* ═══════════════════════════════════════════════════════════════════════════
 * Shared CfC Weights (same structure as ISS, reused)
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

/* ═══════════════════════════════════════════════════════════════════════════
 * Discriminant — same structure as ISS/keystroke, per channel
 * ═══════════════════════════════════════════════════════════════════════════ */

typedef struct {
    float mean[HIDDEN_DIM];
    float dim_std[HIDDEN_DIM];
    float pcs[SEIS_N_PCS][HIDDEN_DIM];
    float pc_mean[SEIS_N_PCS];
    float pc_std[SEIS_N_PCS];
    int valid;
} SeisDiscriminant;

/* ═══════════════════════════════════════════════════════════════════════════
 * Per-Channel Calibration (v2 — frozen pre-scaling)
 * ═══════════════════════════════════════════════════════════════════════════ */

typedef struct {
    float input_mean;
    float input_std;
    float dt_mean;
    int calibrated;
    int n_samples;
} SeisCalibration;

/* ═══════════════════════════════════════════════════════════════════════════
 * STA/LTA — Classic seismology event detector
 *
 * The honest baseline. STA/LTA has been the industry standard since ~1982.
 * If CfC can't beat this, it adds no value for seismic detection.
 *
 * Algorithm: running STA/LTA on squared amplitude (energy).
 * Trigger when STA/LTA > threshold.
 * ═══════════════════════════════════════════════════════════════════════════ */

typedef struct {
    float sta_buf[STA_LEN];     /* circular buffer for STA */
    float lta_buf[LTA_LEN];     /* circular buffer for LTA */
    int sta_idx;
    int lta_idx;
    float sta_sum;               /* running sum of energy in STA window */
    float lta_sum;               /* running sum of energy in LTA window */
    int n_filled;                /* samples seen (for warmup) */
    float last_ratio;            /* most recent STA/LTA ratio */
    int triggered;               /* currently triggered? */
} StaLta;

static void stalta_init(StaLta *s) {
    memset(s, 0, sizeof(*s));
}

static float stalta_update(StaLta *s, float value) {
    float energy = value * value;

    /* Update STA circular buffer */
    s->sta_sum -= s->sta_buf[s->sta_idx];
    s->sta_buf[s->sta_idx] = energy;
    s->sta_sum += energy;
    s->sta_idx = (s->sta_idx + 1) % STA_LEN;

    /* Update LTA circular buffer */
    s->lta_sum -= s->lta_buf[s->lta_idx];
    s->lta_buf[s->lta_idx] = energy;
    s->lta_sum += energy;
    s->lta_idx = (s->lta_idx + 1) % LTA_LEN;

    s->n_filled++;

    /* Need at least LTA_LEN samples before computing ratio */
    if (s->n_filled < LTA_LEN) {
        s->last_ratio = 1.0f;
        return 1.0f;
    }

    float sta = s->sta_sum / STA_LEN;
    float lta = s->lta_sum / LTA_LEN;

    s->last_ratio = (lta > 1e-20f) ? (sta / lta) : 1.0f;
    s->triggered = (s->last_ratio > STA_LTA_TRIGGER);

    return s->last_ratio;
}

/* ═══════════════════════════════════════════════════════════════════════════
 * Per-Channel State
 * ═══════════════════════════════════════════════════════════════════════════ */

typedef struct {
    char name[32];
    float h_state[HIDDEN_DIM];
    SeisDiscriminant disc;
    SeisCalibration cal;
    RunningStats input_stats[INPUT_DIM];
    float last_score;
    int sample_count;
    int enrolled;

    /* Enrollment sample buffer */
    float enroll_buf[SEIS_MAX_SAMPLES][HIDDEN_DIM];
    int n_enroll;
} SeisChannel;

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
    if (norm > 1e-10f)
        for (int i = 0; i < n; i++) v[i] /= norm;
    return norm;
}

/* ═══════════════════════════════════════════════════════════════════════════
 * Core Functions
 * ═══════════════════════════════════════════════════════════════════════════ */

static void channel_init(SeisChannel *ch, const char *name) {
    memset(ch, 0, sizeof(*ch));
    strncpy(ch->name, name, sizeof(ch->name) - 1);
    for (int i = 0; i < INPUT_DIM; i++)
        RUNNING_STATS_INIT(&ch->input_stats[i]);
}

static float prescale_value(float raw, const SeisCalibration *cal) {
    if (!cal->calibrated) return raw;
    return (raw - cal->input_mean) / (cal->input_std + 1e-8f);
}

static float prescale_dt(float dt, const SeisCalibration *cal) {
    if (!cal->calibrated || cal->dt_mean < 1e-6f) return dt;
    return dt / cal->dt_mean;
}

/**
 * Step one channel's CfC — accepts a tau pointer for ablation.
 */
static void channel_step_with_tau(
    SeisChannel *ch, float value, float dt, const float *tau_arr
) {
    float scaled_val = prescale_value(value, &ch->cal);
    float scaled_dt = prescale_dt(dt, &ch->cal);
    float input[INPUT_DIM] = { scaled_val, scaled_dt };
    float h_new[HIDDEN_DIM];

    CFC_CELL_GENERIC(
        input, ch->h_state, dt,  /* real dt for decay */
        W_gate, b_gate, W_cand, b_cand,
        tau_arr, 0,
        INPUT_DIM, HIDDEN_DIM,
        h_new
    );

    memcpy(ch->h_state, h_new, HIDDEN_DIM * sizeof(float));
    ch->sample_count++;

    /* Drift monitor (not in CfC path) */
    RUNNING_STATS_UPDATE(&ch->input_stats[0], value);
    RUNNING_STATS_UPDATE(&ch->input_stats[1], dt);

    /* Collect enrollment samples after warmup */
    if (!ch->enrolled && ch->sample_count > SEIS_WARMUP
        && ch->n_enroll < SEIS_MAX_SAMPLES) {
        memcpy(ch->enroll_buf[ch->n_enroll], ch->h_state,
               HIDDEN_DIM * sizeof(float));
        ch->n_enroll++;
    }
}

/* Default step uses active tau */
static void channel_step(SeisChannel *ch, float value, float dt) {
    channel_step_with_tau(ch, value, dt, active_tau);
}

/**
 * Learn discriminant from enrollment samples.
 */
static void learn_discriminant(
    const float samples[][HIDDEN_DIM],
    int n_samples,
    SeisDiscriminant *disc
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

    /* PCs via power iteration */
    for (int pc = 0; pc < SEIS_N_PCS; pc++) {
        float v[HIDDEN_DIM];
        for (int i = 0; i < HIDDEN_DIM; i++)
            v[i] = centered[0][i] + 0.01f * (i + 1);
        vec_normalize(v, HIDDEN_DIM);

        for (int iter = 0; iter < SEIS_POWER_ITERS; iter++) {
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

        /* Deflate */
        for (int t = 0; t < n_samples; t++) {
            float p = dotf(centered[t], v, HIDDEN_DIM);
            for (int i = 0; i < HIDDEN_DIM; i++)
                centered[t][i] -= p * v[i];
        }
    }

    disc->valid = 1;
}

static void channel_finalize_enrollment(SeisChannel *ch) {
    learn_discriminant(
        (const float (*)[HIDDEN_DIM])ch->enroll_buf,
        ch->n_enroll, &ch->disc
    );
    ch->enrolled = 1;
    memset(ch->h_state, 0, sizeof(ch->h_state));
    ch->sample_count = 0;
}

/**
 * Score current hidden state against discriminant.
 */
static float channel_score(const SeisChannel *ch) {
    if (!ch->disc.valid) return 0.5f;

    float mean_dist = 0;
    float centered[HIDDEN_DIM];
    for (int i = 0; i < HIDDEN_DIM; i++) {
        centered[i] = ch->h_state[i] - ch->disc.mean[i];
        float z = centered[i] / (ch->disc.dim_std[i] + 1e-8f);
        mean_dist += z * z;
    }
    mean_dist /= HIDDEN_DIM;
    float mean_score = SIGMOID_CHIP(2.0f - mean_dist);

    float pca_dist = 0;
    for (int pc = 0; pc < SEIS_N_PCS; pc++) {
        float proj = dotf(centered, ch->disc.pcs[pc], HIDDEN_DIM);
        float z = (proj - ch->disc.pc_mean[pc]) / (ch->disc.pc_std[pc] + 1e-8f);
        pca_dist += z * z;
    }
    pca_dist /= SEIS_N_PCS;
    float pca_score = SIGMOID_CHIP(2.0f - pca_dist);

    return SEIS_MEAN_WEIGHT * mean_score + SEIS_PCA_WEIGHT * pca_score;
}

/* ═══════════════════════════════════════════════════════════════════════════
 * Simulation: Synthetic Seismic Waveforms
 *
 * Background model:
 *   - Ocean microseism: 0.15 Hz primary, 0.3 Hz secondary (dominant noise)
 *   - High-frequency site noise: white noise, low amplitude
 *   - Vertical (Z) gets most P-wave energy, horizontals (N/E) get S/surface
 *
 * Earthquake model (simplified but physically motivated):
 *   - P-wave: high frequency (1–10 Hz), small amplitude, sharp onset
 *   - S-wave: medium frequency (0.5–5 Hz), medium amplitude, ~tp*1.73
 *   - Surface wave: low frequency (0.05–0.5 Hz), large amplitude, arrives last
 *   - Amplitude decays as 1/r, but we just inject at known magnitude
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

/**
 * Generate background microseismic noise for one sample.
 * Returns amplitude in simulated "counts" (like a seismometer digitizer).
 */
static float gen_background(float t, int channel, unsigned int *seed) {
    /* Ocean microseism: primary (0.15 Hz) and secondary (0.30 Hz) peaks */
    float primary = 500.0f * sinf(2.0f * 3.14159265f * 0.15f * t);
    float secondary = 800.0f * sinf(2.0f * 3.14159265f * 0.30f * t
                                    + 1.2f * channel);

    /* Add some random phase modulation for realism */
    float phase_mod = 0.3f * sinf(2.0f * 3.14159265f * 0.02f * t + channel);
    primary *= (1.0f + phase_mod);
    secondary *= (1.0f + 0.5f * phase_mod);

    /* Site noise (white, small amplitude) */
    float site_noise = 50.0f * gaussf(seed);

    /* Vertical gets slightly different character than horizontal */
    if (channel == 0) {
        /* HHZ: more high-frequency content */
        return primary * 0.7f + secondary + site_noise * 1.5f;
    } else {
        /* HHN/HHE: more low-frequency content */
        return primary + secondary * 0.8f + site_noise;
    }
}

/**
 * Generate a synthetic earthquake waveform.
 *
 * t_since_origin: time since earthquake origin (seconds)
 * distance_km: epicentral distance (affects arrival times and amplitude)
 * magnitude: ~Richter-like (affects amplitude scaling)
 * channel: 0=Z, 1=N, 2=E
 *
 * Returns: waveform amplitude in counts. Zero before P arrival.
 */
static float gen_earthquake(
    float t_since_origin, float distance_km, float magnitude,
    int channel, unsigned int *seed
) {
    /* Travel times (simplified, homogeneous earth) */
    float vp = 6.0f;   /* P-wave velocity: ~6 km/s */
    float vs = 3.5f;    /* S-wave velocity: ~3.5 km/s */
    float vsurf = 3.0f; /* surface wave: ~3 km/s */

    float tp = distance_km / vp;
    float ts = distance_km / vs;
    float tsurf = distance_km / vsurf;

    /* Amplitude scaling: 10^magnitude, decay as 1/sqrt(distance) */
    float amp = powf(10.0f, magnitude) / sqrtf(distance_km + 1.0f);

    float result = 0;

    /* P-wave: high frequency, arrives first */
    float dt_p = t_since_origin - tp;
    if (dt_p > 0 && dt_p < 20.0f) {
        float env_p = dt_p * expf(-dt_p * 0.5f);  /* attack-decay envelope */
        float freq_p = 5.0f + 3.0f * gaussf(seed) * 0.1f;
        float p_wave = env_p * sinf(2.0f * 3.14159265f * freq_p * dt_p);
        /* P-wave is strongest on vertical */
        float p_gain = (channel == 0) ? 1.0f : 0.3f;
        result += amp * 200.0f * p_wave * p_gain;
    }

    /* S-wave: medium frequency, arrives second */
    float dt_s = t_since_origin - ts;
    if (dt_s > 0 && dt_s < 40.0f) {
        float env_s = dt_s * expf(-dt_s * 0.2f);
        float freq_s = 2.0f + 1.0f * sinf(dt_s * 0.3f);  /* frequency glide */
        float s_wave = env_s * sinf(2.0f * 3.14159265f * freq_s * dt_s);
        /* S-wave is strongest on horizontals */
        float s_gain = (channel == 0) ? 0.4f : 1.0f;
        /* Different horizontal components get different phases */
        if (channel == 2) s_wave = -s_wave;
        result += amp * 500.0f * s_wave * s_gain;
    }

    /* Surface wave: low frequency, arrives last, largest amplitude */
    float dt_surf = t_since_origin - tsurf;
    if (dt_surf > 0 && dt_surf < 120.0f) {
        float env_surf = sinf(3.14159265f * dt_surf / 120.0f);  /* broad envelope */
        if (env_surf < 0) env_surf = 0;
        float freq_surf = 0.1f + 0.05f * sinf(dt_surf * 0.01f);
        float surf_wave = env_surf * sinf(2.0f * 3.14159265f * freq_surf * dt_surf);
        /* Surface waves are strong on all components */
        float surf_gain = (channel == 0) ? 0.7f : 1.0f;
        result += amp * 2000.0f * surf_wave * surf_gain;
    }

    return result;
}

/* ═══════════════════════════════════════════════════════════════════════════
 * Print Helpers
 * ═══════════════════════════════════════════════════════════════════════════ */

static const char* seis_channel_names[SIM_CHANNELS] = {
    "HHZ", "HHN", "HHE"
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

/* ═══════════════════════════════════════════════════════════════════════════
 * Tau Ablation Engine
 *
 * Runs 3 independent CfC channel sets on identical data, each with
 * different tau configuration. Reports detection results side-by-side.
 * ═══════════════════════════════════════════════════════════════════════════ */

typedef struct {
    const char *label;
    const float *tau;
    SeisChannel channels[SIM_CHANNELS];
    int detected;
    int detection_sample;
    float detection_time;
    float post_score_avg;
    float pre_score_avg;
    float h_std_avg[SIM_CHANNELS];  /* hidden state variability */
} TauConfig;

/* ═══════════════════════════════════════════════════════════════════════════
 * Simulation Mode
 * ═══════════════════════════════════════════════════════════════════════════ */

static void run_simulation(int verbose) {
    printf("=================================================================\n");
    printf("  Seismic Anomaly Detection — Yinsen Chip Stack\n");
    printf("  TAU ABLATION EXPERIMENT\n");
    printf("=================================================================\n\n");

    printf("  3-channel CfC, seismic waveforms at 100 Hz.\n");
    printf("  Testing whether matched-timescale tau outperforms mismatched.\n\n");

    printf("  Tau configurations under test:\n");
    printf("    A (Seismic):  ");
    for (int i = 0; i < HIDDEN_DIM; i++) printf("%.2f ", tau_seismic[i]);
    printf("\n");
    printf("    B (ISS):      ");
    for (int i = 0; i < HIDDEN_DIM; i++) printf("%.0f ", tau_iss[i]);
    printf("\n");
    printf("    C (Constant): ");
    for (int i = 0; i < HIDDEN_DIM; i++) printf("%.1f ", tau_constant[i]);
    printf("\n\n");

    printf("  STA/LTA baseline: STA=%.1fs, LTA=%.1fs, trigger=%.1f\n\n",
           STA_LEN_SEC, LTA_LEN_SEC, STA_LTA_TRIGGER);

    /* ── Initialize 3 tau configs ── */
    TauConfig configs[3] = {
        { .label = "Seismic", .tau = tau_seismic },
        { .label = "ISS",     .tau = tau_iss },
        { .label = "Constant",.tau = tau_constant },
    };
    int n_configs = 3;

    for (int c = 0; c < n_configs; c++) {
        for (int ch = 0; ch < SIM_CHANNELS; ch++) {
            channel_init(&configs[c].channels[ch], seis_channel_names[ch]);
        }
        configs[c].detected = 0;
        configs[c].detection_sample = -1;
    }

    /* STA/LTA for each channel */
    StaLta stalta[SIM_CHANNELS];
    for (int ch = 0; ch < SIM_CHANNELS; ch++)
        stalta_init(&stalta[ch]);

    unsigned int seed = 42;

    /* ═══════════════════════════════════════════════════════════════════════
     * Phase 0: CALIBRATION (60s of background noise)
     * ═══════════════════════════════════════════════════════════════════════ */
    float cal_duration = 60.0f;  /* 60 seconds */
    int cal_steps = (int)(cal_duration / SIM_DT);

    printf("PHASE 0: CALIBRATION (%.0fs = %d samples at %.0f Hz)\n",
           cal_duration, cal_steps, SIM_SR);

    double cal_sum[SIM_CHANNELS] = {0}, cal_sum2[SIM_CHANNELS] = {0};
    int cal_n[SIM_CHANNELS] = {0};

    for (int step = 0; step < cal_steps; step++) {
        float t = step * SIM_DT;
        for (int ch = 0; ch < SIM_CHANNELS; ch++) {
            float value = gen_background(t, ch, &seed);
            cal_sum[ch] += (double)value;
            cal_sum2[ch] += (double)value * (double)value;
            cal_n[ch]++;
        }
    }

    /* Freeze calibration across all configs */
    for (int c = 0; c < n_configs; c++) {
        for (int ch = 0; ch < SIM_CHANNELS; ch++) {
            SeisCalibration *cal = &configs[c].channels[ch].cal;
            cal->input_mean = (float)(cal_sum[ch] / cal_n[ch]);
            double var = cal_sum2[ch] / cal_n[ch]
                       - (cal_sum[ch] / cal_n[ch]) * (cal_sum[ch] / cal_n[ch]);
            cal->input_std = sqrtf((float)(var > 0 ? var : 1e-8));
            cal->dt_mean = SIM_DT;
            cal->n_samples = cal_n[ch];
            cal->calibrated = 1;
        }
    }

    printf("  Calibration results:\n");
    printf("    %-6s  %12s  %12s\n", "Ch", "Mean", "Std");
    for (int ch = 0; ch < SIM_CHANNELS; ch++) {
        SeisCalibration *cal = &configs[0].channels[ch].cal;
        printf("    %-6s  %12.2f  %12.2f\n",
               seis_channel_names[ch], cal->input_mean, cal->input_std);
    }

    /* ═══════════════════════════════════════════════════════════════════════
     * Phase 1: ENROLLMENT (30s of background noise through CfC)
     * ═══════════════════════════════════════════════════════════════════════ */
    float enroll_duration = 30.0f;
    int enroll_steps = (int)(enroll_duration / SIM_DT);

    printf("\nPHASE 1: ENROLLMENT (%.0fs = %d samples, all tau configs)\n",
           enroll_duration, enroll_steps);

    /* Track hidden state stats during enrollment */
    double h_sum[3][SIM_CHANNELS][HIDDEN_DIM];
    double h_sum2[3][SIM_CHANNELS][HIDDEN_DIM];
    int h_n[3][SIM_CHANNELS];
    memset(h_sum, 0, sizeof(h_sum));
    memset(h_sum2, 0, sizeof(h_sum2));
    memset(h_n, 0, sizeof(h_n));

    for (int step = 0; step < enroll_steps; step++) {
        float t = cal_duration + step * SIM_DT;
        for (int ch = 0; ch < SIM_CHANNELS; ch++) {
            float value = gen_background(t, ch, &seed);

            /* Step all 3 configs on same data */
            for (int c = 0; c < n_configs; c++) {
                channel_step_with_tau(&configs[c].channels[ch], value, SIM_DT,
                                      configs[c].tau);
            }

            /* Also feed STA/LTA to warm it up */
            float prescaled = prescale_value(value, &configs[0].channels[ch].cal);
            stalta_update(&stalta[ch], prescaled);

            /* Collect hidden state stats after warmup */
            if (step > SEIS_WARMUP) {
                for (int c = 0; c < n_configs; c++) {
                    for (int d = 0; d < HIDDEN_DIM; d++) {
                        double hv = (double)configs[c].channels[ch].h_state[d];
                        h_sum[c][ch][d] += hv;
                        h_sum2[c][ch][d] += hv * hv;
                    }
                    h_n[c][ch]++;
                }
            }
        }
    }

    /* Finalize enrollment for all configs */
    for (int c = 0; c < n_configs; c++) {
        for (int ch = 0; ch < SIM_CHANNELS; ch++) {
            channel_finalize_enrollment(&configs[c].channels[ch]);
        }
    }

    /* Print hidden state diagnostics */
    printf("\n  Hidden state diagnostics (H-Std, averaged across dims):\n");
    printf("    %-6s", "Ch");
    for (int c = 0; c < n_configs; c++)
        printf("  %12s", configs[c].label);
    printf("\n");
    printf("    %-6s", "------");
    for (int c = 0; c < n_configs; c++)
        printf("  %12s", "------------");
    printf("\n");

    for (int ch = 0; ch < SIM_CHANNELS; ch++) {
        printf("    %-6s", seis_channel_names[ch]);
        for (int c = 0; c < n_configs; c++) {
            float h_std_avg = 0;
            for (int d = 0; d < HIDDEN_DIM; d++) {
                double mean_d = h_sum[c][ch][d] / h_n[c][ch];
                double var_d = h_sum2[c][ch][d] / h_n[c][ch] - mean_d * mean_d;
                h_std_avg += sqrtf((float)(var_d > 0 ? var_d : 0));
            }
            h_std_avg /= HIDDEN_DIM;
            configs[c].h_std_avg[ch] = h_std_avg;
            printf("  %12.6f", h_std_avg);
        }
        printf("\n");
    }

    /* ═══════════════════════════════════════════════════════════════════════
     * Phase 2: NORMAL TEST (20s, no earthquake)
     * ═══════════════════════════════════════════════════════════════════════ */
    float normal_duration = 20.0f;
    int normal_steps = (int)(normal_duration / SIM_DT);

    printf("\nPHASE 2: NORMAL OPERATION (%.0fs = %d samples)\n",
           normal_duration, normal_steps);

    float normal_scores[3][SIM_CHANNELS];
    int normal_scored = 0;
    int stalta_false_pos[SIM_CHANNELS] = {0};
    memset(normal_scores, 0, sizeof(normal_scores));

    for (int step = 0; step < normal_steps; step++) {
        float t = cal_duration + enroll_duration + step * SIM_DT;
        for (int ch = 0; ch < SIM_CHANNELS; ch++) {
            float value = gen_background(t, ch, &seed);

            for (int c = 0; c < n_configs; c++)
                channel_step_with_tau(&configs[c].channels[ch], value, SIM_DT,
                                      configs[c].tau);

            float prescaled = prescale_value(value, &configs[0].channels[ch].cal);
            stalta_update(&stalta[ch], prescaled);

            if (stalta[ch].triggered) stalta_false_pos[ch]++;
        }

        if (step > SEIS_WARMUP) {
            for (int c = 0; c < n_configs; c++)
                for (int ch = 0; ch < SIM_CHANNELS; ch++) {
                    float s = channel_score(&configs[c].channels[ch]);
                    configs[c].channels[ch].last_score = s;
                    normal_scores[c][ch] += s;
                }
            normal_scored++;
        }
    }

    printf("\n  Normal scores (higher = more normal):\n");
    printf("    %-6s", "Ch");
    for (int c = 0; c < n_configs; c++)
        printf("  %12s", configs[c].label);
    printf("  %12s\n", "STA/LTA FP");
    printf("    %-6s", "------");
    for (int c = 0; c < n_configs; c++)
        printf("  %12s", "------------");
    printf("  %12s\n", "------------");
    for (int ch = 0; ch < SIM_CHANNELS; ch++) {
        printf("    %-6s", seis_channel_names[ch]);
        for (int c = 0; c < n_configs; c++) {
            float avg = normal_scores[c][ch] / normal_scored;
            printf("  %12.4f", avg);
        }
        printf("  %8d/%d\n", stalta_false_pos[ch], normal_steps);
    }

    /* ═══════════════════════════════════════════════════════════════════════
     * Phase 3: EARTHQUAKE DETECTION TESTS
     *
     * Inject synthetic earthquakes at known times. Compare all detectors.
     * ═══════════════════════════════════════════════════════════════════════ */
    printf("\n");
    printf("PHASE 3: EARTHQUAKE DETECTION TESTS\n");

    typedef struct {
        const char *label;
        float distance_km;
        float magnitude;
    } EarthquakeTest;

    EarthquakeTest eq_tests[] = {
        { "Regional M4.0 at 100km",   100.0f, 4.0f },
        { "Regional M3.0 at 50km",     50.0f, 3.0f },
        { "Teleseismic M6.0 at 5000km", 5000.0f, 6.0f },
        { "Local M2.0 at 20km",        20.0f, 2.0f },
        { "Weak M1.5 at 100km (limit test)", 100.0f, 1.5f },
    };
    int n_eq_tests = (int)(sizeof(eq_tests) / sizeof(eq_tests[0]));

    for (int ti = 0; ti < n_eq_tests; ti++) {
        printf("\n  TEST %d: %s\n", ti + 1, eq_tests[ti].label);

        /* Reset all configs and STA/LTA */
        for (int c = 0; c < n_configs; c++) {
            for (int ch = 0; ch < SIM_CHANNELS; ch++) {
                memset(configs[c].channels[ch].h_state, 0,
                       sizeof(configs[c].channels[ch].h_state));
                configs[c].channels[ch].sample_count = 0;
                configs[c].channels[ch].last_score = 0.5f;
            }
            configs[c].detected = 0;
            configs[c].detection_sample = -1;
            configs[c].detection_time = -1.0f;
            configs[c].pre_score_avg = 0;
            configs[c].post_score_avg = 0;
        }
        for (int ch = 0; ch < SIM_CHANNELS; ch++)
            stalta_init(&stalta[ch]);

        /* 60s total: 30s quiet + earthquake injection at t=30s */
        float test_duration = 60.0f;
        int test_steps = (int)(test_duration / SIM_DT);
        float eq_origin_time = 30.0f;
        int eq_origin_step = (int)(eq_origin_time / SIM_DT);

        /* STA/LTA tracking */
        int stalta_detected = 0;
        int stalta_detection_sample = -1;
        float stalta_detection_time = -1.0f;
        int stalta_post_triggers[SIM_CHANNELS] = {0};

        /* Score accumulators */
        float pre_sum[3] = {0}, post_sum[3] = {0};
        int n_pre = 0, n_post = 0;

        float t_base = cal_duration + enroll_duration + normal_duration
                       + ti * test_duration;

        for (int step = 0; step < test_steps; step++) {
            float t = t_base + step * SIM_DT;
            float t_local = step * SIM_DT;

            for (int ch = 0; ch < SIM_CHANNELS; ch++) {
                float value = gen_background(t, ch, &seed);

                /* Inject earthquake after origin time */
                if (t_local >= eq_origin_time) {
                    value += gen_earthquake(
                        t_local - eq_origin_time,
                        eq_tests[ti].distance_km,
                        eq_tests[ti].magnitude,
                        ch, &seed
                    );
                }

                /* Step all CfC configs */
                for (int c = 0; c < n_configs; c++)
                    channel_step_with_tau(&configs[c].channels[ch], value,
                                          SIM_DT, configs[c].tau);

                /* STA/LTA on prescaled value */
                float prescaled = prescale_value(value,
                                                  &configs[0].channels[ch].cal);
                stalta_update(&stalta[ch], prescaled);

                /* STA/LTA detection (any channel) */
                if (t_local >= eq_origin_time && stalta[ch].triggered) {
                    stalta_post_triggers[ch]++;
                    if (!stalta_detected) {
                        stalta_detected = 1;
                        stalta_detection_sample = step - eq_origin_step;
                        stalta_detection_time = stalta_detection_sample * SIM_DT;
                    }
                }
            }

            /* Score all configs (after warmup) */
            if (step > SEIS_WARMUP) {
                for (int c = 0; c < n_configs; c++) {
                    /* Use HHZ (vertical, best P-wave pickup) for detection */
                    float s = channel_score(&configs[c].channels[0]);
                    configs[c].channels[0].last_score = s;

                    if (step < eq_origin_step) {
                        pre_sum[c] += s;
                    } else {
                        post_sum[c] += s;

                        /* CfC detection */
                        if (!configs[c].detected && s < SEIS_ANOMALY_THRESH) {
                            configs[c].detected = 1;
                            configs[c].detection_sample = step - eq_origin_step;
                            configs[c].detection_time =
                                configs[c].detection_sample * SIM_DT;
                        }
                    }
                }

                if (step < eq_origin_step) n_pre++;
                else n_post++;
            }
        }

        /* Compute averages */
        for (int c = 0; c < n_configs; c++) {
            configs[c].pre_score_avg = (n_pre > 0) ? pre_sum[c] / n_pre : 0;
            configs[c].post_score_avg = (n_post > 0) ? post_sum[c] / n_post : 0;
        }

        /* Report results */
        printf("    %-12s  %8s  %8s  %8s  %s\n",
               "Detector", "Pre", "Post", "Drop", "Detection");
        printf("    %-12s  %8s  %8s  %8s  %s\n",
               "------------", "--------", "--------", "--------",
               "------------------------");

        for (int c = 0; c < n_configs; c++) {
            printf("    CfC-%-7s  %8.3f  %8.3f  %+8.3f  ",
                   configs[c].label,
                   configs[c].pre_score_avg,
                   configs[c].post_score_avg,
                   configs[c].post_score_avg - configs[c].pre_score_avg);

            if (configs[c].detected) {
                printf("YES at %.2fs (%d samples)\n",
                       configs[c].detection_time,
                       configs[c].detection_sample);
            } else {
                printf("NO\n");
            }
        }

        /* STA/LTA result */
        printf("    STA/LTA      %8s  %8s  %8s  ",
               "---", "---", "---");
        if (stalta_detected) {
            printf("YES at %.2fs (%d samples)",
                   stalta_detection_time, stalta_detection_sample);
            int total_triggers = 0;
            for (int ch = 0; ch < SIM_CHANNELS; ch++)
                total_triggers += stalta_post_triggers[ch];
            printf(" [%d flags]\n", total_triggers);
        } else {
            printf("NO\n");
        }

        /* ── Verdicts ── */
        printf("\n    VERDICTS:\n");

        /* Find fastest CfC */
        int fastest_cfc = -1;
        int fastest_cfc_sample = 999999;
        for (int c = 0; c < n_configs; c++) {
            if (configs[c].detected && configs[c].detection_sample < fastest_cfc_sample) {
                fastest_cfc = c;
                fastest_cfc_sample = configs[c].detection_sample;
            }
        }

        /* Tau ablation verdict */
        if (configs[0].detected && configs[1].detected && configs[2].detected) {
            /* All detected — compare timing */
            if (configs[0].detection_sample < configs[1].detection_sample &&
                configs[0].detection_sample < configs[2].detection_sample) {
                printf("      TAU: Seismic WINS (%.2fs vs ISS %.2fs, Const %.2fs)\n",
                       configs[0].detection_time,
                       configs[1].detection_time,
                       configs[2].detection_time);
            } else if (configs[0].detection_sample == configs[1].detection_sample &&
                       configs[0].detection_sample == configs[2].detection_sample) {
                printf("      TAU: THREE-WAY TIE (all at %.2fs) — tau doesn't matter\n",
                       configs[0].detection_time);
            } else {
                printf("      TAU: Seismic NOT fastest (Seismic=%.2fs, ISS=%.2fs, Const=%.2fs)\n",
                       configs[0].detection_time,
                       configs[1].detected ? configs[1].detection_time : -1.0f,
                       configs[2].detected ? configs[2].detection_time : -1.0f);
            }
        } else {
            /* Check who detected and who didn't */
            int n_detected = 0;
            for (int c = 0; c < n_configs; c++)
                n_detected += configs[c].detected;

            if (n_detected == 0) {
                printf("      TAU: ALL CfC MISSED — can't compare\n");
            } else {
                printf("      TAU: ");
                for (int c = 0; c < n_configs; c++) {
                    printf("%s=%s", configs[c].label,
                           configs[c].detected ? "HIT" : "MISS");
                    if (configs[c].detected)
                        printf("(%.2fs)", configs[c].detection_time);
                    if (c < n_configs - 1) printf(", ");
                }
                printf("\n");

                /* Seismic detected but others didn't = tau matters */
                if (configs[0].detected && !configs[1].detected && !configs[2].detected) {
                    printf("      ***  SEISMIC TAU UNIQUE DETECTION — TAU MATTERS  ***\n");
                }
            }
        }

        /* CfC vs STA/LTA verdict */
        if (fastest_cfc >= 0 && stalta_detected) {
            int diff = fastest_cfc_sample - stalta_detection_sample;
            if (diff < 0) {
                printf("      CfC vs STA/LTA: CfC-%s faster by %.2fs\n",
                       configs[fastest_cfc].label, -diff * SIM_DT);
            } else if (diff > 0) {
                printf("      CfC vs STA/LTA: STA/LTA faster by %.2fs\n",
                       diff * SIM_DT);
            } else {
                printf("      CfC vs STA/LTA: TIE\n");
            }
        } else if (fastest_cfc >= 0 && !stalta_detected) {
            printf("      CfC vs STA/LTA: CfC-%s WINS (STA/LTA missed)\n",
                   configs[fastest_cfc].label);
        } else if (fastest_cfc < 0 && stalta_detected) {
            printf("      CfC vs STA/LTA: STA/LTA WINS (all CfC missed)\n");
        } else {
            printf("      CfC vs STA/LTA: BOTH MISSED\n");
        }

        /* Verbose: per-sample hidden state evolution around earthquake */
        if (verbose && configs[0].detected) {
            printf("\n    HHZ score evolution around detection (Seismic tau):\n");
            printf("    (Re-running for verbose trace...)\n");
        }
    }

    /* ═══════════════════════════════════════════════════════════════════════
     * Phase 4: Execution Benchmarks
     * ═══════════════════════════════════════════════════════════════════════ */
    printf("\n");
    printf("PHASE 4: EXECUTION BENCHMARKS\n");
    {
        int M = 10000;
        struct timeval t0, t1;

        /* CfC step (seismic tau) */
        gettimeofday(&t0, NULL);
        for (int k = 0; k < M; k++) {
            for (int ch = 0; ch < SIM_CHANNELS; ch++) {
                float value = 100.0f * (k % 100);
                channel_step_with_tau(&configs[0].channels[ch], value, SIM_DT,
                                      tau_seismic);
            }
        }
        gettimeofday(&t1, NULL);
        double us_step = (double)(t1.tv_sec - t0.tv_sec) * 1e6
                       + (double)(t1.tv_usec - t0.tv_usec);

        /* Scoring */
        gettimeofday(&t0, NULL);
        volatile float dummy = 0;
        for (int k = 0; k < M; k++) {
            for (int ch = 0; ch < SIM_CHANNELS; ch++) {
                dummy += channel_score(&configs[0].channels[ch]);
            }
        }
        gettimeofday(&t1, NULL);
        double us_score = (double)(t1.tv_sec - t0.tv_sec) * 1e6
                        + (double)(t1.tv_usec - t0.tv_usec);
        (void)dummy;

        /* STA/LTA */
        gettimeofday(&t0, NULL);
        for (int k = 0; k < M; k++) {
            for (int ch = 0; ch < SIM_CHANNELS; ch++) {
                stalta_update(&stalta[ch], (float)(k % 100));
            }
        }
        gettimeofday(&t1, NULL);
        double us_stalta = (double)(t1.tv_sec - t0.tv_sec) * 1e6
                         + (double)(t1.tv_usec - t0.tv_usec);

        printf("  %d iterations x %d channels:\n", M, SIM_CHANNELS);
        printf("    CfC step:    %.0f us total, %.0f ns/channel/step\n",
               us_step, (us_step / M / SIM_CHANNELS) * 1000.0);
        printf("    CfC score:   %.0f us total, %.0f ns/channel/score\n",
               us_score, (us_score / M / SIM_CHANNELS) * 1000.0);
        printf("    STA/LTA:     %.0f us total, %.0f ns/channel/update\n",
               us_stalta, (us_stalta / M / SIM_CHANNELS) * 1000.0);

        /* Can we keep up with 100 Hz? */
        double ns_per_sample = ((us_step + us_score) / M / SIM_CHANNELS) * 1000.0;
        double budget_ns = (1.0 / SIM_SR) * 1e9;  /* 10,000,000 ns at 100 Hz */
        printf("\n    Budget at %.0f Hz: %.0f ns/sample\n", SIM_SR, budget_ns);
        printf("    Actual:          %.0f ns/sample\n", ns_per_sample);
        printf("    Headroom:        %.0fx real-time\n", budget_ns / ns_per_sample);
    }

    /* ═══════════════════════════════════════════════════════════════════════
     * Summary
     * ═══════════════════════════════════════════════════════════════════════ */
    printf("\n=================================================================\n");
    printf("  Summary:\n");

    size_t per_ch = sizeof(float) * HIDDEN_DIM
                  + sizeof(SeisDiscriminant)
                  + sizeof(SeisCalibration)
                  + sizeof(RunningStats) * INPUT_DIM;
    size_t shared = sizeof(W_gate) + sizeof(b_gate)
                  + sizeof(W_cand) + sizeof(b_cand)
                  + sizeof(tau_seismic);
    size_t total = shared + per_ch * SIM_CHANNELS;

    printf("    Pipeline: seismic -> %d x CfC -> hybrid(mean+PCA) vs STA/LTA\n",
           SIM_CHANNELS);
    printf("    Shared weights:  %zu bytes\n", shared);
    printf("    Per-channel:     %zu bytes\n", per_ch);
    printf("    Total (%d ch):    %zu bytes (%.1f%% of 64KB L1)\n",
           SIM_CHANNELS, total, 100.0f * total / 65536.0f);
    printf("    STA/LTA buffer:  %zu bytes/ch\n",
           sizeof(StaLta));

    printf("\n  TAU ABLATION RESULTS SUMMARY:\n");
    printf("    If all 3 tau configs gave identical detection → tau is decorative.\n");
    printf("    If Seismic tau consistently won → matched-timescale tau matters.\n");
    printf("    If ISS/Constant beat Seismic → wrong hypothesis.\n");
    printf("    Full results above. Document honestly.\n");
    printf("=================================================================\n");
}

/* ═══════════════════════════════════════════════════════════════════════════
 * Stdin Streaming Mode (reuses ISS protocol)
 *
 * Protocol: CALIBRATE,N -> data -> ENROLL,M -> data -> FINALIZE -> data
 * Each data line: channel_id,timestamp,value
 * ═══════════════════════════════════════════════════════════════════════════ */

static void run_stdin(void) {
    SeisChannel channels[MAX_CHANNELS];
    StaLta stalta[MAX_CHANNELS];
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

    double cal_sum[MAX_CHANNELS], cal_sum2[MAX_CHANNELS];
    double cal_dt_sum[MAX_CHANNELS];
    int cal_n[MAX_CHANNELS], cal_dt_n[MAX_CHANNELS];
    memset(cal_sum, 0, sizeof(cal_sum));
    memset(cal_sum2, 0, sizeof(cal_sum2));
    memset(cal_dt_sum, 0, sizeof(cal_dt_sum));
    memset(cal_n, 0, sizeof(cal_n));
    memset(cal_dt_n, 0, sizeof(cal_dt_n));

    for (int i = 0; i < MAX_CHANNELS; i++)
        stalta_init(&stalta[i]);

    char line[256];

    fprintf(stderr, "Seismic Detector v1 — stdin mode (seismic tau)\n");
    fprintf(stderr, "  Tau: ");
    for (int i = 0; i < HIDDEN_DIM; i++)
        fprintf(stderr, "%.2f ", tau_seismic[i]);
    fprintf(stderr, "\n");
    fprintf(stderr, "  STA/LTA: STA=%.1fs, LTA=%.1fs, trigger=%.1f\n",
            STA_LEN_SEC, LTA_LEN_SEC, STA_LTA_TRIGGER);
    fprintf(stderr, "Protocol: CALIBRATE,N -> data -> ENROLL,M -> data -> FINALIZE -> data\n");

    while (fgets(line, sizeof(line), stdin)) {
        int len = (int)strlen(line);
        while (len > 0 && (line[len-1] == '\n' || line[len-1] == '\r'))
            line[--len] = '\0';
        if (len == 0) continue;

        /* CALIBRATE,N */
        if (strncmp(line, "CALIBRATE,", 10) == 0) {
            cal_target = atoi(line + 10);
            if (cal_target < 20) cal_target = 20;
            calibrating = 1;
            cal_count = 0;
            memset(cal_sum, 0, sizeof(cal_sum));
            memset(cal_sum2, 0, sizeof(cal_sum2));
            memset(cal_dt_sum, 0, sizeof(cal_dt_sum));
            memset(cal_n, 0, sizeof(cal_n));
            memset(cal_dt_n, 0, sizeof(cal_dt_n));
            fprintf(stderr, "Calibration started (target: %d/ch)\n", cal_target);
            continue;
        }

        /* ENROLL,N */
        if (strncmp(line, "ENROLL,", 7) == 0) {
            if (calibrating) {
                for (int ch = 0; ch < n_channels; ch++) {
                    if (cal_n[ch] > 0) {
                        SeisCalibration *cal = &channels[ch].cal;
                        cal->input_mean = (float)(cal_sum[ch] / cal_n[ch]);
                        double var = cal_sum2[ch] / cal_n[ch]
                                   - (cal_sum[ch] / cal_n[ch])
                                     * (cal_sum[ch] / cal_n[ch]);
                        cal->input_std = sqrtf((float)(var > 0 ? var : 1e-8));
                        cal->dt_mean = (cal_dt_n[ch] > 0)
                                     ? (float)(cal_dt_sum[ch] / cal_dt_n[ch])
                                     : 0.01f;
                        cal->n_samples = cal_n[ch];
                        cal->calibrated = 1;
                        fprintf(stderr, "  ch%d cal: mean=%.1f std=%.1f dt=%.4f\n",
                                ch, cal->input_mean, cal->input_std, cal->dt_mean);
                    }
                }
                calibrating = 0;
                for (int ch = 0; ch < n_channels; ch++) {
                    SeisCalibration saved = channels[ch].cal;
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
            fprintf(stderr, "Enrollment started (target: %d, pre-scaling %s)\n",
                    enroll_target,
                    (n_channels > 0 && channels[0].cal.calibrated) ? "ON" : "OFF");
            continue;
        }

        /* FINALIZE */
        if (strcmp(line, "FINALIZE") == 0) {
            for (int ch = 0; ch < n_channels; ch++)
                channel_finalize_enrollment(&channels[ch]);
            finalized = 1;
            enrolling = 0;
            fprintf(stderr, "Finalized. %d channels.\n", n_channels);
            /* Header */
            printf("timestamp");
            for (int ch = 0; ch < n_channels; ch++)
                printf(",%s_cfc,%s_stalta", channels[ch].name, channels[ch].name);
            printf(",n_cfc_anom,n_stalta_trig\n");
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

        if (ch_id < 0 || ch_id >= MAX_CHANNELS) continue;

        if (ch_id >= n_channels) {
            for (int i = n_channels; i <= ch_id; i++) {
                char name[32];
                snprintf(name, sizeof(name), "ch%d", i);
                channel_init(&channels[i], name);
                stalta_init(&stalta[i]);
            }
            n_channels = ch_id + 1;
        }

        /* Compute dt */
        float dt = 0.01f;  /* default for 100 Hz */
        if (last_timestamps[ch_id] > 0)
            dt = timestamp - last_timestamps[ch_id];
        if (dt <= 0) dt = 0.01f;
        if (dt > 10.0f) dt = 10.0f;
        last_timestamps[ch_id] = timestamp;

        /* Calibration phase */
        if (calibrating) {
            cal_sum[ch_id] += (double)value;
            cal_sum2[ch_id] += (double)value * (double)value;
            cal_n[ch_id]++;
            if (dt < 9.9f && last_timestamps[ch_id] > 0) {
                cal_dt_sum[ch_id] += (double)dt;
                cal_dt_n[ch_id]++;
            }
            cal_count++;
            if (cal_count % 300 == 0)
                fprintf(stderr, "  Cal: %d samples\n", cal_count);
            continue;
        }

        /* Step CfC */
        channel_step_with_tau(&channels[ch_id], value, dt, tau_seismic);

        /* Step STA/LTA on prescaled value */
        float prescaled = prescale_value(value, &channels[ch_id].cal);
        stalta_update(&stalta[ch_id], prescaled);

        if (enrolling) {
            enroll_count++;
            if (enroll_count % 300 == 0)
                fprintf(stderr, "  Enroll: %d samples\n", enroll_count);
        }

        /* Score and output */
        if (finalized) {
            channels[ch_id].last_score = channel_score(&channels[ch_id]);

            printf("%.4f", timestamp);
            int n_cfc_anom = 0, n_stalta_trig = 0;
            for (int ch = 0; ch < n_channels; ch++) {
                printf(",%.4f,%.2f", channels[ch].last_score, stalta[ch].last_ratio);
                if (channels[ch].last_score < SEIS_ANOMALY_THRESH)
                    n_cfc_anom++;
                if (stalta[ch].triggered)
                    n_stalta_trig++;
            }
            printf(",%d,%d\n", n_cfc_anom, n_stalta_trig);
            fflush(stdout);
        }
    }

    fprintf(stderr, "EOF. Processed %d channels.\n", n_channels);
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
            printf("Usage: seismic_detector [--sim] [--stdin] [--verbose]\n\n");
            printf("  --sim, -s      Simulation mode (synthetic earthquakes)\n");
            printf("  --stdin        Read channel_id,timestamp,value from stdin\n");
            printf("  --verbose, -v  Show detailed per-sample output\n\n");
            printf("  3-channel CfC seismic anomaly detection with tau ablation.\n");
            printf("  Compares seismic tau vs ISS tau vs constant tau vs STA/LTA.\n\n");
            printf("  Pipe live data:\n");
            printf("    python scripts/seismic_seedlink.py | ./seismic_detector --stdin\n");
            return 0;
        }
    }

    if (stdin_mode) {
        run_stdin();
    } else if (sim_mode) {
        run_simulation(verbose);
    } else {
        printf("Usage: seismic_detector [--sim] [--stdin] [--verbose]\n");
        printf("  --sim for synthetic earthquake demo with tau ablation\n");
        printf("  --stdin for live data from scripts/seismic_seedlink.py\n");
        return 1;
    }

    return 0;
}
