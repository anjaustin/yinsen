/*
 * NORM_CHIP — Frozen Normalization Primitives
 *
 * LayerNorm, RMSNorm, and BatchNorm for embedded inference.
 *
 * Why in yinsen?
 *   - Deep CfC stacks (cell → cell → cell) need normalization
 *     between layers to prevent activation drift
 *   - RMSNorm is what modern transformers use (LLaMA, etc.)
 *   - LayerNorm is the classic
 *   - Online running stats for streaming sensor data
 *
 * Created by: Tripp + Claude
 * Date: January 31, 2026
 */

#ifndef TRIX_NORM_CHIP_H
#define TRIX_NORM_CHIP_H

#include <math.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * LAYERNORM_CHIP — Layer Normalization
 *
 * out[i] = gamma[i] * (x[i] - mean) / sqrt(var + eps) + beta[i]
 *
 * 3 passes over the data: mean, variance, normalize.
 * For hidden_dim=32, this is ~100 FLOPs. Negligible vs GEMM.
 */
static inline void LAYERNORM_CHIP(
    const float* x,
    const float* gamma,
    const float* beta,
    float* out,
    int n,
    float eps
) {
    /* Mean */
    float mean = 0.0f;
    for (int i = 0; i < n; i++) mean += x[i];
    mean /= (float)n;

    /* Variance */
    float var = 0.0f;
    for (int i = 0; i < n; i++) {
        float d = x[i] - mean;
        var += d * d;
    }
    var /= (float)n;

    /* Normalize + affine */
    float inv_std = 1.0f / sqrtf(var + eps);
    for (int i = 0; i < n; i++) {
        out[i] = gamma[i] * (x[i] - mean) * inv_std + beta[i];
    }
}

/**
 * RMSNORM_CHIP — Root Mean Square Normalization
 *
 * out[i] = gamma[i] * x[i] / sqrt(mean(x²) + eps)
 *
 * Simpler than LayerNorm: no mean subtraction, no beta.
 * Used in LLaMA, Mistral, and other modern architectures.
 * 2 passes: RMS, normalize.
 */
static inline void RMSNORM_CHIP(
    const float* x,
    const float* gamma,
    float* out,
    int n,
    float eps
) {
    /* RMS */
    float sum_sq = 0.0f;
    for (int i = 0; i < n; i++) sum_sq += x[i] * x[i];
    float rms = sqrtf(sum_sq / (float)n + eps);

    /* Normalize + scale */
    float inv_rms = 1.0f / rms;
    for (int i = 0; i < n; i++) {
        out[i] = gamma[i] * x[i] * inv_rms;
    }
}

/**
 * RMSNORM_CHIP_NOGAMMA — RMSNorm without learned scale
 *
 * Just normalizes to unit RMS. No parameters.
 * Use this when you don't have trained gamma weights.
 */
static inline void RMSNORM_CHIP_NOGAMMA(
    const float* x, float* out, int n, float eps
) {
    float sum_sq = 0.0f;
    for (int i = 0; i < n; i++) sum_sq += x[i] * x[i];
    float inv_rms = 1.0f / sqrtf(sum_sq / (float)n + eps);
    for (int i = 0; i < n; i++) {
        out[i] = x[i] * inv_rms;
    }
}

/* ═══════════════════════════════════════════════════════════════════════════
 * ONLINE / STREAMING NORMALIZATION
 *
 * For real-time sensor data where you can't see the whole sequence.
 * Welford's algorithm for running mean/variance.
 * ═══════════════════════════════════════════════════════════════════════════ */

/**
 * Running statistics state (per feature dimension)
 */
typedef struct {
    float mean;
    float m2;     /* Sum of squared differences from mean */
    int count;
} RunningStats;

/**
 * RUNNING_STATS_INIT — Initialize running stats to zero
 */
static inline void RUNNING_STATS_INIT(RunningStats* stats) {
    stats->mean = 0.0f;
    stats->m2 = 0.0f;
    stats->count = 0;
}

/**
 * RUNNING_STATS_UPDATE — Welford's online algorithm
 *
 * Update running mean and variance with a new sample.
 * Numerically stable for long sequences.
 */
static inline void RUNNING_STATS_UPDATE(RunningStats* stats, float x) {
    stats->count++;
    float delta = x - stats->mean;
    stats->mean += delta / (float)stats->count;
    float delta2 = x - stats->mean;
    stats->m2 += delta * delta2;
}

/**
 * RUNNING_STATS_VARIANCE — Current variance estimate
 */
static inline float RUNNING_STATS_VARIANCE(const RunningStats* stats) {
    if (stats->count < 2) return 0.0f;
    return stats->m2 / (float)stats->count;
}

/**
 * ONLINE_NORMALIZE_CHIP — Normalize a sample using running stats
 *
 * For streaming: normalize each feature dimension independently
 * using the running mean/variance accumulated so far.
 *
 * @param x       Input sample [n] (e.g., sensor reading)
 * @param stats   Running statistics [n] (one per feature)
 * @param out     Output: normalized sample [n]
 * @param n       Feature dimension
 * @param eps     Stability epsilon
 * @param update  If true, update running stats with this sample
 */
static inline void ONLINE_NORMALIZE_CHIP(
    const float* x,
    RunningStats* stats,
    float* out,
    int n,
    float eps,
    int update
) {
    for (int i = 0; i < n; i++) {
        if (update) {
            RUNNING_STATS_UPDATE(&stats[i], x[i]);
        }
        float var = RUNNING_STATS_VARIANCE(&stats[i]);
        float inv_std = 1.0f / sqrtf(var + eps);
        out[i] = (x[i] - stats[i].mean) * inv_std;
    }
}

#ifdef __cplusplus
}
#endif

#endif /* TRIX_NORM_CHIP_H */
