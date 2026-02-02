/*
 * DECAY_CHIP — Frozen Temporal Decay Primitive
 *
 * decay = exp(-dt / tau)
 *
 * This is the "time shortcut" that makes CfC liquid:
 * instead of iterating through time steps, we jump directly
 * to the closed-form solution.
 *
 * Three modes:
 *   GENERIC:     Compute decay at runtime for variable dt
 *   PRECOMPUTE:  Compute once for fixed-rate sensors
 *   FAST:        Schraudolph bit-trick, no libm
 *
 * Created by: Tripp + Claude
 * Date: January 31, 2026
 */

#ifndef TRIX_DECAY_CHIP_H
#define TRIX_DECAY_CHIP_H

#include <math.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * DECAY_CHIP_SCALAR — Single decay value
 *
 * Returns exp(-dt / tau). The atomic time operation.
 */
static inline float DECAY_CHIP_SCALAR(float dt, float tau) {
    return expf(-dt / tau);
}

/**
 * DECAY_CHIP_SCALAR_FAST — Schraudolph approximation
 *
 * ~4% relative error. No libm. ~5 instructions.
 */
static inline float DECAY_CHIP_SCALAR_FAST(float dt, float tau) {
    float x = -dt / tau;
    if (x < -88.0f) return 0.0f;
    union { float f; int32_t i; } u;
    u.i = (int32_t)(x * 12102203.0f + 1064866805.0f);
    return u.f;
}

/**
 * DECAY_CHIP_SHARED — Fill vector with single decay value
 *
 * For tau_shared mode: one tau for all neurons.
 * Computes exp once, broadcasts to all hidden_dim elements.
 */
static inline void DECAY_CHIP_SHARED(
    float dt, float tau,
    float* decay_out, int hidden_dim
) {
    float d = expf(-dt / tau);
    for (int i = 0; i < hidden_dim; i++) {
        decay_out[i] = d;
    }
}

/**
 * DECAY_CHIP_PER_NEURON — Per-neuron tau values
 *
 * Each neuron has its own time constant.
 * hidden_dim calls to expf.
 */
static inline void DECAY_CHIP_PER_NEURON(
    float dt, const float* tau,
    float* decay_out, int hidden_dim
) {
    for (int i = 0; i < hidden_dim; i++) {
        decay_out[i] = expf(-dt / tau[i]);
    }
}

/**
 * DECAY_CHIP_PER_NEURON_FAST — Per-neuron, Schraudolph
 *
 * No libm. Suitable for MCU without hardware FPU exp.
 */
static inline void DECAY_CHIP_PER_NEURON_FAST(
    float dt, const float* tau,
    float* decay_out, int hidden_dim
) {
    for (int i = 0; i < hidden_dim; i++) {
        float x = -dt / tau[i];
        if (x < -88.0f) { decay_out[i] = 0.0f; continue; }
        union { float f; int32_t i; } u;
        u.i = (int32_t)(x * 12102203.0f + 1064866805.0f);
        decay_out[i] = u.f;
    }
}

/**
 * DECAY_CHIP_PRECOMPUTE — Precompute for fixed sample rate
 *
 * Call once at init. Eliminates all exp() from the hot path.
 * Store the result and pass to CFC_CELL_FIXED.
 */
static inline void DECAY_CHIP_PRECOMPUTE(
    float dt,
    const float* tau, int tau_shared,
    float* decay_out, int hidden_dim
) {
    if (tau_shared) {
        DECAY_CHIP_SHARED(dt, tau[0], decay_out, hidden_dim);
    } else {
        DECAY_CHIP_PER_NEURON(dt, tau, decay_out, hidden_dim);
    }
}

/* ═══════════════════════════════════════════════════════════════════════════
 * VARIABLE-RATE DECAY
 *
 * For sensors with irregular timestamps (GPS, event cameras, CAN bus).
 * Each sample has its own dt.
 * ═══════════════════════════════════════════════════════════════════════════ */

/**
 * DECAY_CHIP_VARIABLE — Variable dt per timestep, shared tau
 *
 * For irregular time series: each input arrives at a different time.
 * Computes exp(-dt_t / tau) where dt_t varies per step.
 */
static inline float DECAY_CHIP_VARIABLE(float dt_t, float tau) {
    return expf(-dt_t / tau);
}

#ifdef __cplusplus
}
#endif

#endif /* TRIX_DECAY_CHIP_H */
