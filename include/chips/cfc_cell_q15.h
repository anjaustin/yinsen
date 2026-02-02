/*
 * CFC_CELL_Q15 — Fixed-Point Sparse CfC Cell
 *
 * The Sentinel compute core. Zero floating-point operations in the hot path.
 * Designed for RV32IMAC (ESP32-C6) and any MCU without hardware FPU.
 *
 * This is the Q15 fixed-point equivalent of CFC_CELL_SPARSE from
 * cfc_cell_chip.h. Same math, same sparse structure, same enrollment model.
 * Different number representation.
 *
 * Data formats:
 *   - Input signal (x):     Q4.11 (1.0 = 2048, range [-16, +16))
 *   - Hidden state (h):     Q15   (1.0 = 32767, range [-1, +1))
 *   - Biases (b):           Q4.11 (same as pre-activations)
 *   - Decay (precomputed):  Q15   (range [0, 1), always positive)
 *   - Gate output:          Q15   (sigmoid output, range [0, 1))
 *   - Candidate output:     Q15   (tanh output, range [-1, +1))
 *   - Sparse weights:       {-1, 0, +1} as index lists (same as float path)
 *
 * Why Q4.11 for inputs/biases and Q15 for state:
 *   - Pre-activations (bias + sparse dot) can exceed [-1, +1] — they need
 *     headroom. Q4.11 gives [-16, +16) range, adequate for CfC pre-acts
 *     which typically land in [-8, +8].
 *   - Hidden state and gate/candidate outputs are bounded by activation
 *     functions to [-1, +1] or [0, 1]. Q15 gives maximum precision there.
 *   - The sparse dot product accumulates Q4.11 values (adds/subs of input
 *     elements + bias), producing a Q4.11 pre-activation that feeds
 *     directly into the LUT.
 *
 * Hot path operations (per neuron):
 *   - ~2 adds (sparse gate dot, 81% sparsity → ~2 nonzero avg)
 *   - ~2 adds (sparse candidate dot)
 *   - 2 LUT lookups + lerp (integer only, ~10 instructions each)
 *   - 3 Q15 multiplies (mixer: (1-g)*h*decay + g*c)
 *   - 2 adds (mixer accumulation)
 *   Total: ~0 float ops. ~31 integer adds + ~3 integer multiplies.
 *
 * The CfcSparseWeights structure from cfc_cell_chip.h is reused directly.
 * Weights are {-1, 0, +1} — they don't have a number format. An add is
 * an add regardless of whether you're adding float or Q4.11.
 *
 * Requirements:
 *   - Call Q15_LUT_INIT() once before first use
 *   - Build sparse weights via cfc_build_sparse() at init (from float weights)
 *   - Precompute decay as Q15 via cfc_precompute_decay_q15()
 *   - Convert biases to Q4.11 via float_to_q11() at init
 *
 * Created: February 2026
 * Part of the Yinsen Sentinel fixed-point compute stack.
 */

#ifndef TRIX_CFC_CELL_Q15_H
#define TRIX_CFC_CELL_Q15_H

#include "activation_q15.h"
#include "cfc_cell_chip.h"  /* For CfcSparseWeights, CfcSparseRow, cfc_build_sparse */

#ifdef __cplusplus
extern "C" {
#endif

/* ═══════════════════════════════════════════════════════════════════════════
 * Q15 DECAY PRECOMPUTATION
 *
 * Converts float decay values to Q15 at init time.
 * decay = exp(-dt / tau) is in [0, 1), maps directly to Q15 [0, 32767].
 * ═══════════════════════════════════════════════════════════════════════════ */

/**
 * cfc_precompute_decay_q15 — Precompute decay as Q15 values.
 *
 * Call once at init. Uses float math (not in hot path).
 *
 * @param tau         Time constants [hidden_dim] or [1] if shared
 * @param tau_shared  1 if single tau for all neurons
 * @param dt          Fixed time step
 * @param hidden_dim  Hidden dimension
 * @param decay_out   Output: Q15 decay values [hidden_dim]
 */
static inline void cfc_precompute_decay_q15(
    const float* tau, int tau_shared,
    float dt, int hidden_dim,
    int16_t* decay_out
) {
    if (tau_shared) {
        float d = expf(-dt / tau[0]);
        int16_t dq = float_to_q15(d);
        for (int i = 0; i < hidden_dim; i++) {
            decay_out[i] = dq;
        }
    } else {
        for (int i = 0; i < hidden_dim; i++) {
            float d = expf(-dt / tau[i]);
            decay_out[i] = float_to_q15(d);
        }
    }
}

/* ═══════════════════════════════════════════════════════════════════════════
 * Q15 BIAS CONVERSION
 *
 * Converts float biases to Q4.11 at init time.
 * ═══════════════════════════════════════════════════════════════════════════ */

/**
 * cfc_convert_biases_q11 — Convert float biases to Q4.11.
 *
 * Call once at init. Uses float math (not in hot path).
 *
 * @param b_float    Float biases [hidden_dim]
 * @param hidden_dim Hidden dimension
 * @param b_q11_out  Output: Q4.11 biases [hidden_dim]
 */
static inline void cfc_convert_biases_q11(
    const float* b_float, int hidden_dim,
    int16_t* b_q11_out
) {
    for (int i = 0; i < hidden_dim; i++) {
        b_q11_out[i] = float_to_q11(b_float[i]);
    }
}

/* ═══════════════════════════════════════════════════════════════════════════
 * CFC_CELL_SPARSE_Q15 — The Sentinel Hot Path
 *
 * Zero floating-point operations. Integer add/sub/mul only.
 *
 * Input x is Q4.11.
 * Hidden state h is Q15.
 *
 * The sparse dot product accumulates in Q4.11 (adds Q4.11 inputs).
 * The LUT converts Q4.11 pre-activation to Q15 activation output.
 * The mixer operates entirely in Q15.
 *
 * Note on the concat vector: x is Q4.11, h_prev is Q15. We need them
 * in the same format for the sparse dot. Since h_prev (tanh output)
 * is bounded to [-1, +1] and Q15 has range [-1, +1), we convert
 * h_prev from Q15 to Q4.11 by right-shifting 4 bits:
 *   q11_val = q15_val >> 4
 * This loses 4 bits of precision on h_prev (from 15 to 11 fractional
 * bits), but the pre-activation headroom is worth it.
 * ═══════════════════════════════════════════════════════════════════════════ */

/**
 * CFC_CELL_SPARSE_Q15 — Zero-float CfC step (fixed-point)
 *
 * @param x_q11             Input signal in Q4.11 [input_dim]
 * @param h_prev_q15        Previous hidden state in Q15 [hidden_dim]
 * @param sw                Sparse weights (built by cfc_build_sparse)
 * @param b_gate_q11        Gate biases in Q4.11 [hidden_dim]
 * @param b_cand_q11        Candidate biases in Q4.11 [hidden_dim]
 * @param decay_q15         Precomputed decay in Q15 [hidden_dim]
 * @param input_dim         Input dimension
 * @param hidden_dim        Hidden dimension
 * @param h_new_q15         Output: new hidden state in Q15 [hidden_dim]
 */
static inline void CFC_CELL_SPARSE_Q15(
    const int16_t* x_q11,
    const int16_t* h_prev_q15,
    const CfcSparseWeights* sw,
    const int16_t* b_gate_q11,
    const int16_t* b_cand_q11,
    const int16_t* decay_q15,
    int input_dim,
    int hidden_dim,
    int16_t* h_new_q15
) {
    const int concat_dim = input_dim + hidden_dim;

    /* Build concat vector in Q4.11.
     * x is already Q4.11. h_prev is Q15 — shift right 4 to get Q4.11.
     * Using VLA for concat (stack allocated, no malloc). */
    int16_t concat_q11[concat_dim];

    /* Copy x (already Q4.11) */
    memcpy(concat_q11, x_q11, input_dim * sizeof(int16_t));

    /* Convert h_prev from Q15 to Q4.11: shift right 4 bits */
    for (int i = 0; i < hidden_dim; i++) {
        concat_q11[input_dim + i] = (int16_t)(h_prev_q15[i] >> 4);
    }

    for (int i = 0; i < hidden_dim; i++) {
        /* ─────────────────────────────────────────────────────────
         * Gate: sparse dot + bias → sigmoid LUT
         * Accumulate in int32_t for overflow safety, then clamp to Q4.11
         * ───────────────────────────────────────────────────────── */
        int32_t gs = (int32_t)b_gate_q11[i];
        const int8_t *p = sw->gate[i].pos_idx;
        while (*p >= 0) { gs += concat_q11[*p]; p++; }
        p = sw->gate[i].neg_idx;
        while (*p >= 0) { gs -= concat_q11[*p]; p++; }

        /* Clamp to int16_t range before LUT lookup */
        if (gs > 32767) gs = 32767;
        if (gs < -32768) gs = -32768;
        int16_t gate = SIGMOID_Q15((int16_t)gs);  /* Q15 output [0, 32767] */

        /* ─────────────────────────────────────────────────────────
         * Candidate: sparse dot + bias → tanh LUT
         * ───────────────────────────────────────────────────────── */
        int32_t cs = (int32_t)b_cand_q11[i];
        p = sw->cand[i].pos_idx;
        while (*p >= 0) { cs += concat_q11[*p]; p++; }
        p = sw->cand[i].neg_idx;
        while (*p >= 0) { cs -= concat_q11[*p]; p++; }

        if (cs > 32767) cs = 32767;
        if (cs < -32768) cs = -32768;
        int16_t cand = TANH_Q15((int16_t)cs);  /* Q15 output [-32768, 32767] */

        /* ─────────────────────────────────────────────────────────
         * Mixer: h_new = (1 - gate) * h_prev * decay + gate * candidate
         *
         * All values are Q15. Multiply via q15_mul.
         * (1 - gate) in Q15: Q15_ONE - gate = 32767 - gate
         *
         * Three multiplies (RV32 M extension hardware multiply):
         *   retention = (Q15_ONE - gate) * h_prev          → Q15
         *   retention = retention * decay                   → Q15
         *   update    = gate * candidate                    → Q15
         *   h_new     = retention + update                  → Q15 (sat add)
         * ───────────────────────────────────────────────────────── */
        int16_t one_minus_gate = (int16_t)(Q15_ONE - gate);
        int16_t retention = q15_mul(one_minus_gate, h_prev_q15[i]);
        retention = q15_mul(retention, decay_q15[i]);
        int16_t update = q15_mul(gate, cand);
        h_new_q15[i] = q15_sat_add(retention, update);
    }
}

/* ═══════════════════════════════════════════════════════════════════════════
 * CONVENIENCE: Convert float input to Q4.11 vector
 *
 * For use at the ADC/sensor interface. In production, the ADC output
 * would already be integer and this conversion is unnecessary.
 * For testing, this lets us feed float sensor data into the Q15 path.
 * ═══════════════════════════════════════════════════════════════════════════ */

/**
 * cfc_convert_input_q11 — Convert float input vector to Q4.11.
 *
 * @param x_float    Float input [input_dim]
 * @param input_dim  Input dimension
 * @param x_q11_out  Output: Q4.11 input [input_dim]
 */
static inline void cfc_convert_input_q11(
    const float* x_float, int input_dim,
    int16_t* x_q11_out
) {
    for (int i = 0; i < input_dim; i++) {
        x_q11_out[i] = float_to_q11(x_float[i]);
    }
}

/**
 * cfc_convert_state_to_float — Convert Q15 hidden state back to float.
 *
 * For test/debug comparison against float path.
 *
 * @param h_q15      Q15 hidden state [hidden_dim]
 * @param hidden_dim Hidden dimension
 * @param h_float    Output: float hidden state [hidden_dim]
 */
static inline void cfc_convert_state_to_float(
    const int16_t* h_q15, int hidden_dim,
    float* h_float
) {
    for (int i = 0; i < hidden_dim; i++) {
        h_float[i] = q15_to_float(h_q15[i]);
    }
}

/* ═══════════════════════════════════════════════════════════════════════════
 * PCA DISCRIMINANT — Q15 version
 *
 * The 268-byte discriminant from enrollment. In Q15, the PCA projection
 * vectors and mean vector are int16_t. The dot product uses Q15 multiply
 * and int32_t accumulation.
 * ═══════════════════════════════════════════════════════════════════════════ */

/**
 * cfc_pca_score_q15 — Compute anomaly score from hidden state.
 *
 * Projects (h - mean) onto principal components and returns the
 * reconstruction error as a Q15 value.
 *
 * @param h_q15      Hidden state in Q15 [hidden_dim]
 * @param mean_q15   Enrolled mean in Q15 [hidden_dim]
 * @param pcs_q15    Principal components in Q15 [n_pcs * hidden_dim]
 * @param hidden_dim Hidden dimension
 * @param n_pcs      Number of principal components
 * @return           Anomaly score (higher = more anomalous)
 *                   Returned as int64_t because squared Q15 values
 *                   can overflow int32_t when summed.
 */
static inline int64_t cfc_pca_score_q15(
    const int16_t* h_q15,
    const int16_t* mean_q15,
    const int16_t* pcs_q15,
    int hidden_dim,
    int n_pcs
) {
    /* Compute deviation: d = h - mean */
    int16_t dev[hidden_dim];
    for (int i = 0; i < hidden_dim; i++) {
        dev[i] = q15_sat_sub(h_q15[i], mean_q15[i]);
    }

    /* Project onto each PC and accumulate squared reconstruction */
    int64_t reconstructed_energy = 0;
    for (int pc = 0; pc < n_pcs; pc++) {
        /* Dot product: projection = dev . pc_vector */
        const int16_t* pc_vec = pcs_q15 + pc * hidden_dim;
        int32_t proj = 0;
        for (int i = 0; i < hidden_dim; i++) {
            proj += (int32_t)dev[i] * (int32_t)pc_vec[i];
        }
        /* proj is in Q30 (Q15 * Q15). Shift to Q15. */
        int16_t proj_q15 = (int16_t)((proj + (1 << 14)) >> 15);

        /* Accumulate squared projection (energy captured by PCs) */
        reconstructed_energy += (int64_t)proj_q15 * (int64_t)proj_q15;
    }

    /* Total energy of deviation */
    int64_t total_energy = 0;
    for (int i = 0; i < hidden_dim; i++) {
        total_energy += (int64_t)dev[i] * (int64_t)dev[i];
    }

    /* Anomaly score = total_energy - reconstructed_energy
     * High score = deviation is NOT explained by the learned PCs
     * = novel/anomalous pattern */
    return total_energy - reconstructed_energy;
}

#ifdef __cplusplus
}
#endif

#endif /* TRIX_CFC_CELL_Q15_H */
