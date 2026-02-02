/*
 * CfC_CELL Frozen Chip — Atomic Liquid Neural Primitive
 *
 * "The 6502 of Liquid Neural Networks"
 *
 * This is the CfC operation itself, frozen as a single computational
 * primitive. Not a trained network, but the ALGORITHM as a building block.
 *
 * Like how the 6502 has ADC (add with carry) as an atomic instruction,
 * this is CfC_CELL as an atomic operation.
 *
 * The computation is frozen. The parameters are inputs.
 *
 * Created by: Tripp + Manus
 * Date: January 31, 2026
 */

#ifndef TRIXC_CFC_CELL_CHIP_H
#define TRIXC_CFC_CELL_CHIP_H

/*
 * Yinsen integration: use yinsen_ prefix for ONNX primitives.
 * Original chip used trix_onnx_ prefix; adapted for yinsen's onnx_shapes.h.
 */
#include "onnx_shapes.h"
#include <math.h>
#include <string.h>

/* Compatibility aliases for the chip's trix_onnx_ calls */
#ifndef trix_onnx_gemm
#define trix_onnx_gemm     yinsen_gemm
#define trix_onnx_sigmoid  yinsen_sigmoid
#define trix_onnx_tanh     yinsen_tanh
#endif

#ifdef __cplusplus
extern "C" {
#endif

/* ═══════════════════════════════════════════════════════════════════════════
 * CfC_CELL: The Frozen Primitive
 *
 * This is the atomic unit of liquid computation.
 * A single function call that replaces ODE integration with closed-form math.
 *
 * The routing IS the integration.
 * ═══════════════════════════════════════════════════════════════════════════ */

/**
 * CFC_CELL_GENERIC — Variable time step
 *
 * Computes: h_new = (1 - gate) * h_prev * decay + gate * candidate
 *
 * Where:
 *   gate = σ(W_gate @ [x, h] + b_gate)
 *   candidate = tanh(W_cand @ [x, h] + b_cand)
 *   decay = exp(-dt / τ)
 *
 * @param x            Input signal [input_dim]
 * @param h_prev       Previous hidden state [hidden_dim]
 * @param dt           Time delta (seconds or normalized)
 * @param W_gate       Gate weights [hidden_dim, input_dim + hidden_dim]
 * @param b_gate       Gate biases [hidden_dim]
 * @param W_cand       Candidate weights [hidden_dim, input_dim + hidden_dim]
 * @param b_cand       Candidate biases [hidden_dim]
 * @param tau          Time constants [hidden_dim] or [1]
 * @param tau_shared   If true, single tau for all neurons
 * @param input_dim    Input dimension
 * @param hidden_dim   Hidden dimension
 * @param h_new        Output: new hidden state [hidden_dim]
 *
 * Frozen Operations:
 *   - CONCAT (topology)
 *   - GEMM × 2 (MUL + ADD)
 *   - SIGMOID (EXP)
 *   - TANH (EXP)
 *   - DECAY (EXP + DIV)
 *   - MIX (MUL + ADD)
 *
 * All decompose to the 5 Primes: ADD, MUL, EXP, MAX, CONST
 *
 * Performance: O(hidden_dim * (input_dim + hidden_dim))
 * Memory: (2*input_dim + 7*hidden_dim) * sizeof(float) bytes on stack
 * Determinism: Bit-identical for same inputs
 */
static inline void CFC_CELL_GENERIC(
    const float* x,
    const float* h_prev,
    float dt,
    const float* W_gate,
    const float* b_gate,
    const float* W_cand,
    const float* b_cand,
    const float* tau,
    int tau_shared,
    int input_dim,
    int hidden_dim,
    float* h_new
) {
    const int concat_dim = input_dim + hidden_dim;

    /* Stack allocation (no malloc) */
    float concat[concat_dim];
    float gate_pre[hidden_dim];
    float gate[hidden_dim];
    float cand_pre[hidden_dim];
    float candidate[hidden_dim];
    float decay[hidden_dim];

    /* ─────────────────────────────────────────────────────────────────────
     * Step 1: Concatenate [x; h_prev]
     * Pure topology, no computation
     * ───────────────────────────────────────────────────────────────────── */
    memcpy(concat, x, input_dim * sizeof(float));
    memcpy(concat + input_dim, h_prev, hidden_dim * sizeof(float));

    /* ─────────────────────────────────────────────────────────────────────
     * Step 2: Gate computation
     * gate = σ(W_gate @ concat + b_gate)
     * Frozen: GEMM + SIGMOID
     * ───────────────────────────────────────────────────────────────────── */
    trix_onnx_gemm(concat, W_gate, b_gate,
                   gate_pre, 1, hidden_dim, concat_dim, 1.0f, 1.0f);

    for (int i = 0; i < hidden_dim; i++) {
        gate[i] = trix_onnx_sigmoid(gate_pre[i]);
    }

    /* ─────────────────────────────────────────────────────────────────────
     * Step 3: Candidate computation
     * candidate = tanh(W_cand @ concat + b_cand)
     * Frozen: GEMM + TANH
     * ───────────────────────────────────────────────────────────────────── */
    trix_onnx_gemm(concat, W_cand, b_cand,
                   cand_pre, 1, hidden_dim, concat_dim, 1.0f, 1.0f);

    for (int i = 0; i < hidden_dim; i++) {
        candidate[i] = trix_onnx_tanh(cand_pre[i]);
    }

    /* ─────────────────────────────────────────────────────────────────────
     * Step 4: Decay computation (the "time shortcut")
     * decay = exp(-dt / tau)
     * Frozen: EXP + DIV
     *
     * This is where CfC magic happens: instead of iterating through
     * time steps, we jump directly to the answer.
     * ───────────────────────────────────────────────────────────────────── */
    if (tau_shared) {
        float decay_scalar = expf(-dt / tau[0]);
        for (int i = 0; i < hidden_dim; i++) {
            decay[i] = decay_scalar;
        }
    } else {
        for (int i = 0; i < hidden_dim; i++) {
            decay[i] = expf(-dt / tau[i]);
        }
    }

    /* ─────────────────────────────────────────────────────────────────────
     * Step 5: State update (the "mixer")
     * h_new = (1 - gate) * h_prev * decay + gate * candidate
     * Frozen: MUL + ADD + CONST(1)
     * ───────────────────────────────────────────────────────────────────── */
    for (int i = 0; i < hidden_dim; i++) {
        float retention = (1.0f - gate[i]) * h_prev[i] * decay[i];
        float update = gate[i] * candidate[i];
        h_new[i] = retention + update;
    }
}

/* ═══════════════════════════════════════════════════════════════════════════
 * CFC_CELL_FIXED: Precomputed Decay Variant
 *
 * For fixed sample rate systems, decay can be precomputed.
 * This eliminates exp() at runtime for maximum performance.
 * ═══════════════════════════════════════════════════════════════════════════ */

/**
 * CFC_CELL_FIXED — Fixed time step (precomputed decay)
 *
 * Same as CFC_CELL_GENERIC but with precomputed decay.
 * Faster: no exp() at runtime.
 *
 * @param decay_precomputed  Precomputed exp(-dt/tau) [hidden_dim]
 *
 * All other parameters same as CFC_CELL_GENERIC.
 */
static inline void CFC_CELL_FIXED(
    const float* x,
    const float* h_prev,
    const float* W_gate,
    const float* b_gate,
    const float* W_cand,
    const float* b_cand,
    const float* decay_precomputed,
    int input_dim,
    int hidden_dim,
    float* h_new
) {
    const int concat_dim = input_dim + hidden_dim;

    float concat[concat_dim];
    float gate_pre[hidden_dim];
    float gate[hidden_dim];
    float cand_pre[hidden_dim];
    float candidate[hidden_dim];

    /* Concatenate */
    memcpy(concat, x, input_dim * sizeof(float));
    memcpy(concat + input_dim, h_prev, hidden_dim * sizeof(float));

    /* Gate */
    trix_onnx_gemm(concat, W_gate, b_gate,
                   gate_pre, 1, hidden_dim, concat_dim, 1.0f, 1.0f);
    for (int i = 0; i < hidden_dim; i++) {
        gate[i] = trix_onnx_sigmoid(gate_pre[i]);
    }

    /* Candidate */
    trix_onnx_gemm(concat, W_cand, b_cand,
                   cand_pre, 1, hidden_dim, concat_dim, 1.0f, 1.0f);
    for (int i = 0; i < hidden_dim; i++) {
        candidate[i] = trix_onnx_tanh(cand_pre[i]);
    }

    /* Mix with precomputed decay */
    for (int i = 0; i < hidden_dim; i++) {
        float retention = (1.0f - gate[i]) * h_prev[i] * decay_precomputed[i];
        float update = gate[i] * candidate[i];
        h_new[i] = retention + update;
    }
}

/* ═══════════════════════════════════════════════════════════════════════════
 * Helper: Precompute Decay
 *
 * For CFC_CELL_FIXED, precompute decay values.
 * ═══════════════════════════════════════════════════════════════════════════ */

/**
 * cfc_precompute_decay — Precompute decay for fixed dt
 *
 * @param tau          Time constants [hidden_dim] or [1]
 * @param tau_shared   If true, single tau for all neurons
 * @param dt           Fixed time delta
 * @param hidden_dim   Hidden dimension
 * @param decay_out    Output: precomputed decay [hidden_dim]
 */
static inline void cfc_precompute_decay(
    const float* tau,
    int tau_shared,
    float dt,
    int hidden_dim,
    float* decay_out
) {
    if (tau_shared) {
        float decay_scalar = expf(-dt / tau[0]);
        for (int i = 0; i < hidden_dim; i++) {
            decay_out[i] = decay_scalar;
        }
    } else {
        for (int i = 0; i < hidden_dim; i++) {
            decay_out[i] = expf(-dt / tau[i]);
        }
    }
}

/* ═══════════════════════════════════════════════════════════════════════════
 * CFC_CELL_LUT: Maximum Performance Variant
 *
 * Combines three optimizations identified by Probe 2:
 *   1. Ternary-as-float GEMM:  weights are {-1,0,+1} but stored as float.
 *      FPU multiplies 1.0 at the same speed as 0.3. No branches.
 *   2. LUT+lerp activations:   256-entry table + linear interpolation.
 *      200x more accurate than FAST3, only 4ns slower per step.
 *   3. Precomputed decay:       exp(-dt/tau) computed once at init.
 *      Eliminates all expf() from the hot path.
 *
 * Benchmarked (Apple M-series, HIDDEN_DIM=8):
 *   CFC_CELL_GENERIC (precise):  54 ns/step
 *   CFC_CELL_LUT:                35 ns/step  (1.54x faster)
 *
 * Accuracy (L2 divergence from precise after 1000 steps): 6.8e-4
 * Compare: FAST3 = 0.137 (200x worse)
 *
 * Requirements:
 *   - Call ACTIVATION_LUT_INIT() once before first use
 *   - Precompute decay via cfc_precompute_decay()
 *   - Weights can be any float, but ternary {-1,0,+1} recommended
 *     for auditability + 16x weight compression (2-bit packed storage)
 *
 * Cache footprint:
 *   Weights (ternary-as-float): 640 bytes (shared across channels)
 *   LUTs (sigmoid + tanh):     2048 bytes (shared, read-only)
 *   Decay:                       32 bytes (shared for fixed-rate)
 *   Per-channel h_state:          32 bytes
 *   Total 8-channel hot set:   ~4.4 KB (13.4% of 32KB L1D)
 * ═══════════════════════════════════════════════════════════════════════════ */

#ifndef TRIX_ACTIVATION_CHIP_H
/* Forward-declare LUT functions if activation_chip.h not yet included.
 * If it IS included (which it should be), these are already defined. */
static inline float SIGMOID_CHIP_LUT(float x);
static inline float TANH_CHIP_LUT(float x);
#endif

static inline void CFC_CELL_LUT(
    const float* x,
    const float* h_prev,
    const float* W_gate,
    const float* b_gate,
    const float* W_cand,
    const float* b_cand,
    const float* decay_precomputed,
    int input_dim,
    int hidden_dim,
    float* h_new
) {
    const int concat_dim = input_dim + hidden_dim;

    float concat[concat_dim];
    float gate_pre[hidden_dim];
    float gate[hidden_dim];
    float cand_pre[hidden_dim];
    float candidate[hidden_dim];

    /* Concatenate */
    memcpy(concat, x, input_dim * sizeof(float));
    memcpy(concat + input_dim, h_prev, hidden_dim * sizeof(float));

    /* Gate = sigmoid(W_gate @ concat + b_gate) — LUT+lerp */
    trix_onnx_gemm(concat, W_gate, b_gate,
                   gate_pre, 1, hidden_dim, concat_dim, 1.0f, 1.0f);
    for (int i = 0; i < hidden_dim; i++) {
        gate[i] = SIGMOID_CHIP_LUT(gate_pre[i]);
    }

    /* Candidate = tanh(W_cand @ concat + b_cand) — LUT+lerp */
    trix_onnx_gemm(concat, W_cand, b_cand,
                   cand_pre, 1, hidden_dim, concat_dim, 1.0f, 1.0f);
    for (int i = 0; i < hidden_dim; i++) {
        candidate[i] = TANH_CHIP_LUT(cand_pre[i]);
    }

    /* Mix with precomputed decay — no expf in hot path */
    for (int i = 0; i < hidden_dim; i++) {
        float retention = (1.0f - gate[i]) * h_prev[i] * decay_precomputed[i];
        float update = gate[i] * candidate[i];
        h_new[i] = retention + update;
    }
}

/* ═══════════════════════════════════════════════════════════════════════════
 * Composability Examples
 * ═══════════════════════════════════════════════════════════════════════════ */

/*
 * Example 1: Single Cell
 *
 * CFC_CELL_GENERIC(x, h_prev, dt, W_gate, b_gate, W_cand, b_cand,
 *                  tau, tau_shared, input_dim, hidden_dim, h_new);
 *
 * Example 2: Stacked Cells (Deep Network)
 *
 * CFC_CELL_GENERIC(x, h0_prev, dt, W1_gate, b1_gate, W1_cand, b1_cand,
 *                  tau1, 0, input_dim, hidden_dim, h1);
 * CFC_CELL_GENERIC(h1, h1_prev, dt, W2_gate, b2_gate, W2_cand, b2_cand,
 *                  tau2, 0, hidden_dim, hidden_dim, h2);
 *
 * Example 3: Sequence Processing
 *
 * for (int t = 0; t < seq_len; t++) {
 *     CFC_CELL_GENERIC(inputs[t], h_current, dt, W_gate, b_gate,
 *                      W_cand, b_cand, tau, tau_shared,
 *                      input_dim, hidden_dim, h_current);
 * }
 *
 * Example 4: Fixed dt (Precomputed Decay)
 *
 * float decay[hidden_dim];
 * cfc_precompute_decay(tau, tau_shared, dt, hidden_dim, decay);
 *
 * for (int t = 0; t < seq_len; t++) {
 *     CFC_CELL_FIXED(inputs[t], h_current, W_gate, b_gate,
 *                    W_cand, b_cand, decay, input_dim, hidden_dim, h_current);
 * }
 *
 * Example 5: Maximum Performance (LUT+lerp + precomputed decay)
 *
 * #include "chips/activation_chip.h"  // must be included before cfc_cell_chip.h
 * ACTIVATION_LUT_INIT();  // once at startup
 *
 * float decay[hidden_dim];
 * cfc_precompute_decay(tau, tau_shared, dt, hidden_dim, decay);
 *
 * for (int t = 0; t < seq_len; t++) {
 *     CFC_CELL_LUT(inputs[t], h_current, W_gate, b_gate,
 *                  W_cand, b_cand, decay, input_dim, hidden_dim, h_current);
 * }
 *
 * // 35 ns/step on Apple M-series (1.54x faster than GENERIC)
 * // 200x more accurate than FAST3 approximations
 * // Weights can be ternary {-1,0,+1} for auditability (Probe 1 validated)
 */

#ifdef __cplusplus
}
#endif

#endif /* TRIXC_CFC_CELL_CHIP_H */
