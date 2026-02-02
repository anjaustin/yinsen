/*
 * YINSEN CfC - Gated Recurrence with Time Constants
 *
 * A recurrent cell inspired by Hasani et al.'s CfC architecture.
 * Incorporates explicit time step (dt) via exponential decay.
 *
 * Update rule:
 *   h(t) = (1 - gate) * h_prev * decay + gate * candidate
 *
 * Where:
 *   gate      = sigmoid(W_gate @ [x; h_prev] + b_gate)
 *   candidate = tanh(W_cand @ [x; h_prev] + b_cand)
 *   decay     = exp(-dt / tau)
 *
 * Verification status (single platform only):
 *   - Determinism: TESTED (identical calls produce identical results)
 *   - Bounded outputs: TESTED
 *   - Numerical stability: TESTED (10,000 iterations)
 *
 * NOT tested:
 *   - Cross-platform determinism
 *   - Comparison to ODE solver
 *   - Comparison to GRU/LSTM
 */

#ifndef YINSEN_CFC_H
#define YINSEN_CFC_H

#include "onnx_shapes.h"
#include <string.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ============================================================================
 * CfC PARAMETERS
 * ============================================================================ */

typedef struct {
    int input_dim;
    int hidden_dim;

    const float* W_gate;    /* [hidden_dim, input_dim + hidden_dim] */
    const float* b_gate;    /* [hidden_dim] */
    const float* W_cand;    /* [hidden_dim, input_dim + hidden_dim] */
    const float* b_cand;    /* [hidden_dim] */
    const float* tau;       /* [hidden_dim] or [1] */
    int tau_shared;         /* If true, single tau for all neurons */
} CfCParams;

typedef struct {
    int input_dim;
    int hidden_dim;

    const float* W_gate;
    const float* b_gate;
    const float* W_cand;
    const float* b_cand;
    const float* decay;     /* Precomputed exp(-dt/tau) [hidden_dim] */
} CfCParamsFixed;

typedef struct {
    int hidden_dim;
    int output_dim;
    const float* W_out;     /* [output_dim, hidden_dim] -- gemm(h[1,hid], W_out[out,hid]^T) */
    const float* b_out;     /* [output_dim] */
} CfCOutputParams;

/* ============================================================================
 * CfC CELL - Single step
 * ============================================================================ */

/*
 * Stack usage: (input_dim + 6 * hidden_dim) * sizeof(float) bytes.
 *   hidden_dim=256:  ~6.5 KB
 *   hidden_dim=1024: ~25 KB
 *   hidden_dim=4096: ~96 KB
 *
 * Callers on stack-constrained platforms (embedded, threads with small
 * stacks) must ensure sufficient stack space.
 */
static inline void yinsen_cfc_cell(
    const float* x,
    const float* h_prev,
    float dt,
    const CfCParams* params,
    float* h_new
) {
    const int in_dim = params->input_dim;
    const int hid_dim = params->hidden_dim;
    const int concat_dim = in_dim + hid_dim;

    /* Stack allocation (no malloc) */
    float concat[concat_dim];
    float gate_pre[hid_dim];
    float gate[hid_dim];
    float cand_pre[hid_dim];
    float candidate[hid_dim];

    /* Step 1: Concatenate [x; h_prev] */
    memcpy(concat, x, in_dim * sizeof(float));
    memcpy(concat + in_dim, h_prev, hid_dim * sizeof(float));

    /* Step 2: Gate = sigmoid(W_gate @ concat + b_gate) */
    yinsen_gemm(concat, params->W_gate, params->b_gate,
                gate_pre, 1, hid_dim, concat_dim, 1.0f, 1.0f);
    for (int i = 0; i < hid_dim; i++) {
        gate[i] = yinsen_sigmoid(gate_pre[i]);
    }

    /* Step 3: Candidate = tanh(W_cand @ concat + b_cand) */
    yinsen_gemm(concat, params->W_cand, params->b_cand,
                cand_pre, 1, hid_dim, concat_dim, 1.0f, 1.0f);
    for (int i = 0; i < hid_dim; i++) {
        candidate[i] = yinsen_tanh(cand_pre[i]);
    }

    /* Step 4: Decay = exp(-dt / tau)
     * Initialize to NAN so invalid tau propagates cleanly through Step 5
     * rather than reading uninitialized memory (undefined behavior). */
    float decay[hid_dim];
    for (int i = 0; i < hid_dim; i++) decay[i] = NAN;
    if (params->tau_shared) {
        /* Validate tau > 0 - required for continuous-time behavior */
        if (params->tau[0] <= 0.0f) {
            /* Invalid tau - return NaN to signal error */
            for (int i = 0; i < hid_dim; i++) {
                h_new[i] = NAN;
            }
            return;
        }
        float decay_scalar = expf(-dt / params->tau[0]);
        for (int i = 0; i < hid_dim; i++) {
            decay[i] = decay_scalar;
        }
    } else {
        for (int i = 0; i < hid_dim; i++) {
            /* Validate tau > 0 - required for continuous-time behavior */
            if (params->tau[i] <= 0.0f) {
                h_new[i] = NAN;
                continue;
            }
            decay[i] = expf(-dt / params->tau[i]);
        }
    }

    /* Step 5: h_new = (1 - gate) * h_prev * decay + gate * candidate */
    for (int i = 0; i < hid_dim; i++) {
        float retention = (1.0f - gate[i]) * h_prev[i] * decay[i];
        float update = gate[i] * candidate[i];
        h_new[i] = retention + update;
    }
}

/* CfC with precomputed decay (faster for fixed dt) */
static inline void yinsen_cfc_cell_fixed(
    const float* x,
    const float* h_prev,
    const CfCParamsFixed* params,
    float* h_new
) {
    const int in_dim = params->input_dim;
    const int hid_dim = params->hidden_dim;
    const int concat_dim = in_dim + hid_dim;

    float concat[concat_dim];
    float gate_pre[hid_dim];
    float gate[hid_dim];
    float cand_pre[hid_dim];
    float candidate[hid_dim];

    memcpy(concat, x, in_dim * sizeof(float));
    memcpy(concat + in_dim, h_prev, hid_dim * sizeof(float));

    yinsen_gemm(concat, params->W_gate, params->b_gate,
                gate_pre, 1, hid_dim, concat_dim, 1.0f, 1.0f);
    for (int i = 0; i < hid_dim; i++) {
        gate[i] = yinsen_sigmoid(gate_pre[i]);
    }

    yinsen_gemm(concat, params->W_cand, params->b_cand,
                cand_pre, 1, hid_dim, concat_dim, 1.0f, 1.0f);
    for (int i = 0; i < hid_dim; i++) {
        candidate[i] = yinsen_tanh(cand_pre[i]);
    }

    for (int i = 0; i < hid_dim; i++) {
        float retention = (1.0f - gate[i]) * h_prev[i] * params->decay[i];
        float update = gate[i] * candidate[i];
        h_new[i] = retention + update;
    }
}

/* ============================================================================
 * CfC SEQUENCE PROCESSING
 * ============================================================================ */

static inline void yinsen_cfc_forward(
    const float* inputs,
    int seq_len,
    float dt,
    const CfCParams* params,
    const float* h_init,
    float* outputs,
    float* h_final
) {
    const int hid_dim = params->hidden_dim;
    const int in_dim = params->input_dim;

    float h_current[hid_dim];
    if (h_init) {
        memcpy(h_current, h_init, hid_dim * sizeof(float));
    } else {
        memset(h_current, 0, hid_dim * sizeof(float));
    }

    for (int t = 0; t < seq_len; t++) {
        const float* x_t = inputs + t * in_dim;
        float* out_t = outputs + t * hid_dim;

        yinsen_cfc_cell(x_t, h_current, dt, params, out_t);
        memcpy(h_current, out_t, hid_dim * sizeof(float));
    }

    if (h_final) {
        memcpy(h_final, h_current, hid_dim * sizeof(float));
    }
}

/* ============================================================================
 * CfC OUTPUT PROJECTION
 * ============================================================================ */

static inline void yinsen_cfc_output(
    const float* h,
    const CfCOutputParams* params,
    float* output
) {
    yinsen_gemm(h, params->W_out, params->b_out,
                output, 1, params->output_dim, params->hidden_dim,
                1.0f, 1.0f);
}

static inline void yinsen_cfc_output_softmax(
    const float* h,
    const CfCOutputParams* params,
    float* probs
) {
    float logits[params->output_dim];
    yinsen_cfc_output(h, params, logits);
    yinsen_softmax(logits, probs, params->output_dim);
}

/* ============================================================================
 * METRICS
 * ============================================================================ */

static inline size_t yinsen_cfc_memory_footprint(const CfCParams* params) {
    const int in_dim = params->input_dim;
    const int hid_dim = params->hidden_dim;
    const int concat_dim = in_dim + hid_dim;

    size_t weights = 0;
    weights += hid_dim * concat_dim * sizeof(float);  /* W_gate */
    weights += hid_dim * sizeof(float);               /* b_gate */
    weights += hid_dim * concat_dim * sizeof(float);  /* W_cand */
    weights += hid_dim * sizeof(float);               /* b_cand */
    weights += (params->tau_shared ? 1 : hid_dim) * sizeof(float);

    return weights;
}

#ifdef __cplusplus
}
#endif

#endif /* YINSEN_CFC_H */
