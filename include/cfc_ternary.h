/*
 * YINSEN CfC Ternary - Gated Recurrence with Ternary Weights
 *
 * Same update rule as CfC, but weights are ternary {-1, 0, +1}.
 * Linear transformations use add/subtract only.
 * Activations (sigmoid, tanh, exp) still use float.
 *
 * Update rule:
 *   h(t) = (1 - gate) * h_prev * decay + gate * candidate
 *
 * Where:
 *   gate      = sigmoid(W_gate @ [x; h_prev] + b_gate)
 *   candidate = tanh(W_cand @ [x; h_prev] + b_cand)
 *   decay     = exp(-dt / tau)
 *
 * Ternary advantages:
 *   - W_gate, W_cand, W_out are ternary (no multiply in linear)
 *   - 16x smaller weight storage
 *   - Integer-like determinism in linear paths
 *
 * Verification status: See test_cfc_ternary.c
 */

#ifndef YINSEN_CFC_TERNARY_H
#define YINSEN_CFC_TERNARY_H

#include "ternary.h"
#include "onnx_shapes.h"
#include <string.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ============================================================================
 * CfC TERNARY PARAMETERS
 * ============================================================================ */

typedef struct {
    int input_dim;
    int hidden_dim;

    const uint8_t* W_gate;  /* Packed ternary [hidden_dim, input_dim + hidden_dim] */
    const float* b_gate;    /* Bias still float [hidden_dim] */
    const uint8_t* W_cand;  /* Packed ternary [hidden_dim, input_dim + hidden_dim] */
    const float* b_cand;    /* [hidden_dim] */
    const float* tau;       /* Time constant [hidden_dim] or [1] */
    int tau_shared;
} CfCTernaryParams;

typedef struct {
    int hidden_dim;
    int output_dim;
    const uint8_t* W_out;   /* Packed ternary [hidden_dim, output_dim] */
    const float* b_out;     /* [output_dim] */
} CfCTernaryOutputParams;

/* ============================================================================
 * CfC TERNARY CELL - Single step
 * ============================================================================ */

static inline void yinsen_cfc_ternary_cell(
    const float* x,
    const float* h_prev,
    float dt,
    const CfCTernaryParams* params,
    float* h_new
) {
    const int in_dim = params->input_dim;
    const int hid_dim = params->hidden_dim;
    const int concat_dim = in_dim + hid_dim;
    const int bytes_per_row = (concat_dim + 3) / 4;

    /* Stack allocation */
    float concat[concat_dim];
    float gate_pre[hid_dim];
    float gate[hid_dim];
    float cand_pre[hid_dim];
    float candidate[hid_dim];

    /* Step 1: Concatenate [x; h_prev] */
    memcpy(concat, x, in_dim * sizeof(float));
    memcpy(concat + in_dim, h_prev, hid_dim * sizeof(float));

    /* Step 2: Gate = sigmoid(W_gate @ concat + b_gate)
     * W_gate is ternary - no multiply! */
    for (int i = 0; i < hid_dim; i++) {
        gate_pre[i] = ternary_dot(
            params->W_gate + i * bytes_per_row,
            concat,
            concat_dim
        ) + params->b_gate[i];
        gate[i] = yinsen_sigmoid(gate_pre[i]);
    }

    /* Step 3: Candidate = tanh(W_cand @ concat + b_cand)
     * W_cand is ternary - no multiply! */
    for (int i = 0; i < hid_dim; i++) {
        cand_pre[i] = ternary_dot(
            params->W_cand + i * bytes_per_row,
            concat,
            concat_dim
        ) + params->b_cand[i];
        candidate[i] = yinsen_tanh(cand_pre[i]);
    }

    /* Step 4: Decay = exp(-dt / tau)
     * Still uses float exp */
    float decay[hid_dim];
    if (params->tau_shared) {
        float decay_scalar = expf(-dt / params->tau[0]);
        for (int i = 0; i < hid_dim; i++) {
            decay[i] = decay_scalar;
        }
    } else {
        for (int i = 0; i < hid_dim; i++) {
            decay[i] = expf(-dt / params->tau[i]);
        }
    }

    /* Step 5: h_new = (1 - gate) * h_prev * decay + gate * candidate
     * This part still uses float multiply (unavoidable for gating) */
    for (int i = 0; i < hid_dim; i++) {
        float retention = (1.0f - gate[i]) * h_prev[i] * decay[i];
        float update = gate[i] * candidate[i];
        h_new[i] = retention + update;
    }
}

/* ============================================================================
 * CfC TERNARY OUTPUT PROJECTION
 * ============================================================================ */

static inline void yinsen_cfc_ternary_output(
    const float* h,
    const CfCTernaryOutputParams* params,
    float* output
) {
    int bytes_per_row = (params->hidden_dim + 3) / 4;

    for (int i = 0; i < params->output_dim; i++) {
        output[i] = ternary_dot(
            params->W_out + i * bytes_per_row,
            h,
            params->hidden_dim
        ) + params->b_out[i];
    }
}

static inline void yinsen_cfc_ternary_output_softmax(
    const float* h,
    const CfCTernaryOutputParams* params,
    float* probs
) {
    float logits[params->output_dim];
    yinsen_cfc_ternary_output(h, params, logits);
    yinsen_softmax(logits, probs, params->output_dim);
}

/* ============================================================================
 * SEQUENCE PROCESSING
 * ============================================================================ */

static inline void yinsen_cfc_ternary_forward(
    const float* inputs,
    int seq_len,
    float dt,
    const CfCTernaryParams* params,
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

        yinsen_cfc_ternary_cell(x_t, h_current, dt, params, out_t);
        memcpy(h_current, out_t, hid_dim * sizeof(float));
    }

    if (h_final) {
        memcpy(h_final, h_current, hid_dim * sizeof(float));
    }
}

/* ============================================================================
 * MEMORY FOOTPRINT
 * ============================================================================ */

static inline size_t yinsen_cfc_ternary_weight_bytes(const CfCTernaryParams* params) {
    const int in_dim = params->input_dim;
    const int hid_dim = params->hidden_dim;
    const int concat_dim = in_dim + hid_dim;

    size_t bytes = 0;
    bytes += ternary_matrix_bytes(hid_dim, concat_dim);  /* W_gate */
    bytes += hid_dim * sizeof(float);                     /* b_gate */
    bytes += ternary_matrix_bytes(hid_dim, concat_dim);  /* W_cand */
    bytes += hid_dim * sizeof(float);                     /* b_cand */
    bytes += (params->tau_shared ? 1 : hid_dim) * sizeof(float);

    return bytes;
}

static inline void yinsen_cfc_ternary_memory_comparison(
    const CfCTernaryParams* params,
    size_t* ternary_bytes,
    size_t* float_bytes,
    float* ratio
) {
    const int in_dim = params->input_dim;
    const int hid_dim = params->hidden_dim;
    const int concat_dim = in_dim + hid_dim;

    /* Ternary version */
    *ternary_bytes = yinsen_cfc_ternary_weight_bytes(params);

    /* Float version (what standard CfC uses) */
    *float_bytes = 0;
    *float_bytes += hid_dim * concat_dim * sizeof(float);  /* W_gate */
    *float_bytes += hid_dim * sizeof(float);                /* b_gate */
    *float_bytes += hid_dim * concat_dim * sizeof(float);  /* W_cand */
    *float_bytes += hid_dim * sizeof(float);                /* b_cand */
    *float_bytes += (params->tau_shared ? 1 : hid_dim) * sizeof(float);

    *ratio = (float)(*float_bytes) / (float)(*ternary_bytes);
}

#ifdef __cplusplus
}
#endif

#endif /* YINSEN_CFC_TERNARY_H */
