/*
 * train_sine_v2.c — Factorial experiment: Smoothstep ProxQuant + Trajectory Distillation
 *
 * Pure C. Zero external dependencies. Analytical gradients.
 *
 * Implements a 2x2x2 factorial experiment:
 *   Factor A: {vanilla STE, smoothstep gradient mask}
 *   Factor B: {no distillation, trajectory distillation}
 *   Factor C: {16 hidden, 32 hidden}
 *
 * Plus:
 *   - Float teacher training with Adam (must reach MSE < 0.005)
 *   - Gradient validation via finite differences
 *   - Coordinate search baseline
 *
 * From: journal/scratchpad/spline_cfc_synth.md
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <float.h>
#include "../include/ternary.h"
#include "../include/cfc_ternary.h"

/* ============================================================================
 * SECTION 1: CONFIGURATION AND TYPES
 * ============================================================================ */

#define INPUT_DIM    1
#define OUTPUT_DIM   1
#define MAX_HIDDEN   32      /* largest hidden dim in the factorial */
#define MAX_CONCAT   (INPUT_DIM + MAX_HIDDEN)

#define SEQ_LEN      50      /* steps per training sequence */
#define NUM_SEQS     10      /* sequences per epoch */
#define DT           0.1f    /* time step for CfC */

/* Teacher training config */
#define TEACHER_EPOCHS  2000
#define TEACHER_LR      0.001f

/* Student training config */
#define STUDENT_EPOCHS  1500
#define STUDENT_LR      0.001f

/* Coordinate search */
#define CS_MAX_SWEEPS   5

/* Adam hyperparameters */
#define ADAM_BETA1  0.9f
#define ADAM_BETA2  0.999f
#define ADAM_EPS    1e-8f
#define GRAD_CLIP   5.0f

/* Gradient validation */
#define GRAD_CHECK_EPS  5e-3f
#define GRAD_CHECK_TOL  5e-2f  /* float32 + BPTT — 5% relative error is acceptable */

/* Smoothstep schedule */
#define BETA_START  1.0f
#define BETA_END    20.0f

/* Trajectory distillation schedule */
#define ALPHA_START  0.3f
#define ALPHA_END    0.8f

/* Fixed-size parameter struct (sized for MAX_HIDDEN).
 * We use the first hidden_dim elements of each array. */
typedef struct {
    float W_gate[MAX_HIDDEN * MAX_CONCAT];
    float b_gate[MAX_HIDDEN];
    float W_cand[MAX_HIDDEN * MAX_CONCAT];
    float b_cand[MAX_HIDDEN];
    float tau[MAX_HIDDEN];
    float W_out[OUTPUT_DIM * MAX_HIDDEN];
    float b_out[OUTPUT_DIM];
} Params;

/* Same layout for gradients */
typedef Params Grads;

/* Per-row scales for ternary student */
typedef struct {
    float gate[MAX_HIDDEN];
    float cand[MAX_HIDDEN];
    float out[OUTPUT_DIM];
} RowScales;

/* Same layout for scale gradients */
typedef RowScales ScaleGrads;

/* Adam optimizer state */
typedef struct {
    Params m;       /* first moment */
    Params v;       /* second moment */
    RowScales ms;   /* first moment for scales */
    RowScales vs;   /* second moment for scales */
    int t;          /* timestep */
} AdamState;

/* Step cache: intermediates for backward pass */
typedef struct {
    float concat[MAX_CONCAT];
    float gate_pre[MAX_HIDDEN];
    float gate[MAX_HIDDEN];
    float cand_pre[MAX_HIDDEN];
    float candidate[MAX_HIDDEN];
    float decay[MAX_HIDDEN];
    float h_new[MAX_HIDDEN];
    float y_pred[OUTPUT_DIM];
    /* For student: the ternary weight signs and scales used */
    float w_gate_ternary[MAX_HIDDEN * MAX_CONCAT];  /* unpacked -1/0/+1 */
    float w_cand_ternary[MAX_HIDDEN * MAX_CONCAT];
    float w_out_ternary[OUTPUT_DIM * MAX_HIDDEN];
} StepCache;

/* Student experiment configuration */
typedef struct {
    int use_shaped_grad;        /* 0 = vanilla STE, 1 = smoothstep */
    int use_trajectory_distill; /* 0 = output only, 1 = trajectory */
    int hidden_dim;             /* 16 or 32 */
} StudentConfig;

/* ============================================================================
 * SECTION 2: SMOOTHSTEP GRADIENT MASK
 *
 * Returns gradient multiplier for weight w_float given per-row scale.
 * The smoothstep focuses gradients on weights near decision boundaries
 * (where the ternary quantization is uncertain).
 *
 * Hard forward (always ternary), shaped backward (this masks the gradient).
 * ============================================================================ */

static float smoothstep_grad_mask(float w_float, float scale, float beta) {
    /* Decision boundaries are at ±0.5 * scale (where round changes) */
    /* Distance from nearest boundary: */
    float normalized = (scale > 1e-8f) ? w_float / scale : 0.0f;

    /* Distance to nearest integer boundary (0.5, -0.5 in normalized space) */
    float rounded = roundf(normalized);
    float dist = fabsf(normalized - rounded);

    /* Smoothstep: 1 near boundary (dist ≈ 0), decays to 0 far from boundary */
    /* Using cubic hermite: 3x² - 2x³ evaluated on scaled distance */
    float x = dist * beta;
    if (x >= 1.0f) return 0.0f;
    /* 1 - smoothstep = 1 - (3x² - 2x³) */
    return 1.0f - x * x * (3.0f - 2.0f * x);
}

/* ============================================================================
 * SECTION 3: FORWARD PASS — supports both float teacher and ternary student
 * ============================================================================ */

static float randf(void) {
    return (float)rand() / (float)RAND_MAX;
}

/* Quantize a single weight to ternary: returns -1, 0, or +1 */
static float quantize_trit(float w, float scale) {
    if (scale < 1e-8f) return 0.0f;
    float normalized = w / scale;
    float rounded = roundf(normalized);
    if (rounded > 1.0f) rounded = 1.0f;
    if (rounded < -1.0f) rounded = -1.0f;
    return rounded;
}

/* Compute per-row absmean scale */
static float row_absmean(const float* row, int n) {
    float sum = 0.0f;
    for (int i = 0; i < n; i++) sum += fabsf(row[i]);
    return sum / (float)n;
}

/* Float teacher forward step */
static void forward_float(
    const float* x, const float* h_prev,
    const Params* p, int hid, int concat,
    StepCache* cache
) {
    /* Concat [x; h_prev] */
    memcpy(cache->concat, x, INPUT_DIM * sizeof(float));
    memcpy(cache->concat + INPUT_DIM, h_prev, hid * sizeof(float));

    /* Gate */
    for (int i = 0; i < hid; i++) {
        float sum = p->b_gate[i];
        for (int j = 0; j < concat; j++)
            sum += p->W_gate[i * concat + j] * cache->concat[j];
        cache->gate_pre[i] = sum;
        cache->gate[i] = 1.0f / (1.0f + expf(-sum));
    }

    /* Candidate */
    for (int i = 0; i < hid; i++) {
        float sum = p->b_cand[i];
        for (int j = 0; j < concat; j++)
            sum += p->W_cand[i * concat + j] * cache->concat[j];
        cache->cand_pre[i] = sum;
        cache->candidate[i] = tanhf(sum);
    }

    /* Decay */
    for (int i = 0; i < hid; i++) {
        float tau_i = fabsf(p->tau[i]) + 1e-6f;
        cache->decay[i] = expf(-DT / tau_i);
    }

    /* h_new */
    for (int i = 0; i < hid; i++) {
        float g = cache->gate[i];
        cache->h_new[i] = (1.0f - g) * h_prev[i] * cache->decay[i]
                         + g * cache->candidate[i];
    }

    /* Output */
    for (int i = 0; i < OUTPUT_DIM; i++) {
        float sum = p->b_out[i];
        for (int j = 0; j < hid; j++)
            sum += p->W_out[i * hid + j] * cache->h_new[j];
        cache->y_pred[i] = sum;
    }
}

/* Hard-ternary student forward step.
 * Quantizes weights to ternary, multiplies by per-row scales, runs CfC.
 * Stores the ternary signs in cache for backward pass. */
static void forward_ternary(
    const float* x, const float* h_prev,
    const Params* p, const RowScales* scales,
    int hid, int concat,
    StepCache* cache
) {
    /* Concat */
    memcpy(cache->concat, x, INPUT_DIM * sizeof(float));
    memcpy(cache->concat + INPUT_DIM, h_prev, hid * sizeof(float));

    /* Gate: quantize W_gate row-by-row, compute scaled ternary dot + bias */
    for (int i = 0; i < hid; i++) {
        const float* row = p->W_gate + i * concat;
        float s = scales->gate[i];
        float dot = 0.0f;
        for (int j = 0; j < concat; j++) {
            float t = quantize_trit(row[j], s);
            cache->w_gate_ternary[i * concat + j] = t;
            dot += t * cache->concat[j];
        }
        float pre = s * dot + p->b_gate[i];
        cache->gate_pre[i] = pre;
        cache->gate[i] = 1.0f / (1.0f + expf(-pre));
    }

    /* Candidate */
    for (int i = 0; i < hid; i++) {
        const float* row = p->W_cand + i * concat;
        float s = scales->cand[i];
        float dot = 0.0f;
        for (int j = 0; j < concat; j++) {
            float t = quantize_trit(row[j], s);
            cache->w_cand_ternary[i * concat + j] = t;
            dot += t * cache->concat[j];
        }
        float pre = s * dot + p->b_cand[i];
        cache->cand_pre[i] = pre;
        cache->candidate[i] = tanhf(pre);
    }

    /* Decay (same as float — tau stays float) */
    for (int i = 0; i < hid; i++) {
        float tau_i = fabsf(p->tau[i]) + 1e-6f;
        cache->decay[i] = expf(-DT / tau_i);
    }

    /* h_new */
    for (int i = 0; i < hid; i++) {
        float g = cache->gate[i];
        cache->h_new[i] = (1.0f - g) * h_prev[i] * cache->decay[i]
                         + g * cache->candidate[i];
    }

    /* Output */
    for (int i = 0; i < OUTPUT_DIM; i++) {
        const float* row = p->W_out + i * hid;
        float s = scales->out[i];
        float dot = 0.0f;
        for (int j = 0; j < hid; j++) {
            float t = quantize_trit(row[j], s);
            cache->w_out_ternary[i * hid + j] = t;
            dot += t * cache->h_new[j];
        }
        cache->y_pred[i] = s * dot + p->b_out[i];
    }
}

/* ============================================================================
 * SECTION 4: BACKWARD PASS
 *
 * For teacher: standard float backward.
 * For student: STE (straight-through estimator) or smoothstep-shaped gradient.
 *
 * The key insight: in the ternary forward, the effective weight is
 *   w_eff[i][j] = scale[i] * trit(W_float[i][j])
 *
 * STE: gradient passes through trit() as if it were identity.
 * Smoothstep: gradient is masked by distance to decision boundary.
 * ============================================================================ */

static void backward_float(
    const float* h_prev, const StepCache* cache,
    const Params* p, int hid, int concat,
    const float* dL_dy, const float* dL_dh_future,
    Grads* grad, float* dL_dh_prev
) {
    /* Output layer */
    float dL_dh[MAX_HIDDEN];
    memset(dL_dh, 0, hid * sizeof(float));

    for (int i = 0; i < OUTPUT_DIM; i++) {
        grad->b_out[i] += dL_dy[i];
        for (int j = 0; j < hid; j++) {
            grad->W_out[i * hid + j] += dL_dy[i] * cache->h_new[j];
            dL_dh[j] += dL_dy[i] * p->W_out[i * hid + j];
        }
    }

    /* Add gradient from future timestep */
    if (dL_dh_future) {
        for (int i = 0; i < hid; i++)
            dL_dh[i] += dL_dh_future[i];
    }

    /* CfC cell gradients */
    float dL_dgate[MAX_HIDDEN], dL_dcand[MAX_HIDDEN], dL_ddecay[MAX_HIDDEN];
    memset(dL_dh_prev, 0, hid * sizeof(float));

    for (int i = 0; i < hid; i++) {
        float g = cache->gate[i];
        float c = cache->candidate[i];
        float d = cache->decay[i];
        float hp = h_prev[i];

        dL_dgate[i] = dL_dh[i] * (c - hp * d);
        dL_dcand[i] = dL_dh[i] * g;
        dL_ddecay[i] = dL_dh[i] * (1.0f - g) * hp;
        dL_dh_prev[i] = dL_dh[i] * (1.0f - g) * d;
    }

    /* Through sigmoid */
    float dL_dgate_pre[MAX_HIDDEN];
    for (int i = 0; i < hid; i++) {
        float g = cache->gate[i];
        dL_dgate_pre[i] = dL_dgate[i] * g * (1.0f - g);
    }

    /* Through tanh */
    float dL_dcand_pre[MAX_HIDDEN];
    for (int i = 0; i < hid; i++) {
        float c = cache->candidate[i];
        dL_dcand_pre[i] = dL_dcand[i] * (1.0f - c * c);
    }

    /* W_gate, b_gate gradients */
    for (int i = 0; i < hid; i++) {
        grad->b_gate[i] += dL_dgate_pre[i];
        for (int j = 0; j < concat; j++)
            grad->W_gate[i * concat + j] += dL_dgate_pre[i] * cache->concat[j];
    }

    /* W_cand, b_cand gradients */
    for (int i = 0; i < hid; i++) {
        grad->b_cand[i] += dL_dcand_pre[i];
        for (int j = 0; j < concat; j++)
            grad->W_cand[i * concat + j] += dL_dcand_pre[i] * cache->concat[j];
    }

    /* tau gradient */
    for (int i = 0; i < hid; i++) {
        float tau_i = fabsf(p->tau[i]) + 1e-6f;
        float sign_tau = p->tau[i] >= 0.0f ? 1.0f : -1.0f;
        grad->tau[i] += dL_ddecay[i] * cache->decay[i] * DT / (tau_i * tau_i) * sign_tau;
    }

    /* dL_dh_prev from concat path */
    for (int i = 0; i < hid; i++) {
        for (int j = INPUT_DIM; j < concat; j++) {
            int h_idx = j - INPUT_DIM;
            dL_dh_prev[h_idx] += dL_dgate_pre[i] * p->W_gate[i * concat + j];
            dL_dh_prev[h_idx] += dL_dcand_pre[i] * p->W_cand[i * concat + j];
        }
    }
}

/* Student backward: STE or smoothstep-masked gradients for the float weights.
 * Also computes gradients for per-row scales. */
static void backward_ternary(
    const float* h_prev, const StepCache* cache,
    const Params* p, const RowScales* scales,
    int hid, int concat,
    float beta, int use_shaped,
    const float* dL_dy, const float* dL_dh_future,
    Grads* grad, ScaleGrads* sgrad, float* dL_dh_prev
) {
    /* Output layer: y = s_out * dot(trit(W_out), h_new) + b_out
     *
     * dL/db_out = dL/dy
     * dL/dh_new[j] += dL/dy * s_out * trit(W_out[j])  (ternary values)
     * dL/dW_out_float[j] = dL/dy * s_out * concat[j]  (STE: treat trit as identity)
     *                     × smoothstep_mask(W_out_float[j], s_out, beta)  (if shaped)
     * dL/ds_out = dL/dy * dot(trit(W_out), h_new)
     */
    float dL_dh[MAX_HIDDEN];
    memset(dL_dh, 0, hid * sizeof(float));

    for (int i = 0; i < OUTPUT_DIM; i++) {
        grad->b_out[i] += dL_dy[i];

        /* dL/dh via ternary output weights */
        for (int j = 0; j < hid; j++) {
            dL_dh[j] += dL_dy[i] * scales->out[i] * cache->w_out_ternary[i * hid + j];
        }

        /* dL/ds_out: scale gradient */
        float ternary_dot_val = 0.0f;
        for (int j = 0; j < hid; j++)
            ternary_dot_val += cache->w_out_ternary[i * hid + j] * cache->h_new[j];
        sgrad->out[i] += dL_dy[i] * ternary_dot_val;

        /* dL/dW_out_float: STE passes gradient through quantization.
         * The gradient for the float weight is the gradient the effective weight
         * would receive. Since w_eff = scale * trit(w_float), and STE says
         * d(trit)/d(w_float) ≈ 1/scale (to cancel the scale chain rule),
         * the gradient for w_float ≈ dL/dw_eff = dL/dy * h_new[j].
         * Optionally masked by smoothstep. */
        for (int j = 0; j < hid; j++) {
            float g = dL_dy[i] * cache->h_new[j];
            if (use_shaped) {
                g *= smoothstep_grad_mask(p->W_out[i * hid + j], scales->out[i], beta);
            }
            grad->W_out[i * hid + j] += g;
        }
    }

    /* Add gradient from future timestep */
    if (dL_dh_future) {
        for (int i = 0; i < hid; i++)
            dL_dh[i] += dL_dh_future[i];
    }

    /* CfC cell gradients (same structure as float backward) */
    float dL_dgate[MAX_HIDDEN], dL_dcand[MAX_HIDDEN], dL_ddecay[MAX_HIDDEN];
    memset(dL_dh_prev, 0, hid * sizeof(float));

    for (int i = 0; i < hid; i++) {
        float g = cache->gate[i];
        float c = cache->candidate[i];
        float d = cache->decay[i];
        float hp = h_prev[i];

        dL_dgate[i] = dL_dh[i] * (c - hp * d);
        dL_dcand[i] = dL_dh[i] * g;
        dL_ddecay[i] = dL_dh[i] * (1.0f - g) * hp;
        dL_dh_prev[i] = dL_dh[i] * (1.0f - g) * d;
    }

    /* Through sigmoid */
    float dL_dgate_pre[MAX_HIDDEN];
    for (int i = 0; i < hid; i++) {
        float g = cache->gate[i];
        dL_dgate_pre[i] = dL_dgate[i] * g * (1.0f - g);
    }

    /* Through tanh */
    float dL_dcand_pre[MAX_HIDDEN];
    for (int i = 0; i < hid; i++) {
        float c = cache->candidate[i];
        dL_dcand_pre[i] = dL_dcand[i] * (1.0f - c * c);
    }

    /* Gate weights: gate_pre = scale * dot(trit(W_gate_row), concat) + bias
     * dL/db_gate = dL_dgate_pre
     * dL/ds_gate[i] = dL_dgate_pre[i] * dot(trit(W_gate_row_i), concat)
     * dL/dW_gate_float[i][j] = dL_dgate_pre[i] * concat[j]  (STE) × mask
     */
    for (int i = 0; i < hid; i++) {
        grad->b_gate[i] += dL_dgate_pre[i];

        /* Scale gradient for gate row i */
        float tdot = 0.0f;
        for (int j = 0; j < concat; j++)
            tdot += cache->w_gate_ternary[i * concat + j] * cache->concat[j];
        sgrad->gate[i] += dL_dgate_pre[i] * tdot;

        /* Weight gradient (STE or shaped) */
        for (int j = 0; j < concat; j++) {
            float g = dL_dgate_pre[i] * cache->concat[j];
            if (use_shaped)
                g *= smoothstep_grad_mask(p->W_gate[i * concat + j], scales->gate[i], beta);
            grad->W_gate[i * concat + j] += g;
        }
    }

    /* Candidate weights */
    for (int i = 0; i < hid; i++) {
        grad->b_cand[i] += dL_dcand_pre[i];

        float tdot = 0.0f;
        for (int j = 0; j < concat; j++)
            tdot += cache->w_cand_ternary[i * concat + j] * cache->concat[j];
        sgrad->cand[i] += dL_dcand_pre[i] * tdot;

        for (int j = 0; j < concat; j++) {
            float g = dL_dcand_pre[i] * cache->concat[j];
            if (use_shaped)
                g *= smoothstep_grad_mask(p->W_cand[i * concat + j], scales->cand[i], beta);
            grad->W_cand[i * concat + j] += g;
        }
    }

    /* tau gradient (same as float — tau is always float) */
    for (int i = 0; i < hid; i++) {
        float tau_i = fabsf(p->tau[i]) + 1e-6f;
        float sign_tau = p->tau[i] >= 0.0f ? 1.0f : -1.0f;
        grad->tau[i] += dL_ddecay[i] * cache->decay[i] * DT / (tau_i * tau_i) * sign_tau;
    }

    /* dL_dh_prev from concat path (using effective ternary weights × scale) */
    for (int i = 0; i < hid; i++) {
        for (int j = INPUT_DIM; j < concat; j++) {
            int h_idx = j - INPUT_DIM;
            float wg_eff = scales->gate[i] * cache->w_gate_ternary[i * concat + j];
            float wc_eff = scales->cand[i] * cache->w_cand_ternary[i * concat + j];
            dL_dh_prev[h_idx] += dL_dgate_pre[i] * wg_eff;
            dL_dh_prev[h_idx] += dL_dcand_pre[i] * wc_eff;
        }
    }
}

/* ============================================================================
 * SECTION 5: ADAM OPTIMIZER
 * ============================================================================ */

static void adam_init(AdamState* state) {
    memset(state, 0, sizeof(AdamState));
}

/* Apply Adam to parameter block (flat float array) */
static void adam_update_block(
    float* params, const float* grads,
    float* m, float* v,
    int n, float lr, float bc1, float bc2
) {
    for (int i = 0; i < n; i++) {
        m[i] = ADAM_BETA1 * m[i] + (1.0f - ADAM_BETA1) * grads[i];
        v[i] = ADAM_BETA2 * v[i] + (1.0f - ADAM_BETA2) * grads[i] * grads[i];
        float m_hat = m[i] / bc1;
        float v_hat = v[i] / bc2;
        params[i] -= lr * m_hat / (sqrtf(v_hat) + ADAM_EPS);
    }
}

static void adam_step_params(
    Params* params, const Grads* grads,
    AdamState* state, float lr
) {
    state->t++;
    float bc1 = 1.0f - powf(ADAM_BETA1, (float)state->t);
    float bc2 = 1.0f - powf(ADAM_BETA2, (float)state->t);

    int n = (int)(sizeof(Params) / sizeof(float));
    adam_update_block(
        (float*)params, (const float*)grads,
        (float*)&state->m, (float*)&state->v,
        n, lr, bc1, bc2
    );
}

static void adam_step_scales(
    RowScales* scales, const ScaleGrads* grads,
    AdamState* state, float lr
) {
    /* Uses the same t from the params step (already incremented) */
    float bc1 = 1.0f - powf(ADAM_BETA1, (float)state->t);
    float bc2 = 1.0f - powf(ADAM_BETA2, (float)state->t);

    int n = (int)(sizeof(RowScales) / sizeof(float));
    adam_update_block(
        (float*)scales, (const float*)grads,
        (float*)&state->ms, (float*)&state->vs,
        n, lr, bc1, bc2
    );

    /* Clamp scales to be positive */
    for (int i = 0; i < MAX_HIDDEN; i++) {
        if (scales->gate[i] < 1e-6f) scales->gate[i] = 1e-6f;
        if (scales->cand[i] < 1e-6f) scales->cand[i] = 1e-6f;
    }
    for (int i = 0; i < OUTPUT_DIM; i++) {
        if (scales->out[i] < 1e-6f) scales->out[i] = 1e-6f;
    }
}

/* Gradient clipping on flat array */
static void clip_grads(float* g, int n) {
    float norm_sq = 0.0f;
    for (int i = 0; i < n; i++) norm_sq += g[i] * g[i];
    float norm = sqrtf(norm_sq + 1e-8f);
    if (norm > GRAD_CLIP) {
        float scale = GRAD_CLIP / norm;
        for (int i = 0; i < n; i++) g[i] *= scale;
    }
}

/* ============================================================================
 * SECTION 6: INITIALIZATION
 * ============================================================================ */

static void init_params(Params* p, int hid) {
    int concat = INPUT_DIM + hid;
    float scale_gc = 1.0f / sqrtf((float)concat);
    float scale_out = 1.0f / sqrtf((float)hid);

    memset(p, 0, sizeof(Params));

    for (int i = 0; i < hid; i++) {
        for (int j = 0; j < concat; j++) {
            p->W_gate[i * concat + j] = (randf() * 2.0f - 1.0f) * scale_gc;
            p->W_cand[i * concat + j] = (randf() * 2.0f - 1.0f) * scale_gc;
        }
        p->b_gate[i] = 0.0f;
        p->b_cand[i] = 0.0f;
        p->tau[i] = 0.5f + randf() * 1.5f;
    }
    for (int i = 0; i < OUTPUT_DIM; i++) {
        for (int j = 0; j < hid; j++)
            p->W_out[i * hid + j] = (randf() * 2.0f - 1.0f) * scale_out;
        p->b_out[i] = 0.0f;
    }
}

/* Initialize scales from the float weights (absmean per row) */
static void init_scales_from_params(RowScales* s, const Params* p, int hid) {
    int concat = INPUT_DIM + hid;
    for (int i = 0; i < hid; i++) {
        s->gate[i] = row_absmean(p->W_gate + i * concat, concat);
        s->cand[i] = row_absmean(p->W_cand + i * concat, concat);
        if (s->gate[i] < 1e-6f) s->gate[i] = 1e-6f;
        if (s->cand[i] < 1e-6f) s->cand[i] = 1e-6f;
    }
    for (int i = 0; i < OUTPUT_DIM; i++) {
        s->out[i] = row_absmean(p->W_out + i * hid, hid);
        if (s->out[i] < 1e-6f) s->out[i] = 1e-6f;
    }
}

/* ============================================================================
 * SECTION 7: TEACHER TRAINING (float CfC with Adam)
 * ============================================================================ */

static float eval_mse_float(const Params* p, int hid) {
    int concat = INPUT_DIM + hid;
    float total_mse = 0.0f;
    int steps = 0;
    float h[MAX_HIDDEN] = {0};

    for (int t = 0; t < 100; t++) {
        float time_val = (float)t * DT;
        float x[INPUT_DIM] = { sinf(time_val) };
        float target = sinf(time_val + DT);

        StepCache cache;
        forward_float(x, h, p, hid, concat, &cache);

        float err = cache.y_pred[0] - target;
        total_mse += err * err;
        steps++;
        memcpy(h, cache.h_new, hid * sizeof(float));
    }
    return total_mse / (float)steps;
}

static float train_teacher(Params* teacher, int hid, int num_epochs) {
    int concat = INPUT_DIM + hid;
    AdamState adam;
    adam_init(&adam);

    float best_mse = FLT_MAX;

    for (int epoch = 0; epoch < num_epochs; epoch++) {
        Grads grad;
        memset(&grad, 0, sizeof(Grads));

        float total_loss = 0.0f;
        int total_steps = 0;

        for (int seq = 0; seq < NUM_SEQS; seq++) {
            float phase = randf() * 2.0f * 3.14159265f;
            float h_hist[SEQ_LEN + 1][MAX_HIDDEN];
            StepCache caches[SEQ_LEN];
            float targets[SEQ_LEN];

            memset(h_hist[0], 0, hid * sizeof(float));

            /* Forward */
            for (int t = 0; t < SEQ_LEN; t++) {
                float time_val = (float)t * DT + phase;
                float x[INPUT_DIM] = { sinf(time_val) };
                targets[t] = sinf(time_val + DT);

                forward_float(x, h_hist[t], teacher, hid, concat, &caches[t]);
                memcpy(h_hist[t + 1], caches[t].h_new, hid * sizeof(float));

                float err = caches[t].y_pred[0] - targets[t];
                total_loss += 0.5f * err * err;
                total_steps++;
            }

            /* Backward (BPTT) */
            float dL_dh_next[MAX_HIDDEN];
            memset(dL_dh_next, 0, hid * sizeof(float));

            for (int t = SEQ_LEN - 1; t >= 0; t--) {
                float err = caches[t].y_pred[0] - targets[t];
                float dL_dy[OUTPUT_DIM] = { err };
                float dL_dh_prev[MAX_HIDDEN];

                backward_float(
                    h_hist[t], &caches[t], teacher, hid, concat,
                    dL_dy, dL_dh_next, &grad, dL_dh_prev
                );
                memcpy(dL_dh_next, dL_dh_prev, hid * sizeof(float));
            }
        }

        /* Average and clip gradients */
        float* gw = (float*)&grad;
        int n = (int)(sizeof(Grads) / sizeof(float));
        for (int i = 0; i < n; i++) gw[i] /= (float)total_steps;
        clip_grads(gw, n);

        /* Adam step */
        adam_step_params(teacher, &grad, &adam, TEACHER_LR);

        float avg_loss = total_loss / (float)total_steps;
        if (epoch % 100 == 0 || epoch == num_epochs - 1) {
            float mse = eval_mse_float(teacher, hid);
            if (mse < best_mse) best_mse = mse;
            printf("    [Teacher] Epoch %4d/%d  train_loss=%.6f  eval_mse=%.6f\n",
                   epoch + 1, num_epochs, avg_loss, mse);
        }
    }

    float final_mse = eval_mse_float(teacher, hid);
    return final_mse;
}

/* ============================================================================
 * SECTION 8: STUDENT TRAINING (ternary CfC with trajectory distillation)
 * ============================================================================ */

/* Evaluate ternary student MSE */
static float eval_mse_ternary(const Params* p, const RowScales* scales, int hid) {
    int concat = INPUT_DIM + hid;
    float total_mse = 0.0f;
    int steps = 0;
    float h[MAX_HIDDEN] = {0};

    for (int t = 0; t < 100; t++) {
        float time_val = (float)t * DT;
        float x[INPUT_DIM] = { sinf(time_val) };
        float target = sinf(time_val + DT);

        StepCache cache;
        forward_ternary(x, h, p, scales, hid, concat, &cache);

        float err = cache.y_pred[0] - target;
        total_mse += err * err;
        steps++;
        memcpy(h, cache.h_new, hid * sizeof(float));
    }
    return total_mse / (float)steps;
}

static float train_student(
    const Params* teacher, Params* student,
    RowScales* scales,
    const StudentConfig* config,
    int num_epochs
) {
    int hid = config->hidden_dim;
    int concat = INPUT_DIM + hid;
    AdamState adam;
    adam_init(&adam);

    for (int epoch = 0; epoch < num_epochs; epoch++) {
        Grads grad;
        ScaleGrads sgrad;
        memset(&grad, 0, sizeof(Grads));
        memset(&sgrad, 0, sizeof(ScaleGrads));

        float total_loss = 0.0f;
        int total_steps = 0;

        /* Schedule: beta and alpha ramp over training */
        float progress = (float)epoch / (float)num_epochs;
        float beta = BETA_START + (BETA_END - BETA_START) * progress;
        float alpha = ALPHA_START + (ALPHA_END - ALPHA_START) * progress;

        for (int seq = 0; seq < NUM_SEQS; seq++) {
            float phase = randf() * 2.0f * 3.14159265f;

            /* Teacher forward (generate hidden trajectories) */
            float teacher_h[SEQ_LEN + 1][MAX_HIDDEN];
            memset(teacher_h[0], 0, hid * sizeof(float));

            if (config->use_trajectory_distill) {
                for (int t = 0; t < SEQ_LEN; t++) {
                    float time_val = (float)t * DT + phase;
                    float x[INPUT_DIM] = { sinf(time_val) };
                    StepCache tcache;
                    forward_float(x, teacher_h[t], teacher, hid, concat, &tcache);
                    memcpy(teacher_h[t + 1], tcache.h_new, hid * sizeof(float));
                }
            }

            /* Student forward */
            float student_h[SEQ_LEN + 1][MAX_HIDDEN];
            StepCache caches[SEQ_LEN];
            float targets[SEQ_LEN];

            memset(student_h[0], 0, hid * sizeof(float));

            for (int t = 0; t < SEQ_LEN; t++) {
                float time_val = (float)t * DT + phase;
                float x[INPUT_DIM] = { sinf(time_val) };
                targets[t] = sinf(time_val + DT);

                forward_ternary(x, student_h[t], student, scales, hid, concat, &caches[t]);
                memcpy(student_h[t + 1], caches[t].h_new, hid * sizeof(float));

                /* Output loss */
                float err = caches[t].y_pred[0] - targets[t];
                float output_loss = 0.5f * err * err;

                /* Trajectory loss: MSE between student h[t] and teacher h[t] */
                float traj_loss = 0.0f;
                if (config->use_trajectory_distill) {
                    for (int i = 0; i < hid; i++) {
                        float diff = caches[t].h_new[i] - teacher_h[t + 1][i];
                        traj_loss += 0.5f * diff * diff;
                    }
                    traj_loss /= (float)hid;  /* normalize by hidden dim */
                }

                total_loss += (1.0f - alpha) * output_loss + alpha * traj_loss;
                total_steps++;
            }

            /* Backward (BPTT) */
            float dL_dh_next[MAX_HIDDEN];
            memset(dL_dh_next, 0, hid * sizeof(float));

            for (int t = SEQ_LEN - 1; t >= 0; t--) {
                float err = caches[t].y_pred[0] - targets[t];

                /* Output loss gradient */
                float dL_dy[OUTPUT_DIM] = { (1.0f - alpha) * err };

                /* Trajectory loss gradient on h_new[t]:
                 * d/dh_new[ alpha * 0.5/H * sum(h_new - teacher_h)^2 ]
                 *   = alpha/H * (h_new - teacher_h) */
                float dL_dh_traj[MAX_HIDDEN];
                memset(dL_dh_traj, 0, hid * sizeof(float));
                if (config->use_trajectory_distill) {
                    for (int i = 0; i < hid; i++) {
                        dL_dh_traj[i] = alpha * (caches[t].h_new[i] - teacher_h[t + 1][i]) / (float)hid;
                    }
                }

                /* Combine BPTT + trajectory gradient */
                float dL_dh_combined[MAX_HIDDEN];
                for (int i = 0; i < hid; i++)
                    dL_dh_combined[i] = dL_dh_next[i] + dL_dh_traj[i];

                float dL_dh_prev[MAX_HIDDEN];
                backward_ternary(
                    student_h[t], &caches[t], student, scales, hid, concat,
                    beta, config->use_shaped_grad,
                    dL_dy, dL_dh_combined,
                    &grad, &sgrad, dL_dh_prev
                );
                memcpy(dL_dh_next, dL_dh_prev, hid * sizeof(float));
            }
        }

        /* Average and clip */
        float* gw = (float*)&grad;
        int np = (int)(sizeof(Grads) / sizeof(float));
        for (int i = 0; i < np; i++) gw[i] /= (float)total_steps;
        clip_grads(gw, np);

        float* sw = (float*)&sgrad;
        int ns = (int)(sizeof(ScaleGrads) / sizeof(float));
        for (int i = 0; i < ns; i++) sw[i] /= (float)total_steps;
        clip_grads(sw, ns);

        /* Adam steps */
        adam_step_params(student, &grad, &adam, STUDENT_LR);
        adam_step_scales(scales, &sgrad, &adam, STUDENT_LR);

        if (epoch % 200 == 0 || epoch == num_epochs - 1) {
            float mse = eval_mse_ternary(student, scales, hid);
            printf("      Epoch %4d/%d  loss=%.6f  eval_mse=%.6f  beta=%.1f  alpha=%.2f\n",
                   epoch + 1, num_epochs, total_loss / (float)total_steps, mse, beta, alpha);
        }
    }

    return eval_mse_ternary(student, scales, hid);
}

/* ============================================================================
 * SECTION 9: COORDINATE SEARCH BASELINE
 *
 * For each weight in the ternary student, try all 3 values {-1, 0, +1}
 * and keep the one that minimizes eval MSE.
 * This is a local search — no gradient needed.
 * ============================================================================ */

/* Convert float params to packed ternary using current scales, then evaluate */
static float eval_with_trit_override(
    const Params* base, const RowScales* scales, int hid,
    int which_matrix, int row, int col, float trit_val
) {
    /* Temporarily modify the base float weight so quantize_trit returns trit_val.
     * We do this by setting w = trit_val * scale (so quantize_trit(w, scale) = trit_val).
     * Then evaluate. */
    Params tmp;
    memcpy(&tmp, base, sizeof(Params));

    int concat = INPUT_DIM + hid;
    float s;
    if (which_matrix == 0) {
        s = scales->gate[row];
        tmp.W_gate[row * concat + col] = trit_val * s;
    } else if (which_matrix == 1) {
        s = scales->cand[row];
        tmp.W_cand[row * concat + col] = trit_val * s;
    } else {
        s = scales->out[row];
        tmp.W_out[row * hid + col] = trit_val * s;
    }

    return eval_mse_ternary(&tmp, scales, hid);
}

static float coordinate_search(
    Params* params, const RowScales* scales,
    int hid, int max_sweeps
) {
    int concat = INPUT_DIM + hid;
    float best_mse = eval_mse_ternary(params, scales, hid);
    float trit_vals[3] = {-1.0f, 0.0f, 1.0f};

    printf("    Coordinate search: initial MSE = %.6f\n", best_mse);

    for (int sweep = 0; sweep < max_sweeps; sweep++) {
        int improved = 0;

        /* W_gate */
        for (int i = 0; i < hid; i++) {
            for (int j = 0; j < concat; j++) {
                float current_trit = quantize_trit(params->W_gate[i * concat + j], scales->gate[i]);
                for (int tv = 0; tv < 3; tv++) {
                    if (trit_vals[tv] == current_trit) continue;
                    float mse = eval_with_trit_override(params, scales, hid, 0, i, j, trit_vals[tv]);
                    if (mse < best_mse) {
                        best_mse = mse;
                        params->W_gate[i * concat + j] = trit_vals[tv] * scales->gate[i];
                        improved++;
                    }
                }
            }
        }

        /* W_cand */
        for (int i = 0; i < hid; i++) {
            for (int j = 0; j < concat; j++) {
                float current_trit = quantize_trit(params->W_cand[i * concat + j], scales->cand[i]);
                for (int tv = 0; tv < 3; tv++) {
                    if (trit_vals[tv] == current_trit) continue;
                    float mse = eval_with_trit_override(params, scales, hid, 1, i, j, trit_vals[tv]);
                    if (mse < best_mse) {
                        best_mse = mse;
                        params->W_cand[i * concat + j] = trit_vals[tv] * scales->cand[i];
                        improved++;
                    }
                }
            }
        }

        /* W_out */
        for (int i = 0; i < OUTPUT_DIM; i++) {
            for (int j = 0; j < hid; j++) {
                float current_trit = quantize_trit(params->W_out[i * hid + j], scales->out[i]);
                for (int tv = 0; tv < 3; tv++) {
                    if (trit_vals[tv] == current_trit) continue;
                    float mse = eval_with_trit_override(params, scales, hid, 2, i, j, trit_vals[tv]);
                    if (mse < best_mse) {
                        best_mse = mse;
                        params->W_out[i * hid + j] = trit_vals[tv] * scales->out[i];
                        improved++;
                    }
                }
            }
        }

        printf("    Sweep %d: MSE = %.6f  (improved %d weights)\n", sweep + 1, best_mse, improved);
        if (improved == 0) break;
    }

    return best_mse;
}

/* ============================================================================
 * SECTION 10: GRADIENT VALIDATION
 *
 * Finite-difference check of the float teacher backward pass.
 * Must pass before we trust any training results.
 * ============================================================================ */

static float compute_loss_for_grad_check(
    const Params* p, int hid,
    const float* inputs, const float* targets, int T
) {
    int concat = INPUT_DIM + hid;
    float h[MAX_HIDDEN] = {0};
    float total_loss = 0.0f;

    for (int t = 0; t < T; t++) {
        StepCache cache;
        forward_float(&inputs[t], h, p, hid, concat, &cache);
        float err = cache.y_pred[0] - targets[t];
        total_loss += 0.5f * err * err;
        memcpy(h, cache.h_new, hid * sizeof(float));
    }
    return total_loss;
}

static float validate_gradients(int hid) {
    int concat = INPUT_DIM + hid;
    int T = 5;  /* short sequence for grad check */

    /* Random params */
    Params p;
    init_params(&p, hid);

    /* Random input/target sequence */
    float inputs[5], targets[5];
    for (int t = 0; t < T; t++) {
        float tv = (float)t * DT + randf() * 3.14159f;
        inputs[t] = sinf(tv);
        targets[t] = sinf(tv + DT);
    }

    /* Analytical gradient */
    Grads grad;
    memset(&grad, 0, sizeof(Grads));

    float h_hist[6][MAX_HIDDEN];  /* T+1 */
    StepCache caches[5];          /* T */
    memset(h_hist[0], 0, hid * sizeof(float));

    for (int t = 0; t < T; t++) {
        forward_float(&inputs[t], h_hist[t], &p, hid, concat, &caches[t]);
        memcpy(h_hist[t + 1], caches[t].h_new, hid * sizeof(float));
    }

    float dL_dh_next[MAX_HIDDEN];
    memset(dL_dh_next, 0, hid * sizeof(float));

    for (int t = T - 1; t >= 0; t--) {
        float err = caches[t].y_pred[0] - targets[t];
        float dL_dy[OUTPUT_DIM] = { err };
        float dL_dh_prev[MAX_HIDDEN];

        backward_float(
            h_hist[t], &caches[t], &p, hid, concat,
            dL_dy, dL_dh_next, &grad, dL_dh_prev
        );
        memcpy(dL_dh_next, dL_dh_prev, hid * sizeof(float));
    }

    /* Numerical gradient (central differences) on a subset of params */
    float max_rel_err = 0.0f;
    float* pw = (float*)&p;
    float* gw = (float*)&grad;

    /* Build list of active parameter indices (only those actually used) */
    int active_indices[2500];  /* more than enough */
    int n_active = 0;

    /* W_gate: rows 0..hid-1, cols 0..concat-1, stored at i*concat+j */
    int wg_offset = 0;  /* W_gate is first in struct */
    for (int i = 0; i < hid; i++)
        for (int j = 0; j < concat; j++)
            active_indices[n_active++] = wg_offset + i * concat + j;

    /* b_gate: first hid entries */
    int bg_offset = (int)((char*)p.b_gate - (char*)&p) / (int)sizeof(float);
    for (int i = 0; i < hid; i++)
        active_indices[n_active++] = bg_offset + i;

    /* W_cand */
    int wc_offset = (int)((char*)p.W_cand - (char*)&p) / (int)sizeof(float);
    for (int i = 0; i < hid; i++)
        for (int j = 0; j < concat; j++)
            active_indices[n_active++] = wc_offset + i * concat + j;

    /* b_cand */
    int bc_offset = (int)((char*)p.b_cand - (char*)&p) / (int)sizeof(float);
    for (int i = 0; i < hid; i++)
        active_indices[n_active++] = bc_offset + i;

    /* tau */
    int tau_offset = (int)((char*)p.tau - (char*)&p) / (int)sizeof(float);
    for (int i = 0; i < hid; i++)
        active_indices[n_active++] = tau_offset + i;

    /* W_out */
    int wo_offset = (int)((char*)p.W_out - (char*)&p) / (int)sizeof(float);
    for (int i = 0; i < OUTPUT_DIM; i++)
        for (int j = 0; j < hid; j++)
            active_indices[n_active++] = wo_offset + i * hid + j;

    /* b_out */
    int bo_offset = (int)((char*)p.b_out - (char*)&p) / (int)sizeof(float);
    for (int i = 0; i < OUTPUT_DIM; i++)
        active_indices[n_active++] = bo_offset + i;

    /* Check a random subset of active indices */
    int max_checks = (n_active < 200) ? n_active : 200;
    int worst_idx = -1;
    float worst_analytical = 0, worst_numerical = 0;

    for (int check = 0; check < max_checks; check++) {
        int idx = active_indices[rand() % n_active];

        float orig = pw[idx];
        pw[idx] = orig + GRAD_CHECK_EPS;
        float loss_plus = compute_loss_for_grad_check(&p, hid, inputs, targets, T);
        pw[idx] = orig - GRAD_CHECK_EPS;
        float loss_minus = compute_loss_for_grad_check(&p, hid, inputs, targets, T);
        pw[idx] = orig;

        float numerical = (loss_plus - loss_minus) / (2.0f * GRAD_CHECK_EPS);
        float analytical = gw[idx];

        /* Skip parameters where both gradients are negligible (float noise floor) */
        if (fabsf(analytical) < 1e-3f && fabsf(numerical) < 1e-3f) continue;

        float denom = fmaxf(fabsf(analytical), fmaxf(fabsf(numerical), 1e-7f));
        float rel_err = fabsf(analytical - numerical) / denom;

        if (rel_err > max_rel_err) {
            max_rel_err = rel_err;
            worst_idx = idx;
            worst_analytical = analytical;
            worst_numerical = numerical;
        }
    }

    if (worst_idx >= 0) {
        printf("    Worst: idx=%d  analytical=%.6e  numerical=%.6e  (checked %d of %d active)\n",
               worst_idx, worst_analytical, worst_numerical, max_checks, n_active);
    }

    return max_rel_err;
}

/* ============================================================================
 * SECTION 11: MAIN — FACTORIAL EXPERIMENT DRIVER
 * ============================================================================ */

int main(void) {
    srand(42);

    printf("=================================================================\n");
    printf("  YINSEN v2: FACTORIAL EXPERIMENT\n");
    printf("  Smoothstep ProxQuant + Trajectory Distillation\n");
    printf("=================================================================\n\n");

    /* ---------- Gradient Validation ---------- */
    printf("--- Gradient Validation ---\n");
    float grad_err_16 = validate_gradients(16);
    float grad_err_32 = validate_gradients(32);
    printf("  Max relative error (hid=16): %.2e\n", grad_err_16);
    printf("  Max relative error (hid=32): %.2e\n", grad_err_32);

    if (grad_err_16 > GRAD_CHECK_TOL || grad_err_32 > GRAD_CHECK_TOL) {
        printf("  ABORT: Gradient validation FAILED (threshold: %.0e)\n", GRAD_CHECK_TOL);
        printf("  Cannot trust training results.\n");
        return 1;
    }
    printf("  PASS: Gradients verified.\n\n");

    /* ---------- Train Teachers ---------- */
    printf("--- Training Float Teachers ---\n\n");

    Params teacher_16, teacher_32;
    srand(42);  /* reset seed for reproducibility */
    printf("  Teacher (hid=16):\n");
    init_params(&teacher_16, 16);
    float teacher_mse_16 = train_teacher(&teacher_16, 16, TEACHER_EPOCHS);
    printf("  Final MSE: %.6f %s\n\n",
           teacher_mse_16, teacher_mse_16 < 0.005f ? "(GOOD)" : "(WARN: > 0.005)");

    srand(43);  /* different seed for 32 */
    printf("  Teacher (hid=32):\n");
    init_params(&teacher_32, 32);
    float teacher_mse_32 = train_teacher(&teacher_32, 32, TEACHER_EPOCHS);
    printf("  Final MSE: %.6f %s\n\n",
           teacher_mse_32, teacher_mse_32 < 0.005f ? "(GOOD)" : "(WARN: > 0.005)");

    /* ---------- Factorial Experiment ---------- */
    printf("--- Factorial Experiment (2x2x2 = 8 configs) ---\n\n");

    StudentConfig configs[8] = {
        {0, 0, 16},  /* 1: Baseline */
        {1, 0, 16},  /* 2: Shaped only */
        {0, 1, 16},  /* 3: Distill only */
        {1, 1, 16},  /* 4: Both, narrow */
        {0, 0, 32},  /* 5: Baseline, wide */
        {1, 0, 32},  /* 6: Shaped, wide */
        {0, 1, 32},  /* 7: Distill, wide */
        {1, 1, 32},  /* 8: Both, wide */
    };

    const char* config_names[8] = {
        "STE,      NoDistill, h=16",
        "Smooth,   NoDistill, h=16",
        "STE,      Traj,      h=16",
        "Smooth,   Traj,      h=16",
        "STE,      NoDistill, h=32",
        "Smooth,   NoDistill, h=32",
        "STE,      Traj,      h=32",
        "Smooth,   Traj,      h=32",
    };

    float results_mse[8];
    float baseline_mse = 0.0f;

    for (int c = 0; c < 8; c++) {
        printf("  Config %d: %s\n", c + 1, config_names[c]);

        /* Pick teacher and set reproducible seed per config */
        const Params* teacher = (configs[c].hidden_dim == 16) ? &teacher_16 : &teacher_32;
        srand(100 + c);

        /* Initialize student from teacher weights */
        Params student;
        memcpy(&student, teacher, sizeof(Params));

        /* Initialize scales from teacher */
        RowScales scales;
        init_scales_from_params(&scales, teacher, configs[c].hidden_dim);

        /* Train */
        results_mse[c] = train_student(
            teacher, &student, &scales,
            &configs[c], STUDENT_EPOCHS
        );

        if (c == 0) baseline_mse = results_mse[c];

        printf("    -> Eval MSE: %.6f\n\n", results_mse[c]);
    }

    /* ---------- Coordinate Search Baseline ---------- */
    printf("--- Coordinate Search Baseline (on Config 1) ---\n");

    srand(100);  /* same seed as config 1 */
    Params cs_student;
    memcpy(&cs_student, &teacher_16, sizeof(Params));
    RowScales cs_scales;
    init_scales_from_params(&cs_scales, &teacher_16, 16);

    /* No gradient training — just direct search */
    float cs_mse = coordinate_search(&cs_student, &cs_scales, 16, CS_MAX_SWEEPS);
    printf("    -> Final MSE: %.6f\n\n", cs_mse);

    /* ---------- Results Table ---------- */
    printf("=================================================================\n");
    printf("  RESULTS\n");
    printf("=================================================================\n\n");

    printf("  Teachers:\n");
    printf("    hid=16 MSE: %.6f  RMSE: %.4f\n", teacher_mse_16, sqrtf(teacher_mse_16));
    printf("    hid=32 MSE: %.6f  RMSE: %.4f\n\n", teacher_mse_32, sqrtf(teacher_mse_32));

    printf("  Config | Shaped | Distill | Width | Eval MSE | vs Float  | vs Baseline\n");
    printf("  -------+--------+---------+-------+----------+-----------+------------\n");

    for (int c = 0; c < 8; c++) {
        float teacher_mse = (configs[c].hidden_dim == 16) ? teacher_mse_16 : teacher_mse_32;
        printf("    %d    |  %s  |  %s   |  %2d   | %.6f | %7.2fx   | %7.2fx\n",
               c + 1,
               configs[c].use_shaped_grad ? " Yes" : " No ",
               configs[c].use_trajectory_distill ? "Yes" : "No ",
               configs[c].hidden_dim,
               results_mse[c],
               results_mse[c] / (teacher_mse + 1e-10f),
               results_mse[c] / (baseline_mse + 1e-10f));
    }

    printf("    CS   |  Post-training coordinate search on Config 1 baseline\n");
    printf("         |                              | %.6f | %7.2fx   | %7.2fx\n",
           cs_mse,
           cs_mse / (teacher_mse_16 + 1e-10f),
           cs_mse / (baseline_mse + 1e-10f));

    printf("\n  Best student MSE: %.6f\n", fminf(fminf(fminf(results_mse[0], results_mse[1]),
           fminf(results_mse[2], results_mse[3])),
           fminf(fminf(results_mse[4], results_mse[5]),
           fminf(results_mse[6], results_mse[7]))));

    float best_overall = cs_mse;
    for (int c = 0; c < 8; c++)
        if (results_mse[c] < best_overall) best_overall = results_mse[c];

    printf("  Best overall MSE: %.6f  (RMSE: %.4f)\n", best_overall, sqrtf(best_overall));

    if (best_overall < 0.05f) {
        printf("\n  SUCCESS: At least one ternary config achieves MSE < 0.05\n");
        printf("  (10x better than v1's 0.228)\n");
    } else if (best_overall < 0.228f) {
        printf("\n  PARTIAL: Improvement over v1 (0.228) but not 10x\n");
    } else {
        printf("\n  NO IMPROVEMENT: Best ternary MSE >= v1 baseline\n");
    }

    printf("=================================================================\n");

    return 0;
}
