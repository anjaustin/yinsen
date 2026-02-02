/*
 * train_sine.c — Train a CfC network on sine wave prediction, then
 *                quantize to ternary and run inference through Yinsen.
 *
 * Pure C. Zero external dependencies. Analytical gradients.
 *
 * Task: Given sin(t), predict sin(t + dt).
 *       1 input, HIDDEN_DIM hidden, 1 output.
 *
 * Pipeline:
 *   1. Train float CfC with SGD (hand-rolled backward pass)
 *   2. Quantize weights to ternary via absmean
 *   3. Run inference with yinsen_cfc_ternary_cell
 *   4. Compare float vs ternary MSE
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <time.h>
#include "../include/ternary.h"
#include "../include/cfc_ternary.h"

/* ============================================================================
 * CONFIGURATION
 * ============================================================================ */

#define INPUT_DIM    1
#define HIDDEN_DIM   16
#define OUTPUT_DIM   1
#define CONCAT_DIM   (INPUT_DIM + HIDDEN_DIM)

#define SEQ_LEN      50    /* steps per training sequence */
#define NUM_EPOCHS   1000
#define LEARNING_RATE 0.005f
#define DT           0.1f  /* time step */

/* ============================================================================
 * TRAINABLE PARAMETERS (float, mutable)
 * ============================================================================ */

typedef struct {
    float W_gate[HIDDEN_DIM * CONCAT_DIM];
    float b_gate[HIDDEN_DIM];
    float W_cand[HIDDEN_DIM * CONCAT_DIM];
    float b_cand[HIDDEN_DIM];
    float tau[HIDDEN_DIM];

    float W_out[OUTPUT_DIM * HIDDEN_DIM];
    float b_out[OUTPUT_DIM];
} TrainParams;

/* Same layout for gradient accumulation */
typedef TrainParams GradParams;

/* ============================================================================
 * FORWARD PASS — single CfC step + output projection
 *
 * Stores intermediates needed for backward pass.
 * ============================================================================ */

typedef struct {
    float concat[CONCAT_DIM];
    float gate_pre[HIDDEN_DIM];
    float gate[HIDDEN_DIM];
    float cand_pre[HIDDEN_DIM];
    float candidate[HIDDEN_DIM];
    float decay[HIDDEN_DIM];
    float h_new[HIDDEN_DIM];
    float y_pred[OUTPUT_DIM];
} StepCache;

static void forward_step(
    const float* x,            /* [INPUT_DIM] */
    const float* h_prev,       /* [HIDDEN_DIM] */
    const TrainParams* p,
    StepCache* cache
) {
    /* Concat [x; h_prev] */
    memcpy(cache->concat, x, INPUT_DIM * sizeof(float));
    memcpy(cache->concat + INPUT_DIM, h_prev, HIDDEN_DIM * sizeof(float));

    /* Gate = sigmoid(W_gate @ concat + b_gate) */
    for (int i = 0; i < HIDDEN_DIM; i++) {
        float sum = p->b_gate[i];
        for (int j = 0; j < CONCAT_DIM; j++) {
            sum += p->W_gate[i * CONCAT_DIM + j] * cache->concat[j];
        }
        cache->gate_pre[i] = sum;
        cache->gate[i] = 1.0f / (1.0f + expf(-sum));  /* sigmoid */
    }

    /* Candidate = tanh(W_cand @ concat + b_cand) */
    for (int i = 0; i < HIDDEN_DIM; i++) {
        float sum = p->b_cand[i];
        for (int j = 0; j < CONCAT_DIM; j++) {
            sum += p->W_cand[i * CONCAT_DIM + j] * cache->concat[j];
        }
        cache->cand_pre[i] = sum;
        cache->candidate[i] = tanhf(sum);
    }

    /* Decay = exp(-dt / tau) */
    for (int i = 0; i < HIDDEN_DIM; i++) {
        float tau_i = fabsf(p->tau[i]) + 1e-6f;  /* ensure positive */
        cache->decay[i] = expf(-DT / tau_i);
    }

    /* h_new = (1 - gate) * h_prev * decay + gate * candidate */
    for (int i = 0; i < HIDDEN_DIM; i++) {
        float g = cache->gate[i];
        cache->h_new[i] = (1.0f - g) * h_prev[i] * cache->decay[i]
                         + g * cache->candidate[i];
    }

    /* Output = W_out @ h_new + b_out */
    for (int i = 0; i < OUTPUT_DIM; i++) {
        float sum = p->b_out[i];
        for (int j = 0; j < HIDDEN_DIM; j++) {
            sum += p->W_out[i * HIDDEN_DIM + j] * cache->h_new[j];
        }
        cache->y_pred[i] = sum;
    }
}

/* ============================================================================
 * BACKWARD PASS — single CfC step
 *
 * Given dL/dh_new and dL/dy_pred, computes gradients for all parameters
 * and dL/dh_prev (for BPTT).
 * ============================================================================ */

static void backward_step(
    const float* h_prev,       /* [HIDDEN_DIM] */
    const StepCache* cache,
    const TrainParams* p,
    const float* dL_dy,        /* [OUTPUT_DIM] */
    const float* dL_dh_new,    /* [HIDDEN_DIM] from future step, or NULL */
    GradParams* grad,          /* accumulate into */
    float* dL_dh_prev          /* [HIDDEN_DIM] output */
) {
    /* --- Output layer gradients --- */
    /* y = W_out @ h_new + b_out */
    /* dL/dW_out[i][j] += dL/dy[i] * h_new[j] */
    /* dL/db_out[i] += dL/dy[i] */
    /* dL/dh_new_out[j] += sum_i dL/dy[i] * W_out[i][j] */

    float dL_dh[HIDDEN_DIM];
    memset(dL_dh, 0, sizeof(dL_dh));

    for (int i = 0; i < OUTPUT_DIM; i++) {
        grad->b_out[i] += dL_dy[i];
        for (int j = 0; j < HIDDEN_DIM; j++) {
            grad->W_out[i * HIDDEN_DIM + j] += dL_dy[i] * cache->h_new[j];
            dL_dh[j] += dL_dy[i] * p->W_out[i * HIDDEN_DIM + j];
        }
    }

    /* Add gradient from future timestep (BPTT) */
    if (dL_dh_new) {
        for (int i = 0; i < HIDDEN_DIM; i++) {
            dL_dh[i] += dL_dh_new[i];
        }
    }

    /* --- CfC cell gradients --- */
    /* h_new[i] = (1 - gate[i]) * h_prev[i] * decay[i] + gate[i] * candidate[i] */

    float dL_dgate[HIDDEN_DIM];
    float dL_dcand[HIDDEN_DIM];
    float dL_ddecay[HIDDEN_DIM];
    memset(dL_dh_prev, 0, HIDDEN_DIM * sizeof(float));

    for (int i = 0; i < HIDDEN_DIM; i++) {
        float g = cache->gate[i];
        float c = cache->candidate[i];
        float d = cache->decay[i];
        float hp = h_prev[i];

        /* dL/dgate = dL/dh * (candidate - h_prev * decay) */
        dL_dgate[i] = dL_dh[i] * (c - hp * d);

        /* dL/dcandidate = dL/dh * gate */
        dL_dcand[i] = dL_dh[i] * g;

        /* dL/ddecay = dL/dh * (1 - gate) * h_prev */
        dL_ddecay[i] = dL_dh[i] * (1.0f - g) * hp;

        /* dL/dh_prev = dL/dh * (1 - gate) * decay */
        dL_dh_prev[i] = dL_dh[i] * (1.0f - g) * d;
    }

    /* --- Through sigmoid: dL/dgate_pre = dL/dgate * gate * (1 - gate) --- */
    float dL_dgate_pre[HIDDEN_DIM];
    for (int i = 0; i < HIDDEN_DIM; i++) {
        float g = cache->gate[i];
        dL_dgate_pre[i] = dL_dgate[i] * g * (1.0f - g);
    }

    /* --- Through tanh: dL/dcand_pre = dL/dcand * (1 - candidate^2) --- */
    float dL_dcand_pre[HIDDEN_DIM];
    for (int i = 0; i < HIDDEN_DIM; i++) {
        float c = cache->candidate[i];
        dL_dcand_pre[i] = dL_dcand[i] * (1.0f - c * c);
    }

    /* --- W_gate, b_gate gradients --- */
    /* gate_pre[i] = sum_j W_gate[i][j] * concat[j] + b_gate[i] */
    for (int i = 0; i < HIDDEN_DIM; i++) {
        grad->b_gate[i] += dL_dgate_pre[i];
        for (int j = 0; j < CONCAT_DIM; j++) {
            grad->W_gate[i * CONCAT_DIM + j] += dL_dgate_pre[i] * cache->concat[j];
        }
    }

    /* --- W_cand, b_cand gradients --- */
    for (int i = 0; i < HIDDEN_DIM; i++) {
        grad->b_cand[i] += dL_dcand_pre[i];
        for (int j = 0; j < CONCAT_DIM; j++) {
            grad->W_cand[i * CONCAT_DIM + j] += dL_dcand_pre[i] * cache->concat[j];
        }
    }

    /* --- tau gradient --- */
    /* decay = exp(-dt / |tau|), d(decay)/d(tau) = decay * dt / tau^2 * sign(tau) */
    for (int i = 0; i < HIDDEN_DIM; i++) {
        float tau_i = fabsf(p->tau[i]) + 1e-6f;
        float sign_tau = p->tau[i] >= 0.0f ? 1.0f : -1.0f;
        grad->tau[i] += dL_ddecay[i] * cache->decay[i] * DT / (tau_i * tau_i) * sign_tau;
    }

    /* --- dL/dh_prev also gets contributions through concat (positions INPUT_DIM..CONCAT_DIM-1) --- */
    for (int i = 0; i < HIDDEN_DIM; i++) {
        for (int j = INPUT_DIM; j < CONCAT_DIM; j++) {
            int h_idx = j - INPUT_DIM;
            dL_dh_prev[h_idx] += dL_dgate_pre[i] * p->W_gate[i * CONCAT_DIM + j];
            dL_dh_prev[h_idx] += dL_dcand_pre[i] * p->W_cand[i * CONCAT_DIM + j];
        }
    }
}

/* ============================================================================
 * INITIALIZATION
 * ============================================================================ */

static float randf(void) {
    return (float)rand() / (float)RAND_MAX;
}

static void init_params(TrainParams* p) {
    /* Xavier-ish initialization scaled for small network */
    float scale_gc = 1.0f / sqrtf((float)CONCAT_DIM);
    float scale_out = 1.0f / sqrtf((float)HIDDEN_DIM);

    for (int i = 0; i < HIDDEN_DIM * CONCAT_DIM; i++) {
        p->W_gate[i] = (randf() * 2.0f - 1.0f) * scale_gc;
        p->W_cand[i] = (randf() * 2.0f - 1.0f) * scale_gc;
    }
    for (int i = 0; i < HIDDEN_DIM; i++) {
        p->b_gate[i] = 0.0f;
        p->b_cand[i] = 0.0f;
        p->tau[i] = 0.5f + randf() * 1.5f;  /* tau in [0.5, 2.0] */
    }
    for (int i = 0; i < OUTPUT_DIM * HIDDEN_DIM; i++) {
        p->W_out[i] = (randf() * 2.0f - 1.0f) * scale_out;
    }
    for (int i = 0; i < OUTPUT_DIM; i++) {
        p->b_out[i] = 0.0f;
    }
}

/* ============================================================================
 * SGD UPDATE (with gradient clipping)
 * ============================================================================ */

static void sgd_step(TrainParams* p, const GradParams* grad, float lr) {
    float* pw = (float*)p;
    const float* gw = (const float*)grad;
    int n = sizeof(TrainParams) / sizeof(float);

    /* Gradient norm for clipping */
    float norm_sq = 0.0f;
    for (int i = 0; i < n; i++) norm_sq += gw[i] * gw[i];
    float norm = sqrtf(norm_sq + 1e-8f);
    float clip = (norm > 5.0f) ? 5.0f / norm : 1.0f;

    for (int i = 0; i < n; i++) {
        pw[i] -= lr * gw[i] * clip;
    }
}

/* ============================================================================
 * TRAINING LOOP
 * ============================================================================ */

static float train_epoch(TrainParams* p, int epoch) {
    GradParams grad;
    memset(&grad, 0, sizeof(grad));

    float total_loss = 0.0f;
    int total_steps = 0;

    /* Generate 10 random-phase sine sequences per epoch */
    for (int seq = 0; seq < 10; seq++) {
        float phase = randf() * 2.0f * 3.14159265f;
        float h[HIDDEN_DIM] = {0};
        StepCache caches[SEQ_LEN];
        float h_history[SEQ_LEN + 1][HIDDEN_DIM];
        float targets[SEQ_LEN];
        float losses[SEQ_LEN];

        memset(h_history[0], 0, sizeof(h));

        /* Forward pass through sequence */
        for (int t = 0; t < SEQ_LEN; t++) {
            float time_val = (float)t * DT + phase;
            float x[INPUT_DIM] = { sinf(time_val) };
            targets[t] = sinf(time_val + DT);  /* predict next step */

            forward_step(x, h_history[t], p, &caches[t]);
            memcpy(h_history[t + 1], caches[t].h_new, sizeof(h));

            /* MSE loss */
            float err = caches[t].y_pred[0] - targets[t];
            losses[t] = 0.5f * err * err;
            total_loss += losses[t];
            total_steps++;
        }

        /* Backward pass (BPTT) */
        float dL_dh_next[HIDDEN_DIM];
        memset(dL_dh_next, 0, sizeof(dL_dh_next));

        for (int t = SEQ_LEN - 1; t >= 0; t--) {
            float err = caches[t].y_pred[0] - targets[t];
            float dL_dy[OUTPUT_DIM] = { err };
            float dL_dh_prev[HIDDEN_DIM];

            backward_step(
                h_history[t], &caches[t], p,
                dL_dy, dL_dh_next, &grad, dL_dh_prev
            );
            memcpy(dL_dh_next, dL_dh_prev, sizeof(dL_dh_next));
        }
    }

    /* Average gradients */
    float* gw = (float*)&grad;
    int n = sizeof(GradParams) / sizeof(float);
    for (int i = 0; i < n; i++) gw[i] /= (float)total_steps;

    /* SGD update */
    sgd_step(p, &grad, LEARNING_RATE);

    return total_loss / (float)total_steps;
}

/* ============================================================================
 * TERNARY QUANTIZATION AND YINSEN INFERENCE
 * ============================================================================ */

static float eval_ternary(const TrainParams* float_params) {
    /* Quantize weight matrices to ternary with per-row scales.
     *
     * BitNet approach: for each row, scale = mean(|w|).
     * Ternary dot product result is multiplied by scale to recover magnitude.
     * This means inference is: y[i] = scale[i] * ternary_dot(W_row[i], x) + bias[i]
     *
     * We do this manually rather than through yinsen_cfc_ternary_cell because
     * the cell doesn't know about per-row scales. This is the scaled-ternary
     * inference path.
     */

    uint8_t W_gate_packed[HIDDEN_DIM * ((CONCAT_DIM + 3) / 4)];
    uint8_t W_cand_packed[HIDDEN_DIM * ((CONCAT_DIM + 3) / 4)];
    uint8_t W_out_packed[OUTPUT_DIM * ((HIDDEN_DIM + 3) / 4)];

    float gate_scales[HIDDEN_DIM];
    float cand_scales[HIDDEN_DIM];
    float out_scales[OUTPUT_DIM];

    /* Per-row quantization: quantize and save the absmean scale */
    {
        int bpr = (CONCAT_DIM + 3) / 4;
        for (int r = 0; r < HIDDEN_DIM; r++) {
            const float* row = float_params->W_gate + r * CONCAT_DIM;
            float absmean = 0.0f;
            for (int j = 0; j < CONCAT_DIM; j++) absmean += fabsf(row[j]);
            absmean /= (float)CONCAT_DIM;
            gate_scales[r] = absmean;
            ternary_quantize_absmean(row, W_gate_packed + r * bpr, CONCAT_DIM);
        }
        for (int r = 0; r < HIDDEN_DIM; r++) {
            const float* row = float_params->W_cand + r * CONCAT_DIM;
            float absmean = 0.0f;
            for (int j = 0; j < CONCAT_DIM; j++) absmean += fabsf(row[j]);
            absmean /= (float)CONCAT_DIM;
            cand_scales[r] = absmean;
            ternary_quantize_absmean(row, W_cand_packed + r * bpr, CONCAT_DIM);
        }
        int bpr_out = (HIDDEN_DIM + 3) / 4;
        for (int r = 0; r < OUTPUT_DIM; r++) {
            const float* row = float_params->W_out + r * HIDDEN_DIM;
            float absmean = 0.0f;
            for (int j = 0; j < HIDDEN_DIM; j++) absmean += fabsf(row[j]);
            absmean /= (float)HIDDEN_DIM;
            out_scales[r] = absmean;
            ternary_quantize_absmean(row, W_out_packed + r * bpr_out, HIDDEN_DIM);
        }
    }

    /* Print ternary weight stats */
    int gate_size = HIDDEN_DIM * CONCAT_DIM;
    int cand_size = HIDDEN_DIM * CONCAT_DIM;
    int out_size  = OUTPUT_DIM * HIDDEN_DIM;
    TernaryStats gate_stats, cand_stats, out_stats;
    ternary_stats(W_gate_packed, gate_size, &gate_stats);
    ternary_stats(W_cand_packed, cand_size, &cand_stats);
    ternary_stats(W_out_packed, out_size, &out_stats);

    printf("  W_gate sparsity: %.1f%% (+:%d, -:%d, 0:%d)\n",
           gate_stats.sparsity * 100.0f, gate_stats.positive, gate_stats.negative, gate_stats.zeros);
    printf("  W_cand sparsity: %.1f%% (+:%d, -:%d, 0:%d)\n",
           cand_stats.sparsity * 100.0f, cand_stats.positive, cand_stats.negative, cand_stats.zeros);
    printf("  W_out  sparsity: %.1f%% (+:%d, -:%d, 0:%d)\n",
           out_stats.sparsity * 100.0f, out_stats.positive, out_stats.negative, out_stats.zeros);

    /* Scaled-ternary CfC inference (manual, since yinsen_cfc_ternary_cell
     * doesn't support per-row scales) */
    int bpr = (CONCAT_DIM + 3) / 4;
    int bpr_out = (HIDDEN_DIM + 3) / 4;

    float total_mse = 0.0f;
    float total_mse_float = 0.0f;
    int steps = 0;
    float h[HIDDEN_DIM] = {0};
    float h_float[HIDDEN_DIM] = {0};

    printf("\n  t       | target  | float   | ternary | f_err   | t_err\n");
    printf("  --------+---------+---------+---------+---------+--------\n");

    for (int t = 0; t < 100; t++) {
        float time_val = (float)t * DT;
        float x_val = sinf(time_val);
        float x[INPUT_DIM] = { x_val };
        float target = sinf(time_val + DT);

        /* Build concat = [x; h] */
        float concat[CONCAT_DIM];
        concat[0] = x_val;
        memcpy(concat + INPUT_DIM, h, HIDDEN_DIM * sizeof(float));

        /* Gate = sigmoid(scale * ternary_dot(W_gate_row, concat) + bias) */
        float gate[HIDDEN_DIM], candidate[HIDDEN_DIM], decay[HIDDEN_DIM];
        for (int i = 0; i < HIDDEN_DIM; i++) {
            float dot = ternary_dot(W_gate_packed + i * bpr, concat, CONCAT_DIM);
            float pre = gate_scales[i] * dot + float_params->b_gate[i];
            gate[i] = 1.0f / (1.0f + expf(-pre));
        }

        /* Candidate = tanh(scale * ternary_dot(W_cand_row, concat) + bias) */
        for (int i = 0; i < HIDDEN_DIM; i++) {
            float dot = ternary_dot(W_cand_packed + i * bpr, concat, CONCAT_DIM);
            float pre = cand_scales[i] * dot + float_params->b_cand[i];
            candidate[i] = tanhf(pre);
        }

        /* Decay */
        for (int i = 0; i < HIDDEN_DIM; i++) {
            float tau_i = fabsf(float_params->tau[i]) + 1e-6f;
            decay[i] = expf(-DT / tau_i);
        }

        /* h_new */
        float h_new[HIDDEN_DIM];
        for (int i = 0; i < HIDDEN_DIM; i++) {
            h_new[i] = (1.0f - gate[i]) * h[i] * decay[i] + gate[i] * candidate[i];
        }

        /* Output = scale * ternary_dot(W_out_row, h_new) + bias */
        float y_pred[OUTPUT_DIM];
        for (int i = 0; i < OUTPUT_DIM; i++) {
            float dot = ternary_dot(W_out_packed + i * bpr_out, h_new, HIDDEN_DIM);
            y_pred[i] = out_scales[i] * dot + float_params->b_out[i];
        }

        /* Float path (for comparison) */
        StepCache fcache;
        forward_step(x, h_float, float_params, &fcache);

        float t_err = y_pred[0] - target;
        float f_err = fcache.y_pred[0] - target;
        total_mse += t_err * t_err;
        total_mse_float += f_err * f_err;
        steps++;

        if (t % 10 == 0) {
            printf("  %6.2f  | %+.4f | %+.4f | %+.4f | %+.4f | %+.4f\n",
                   time_val, target, fcache.y_pred[0], y_pred[0], f_err, t_err);
        }

        memcpy(h, h_new, sizeof(h));
        memcpy(h_float, fcache.h_new, sizeof(h_float));
    }

    return total_mse / (float)steps;
}

static float eval_float(const TrainParams* p) {
    float total_mse = 0.0f;
    int steps = 0;
    float h[HIDDEN_DIM] = {0};

    for (int t = 0; t < 100; t++) {
        float time_val = (float)t * DT;
        float x[INPUT_DIM] = { sinf(time_val) };
        float target = sinf(time_val + DT);

        StepCache cache;
        forward_step(x, h, p, &cache);

        float err = cache.y_pred[0] - target;
        total_mse += err * err;
        steps++;

        memcpy(h, cache.h_new, sizeof(h));
    }

    return total_mse / (float)steps;
}

/* ============================================================================
 * MAIN
 * ============================================================================ */

int main(void) {
    srand(42);

    printf("===================================================\n");
    printf("  YINSEN CfC TRAINING: SINE WAVE PREDICTION\n");
    printf("===================================================\n\n");

    printf("Architecture: %d -> %d -> %d\n", INPUT_DIM, HIDDEN_DIM, OUTPUT_DIM);
    printf("Parameters:   %d (float training)\n",
           (int)(sizeof(TrainParams) / sizeof(float)));
    printf("Task:         predict sin(t + dt) from sin(t)\n");
    printf("Epochs:       %d\n", NUM_EPOCHS);
    printf("LR:           %.4f\n\n", LEARNING_RATE);

    /* Initialize */
    TrainParams params;
    init_params(&params);

    /* Train */
    printf("Training...\n");
    for (int epoch = 0; epoch < NUM_EPOCHS; epoch++) {
        float loss = train_epoch(&params, epoch);
        if (epoch % 20 == 0 || epoch == NUM_EPOCHS - 1) {
            printf("  Epoch %3d/%d  loss=%.6f\n", epoch + 1, NUM_EPOCHS, loss);
        }
    }

    /* Evaluate float model */
    printf("\n--- Float Evaluation ---\n");
    float float_mse = eval_float(&params);
    printf("  Float MSE:  %.6f  (RMSE: %.4f)\n", float_mse, sqrtf(float_mse));

    /* Quantize and evaluate ternary */
    printf("\n--- Ternary Quantization & Evaluation ---\n");
    float ternary_mse = eval_ternary(&params);
    printf("\n  Ternary MSE: %.6f  (RMSE: %.4f)\n", ternary_mse, sqrtf(ternary_mse));

    /* Summary */
    printf("\n===================================================\n");
    printf("  RESULTS\n");
    printf("===================================================\n");
    printf("  Float MSE:        %.6f\n", float_mse);
    printf("  Ternary MSE:      %.6f\n", ternary_mse);
    printf("  Degradation:      %.2fx\n", ternary_mse / (float_mse + 1e-10f));

    size_t float_bytes = sizeof(TrainParams);
    size_t ternary_w_bytes = (HIDDEN_DIM * CONCAT_DIM + 3) / 4 * 2  /* W_gate + W_cand */
                           + (OUTPUT_DIM * HIDDEN_DIM + 3) / 4;     /* W_out */
    size_t ternary_total = ternary_w_bytes
                         + (HIDDEN_DIM * 2 + OUTPUT_DIM) * sizeof(float) /* biases */
                         + HIDDEN_DIM * sizeof(float);                    /* tau */

    printf("  Float size:       %zu bytes\n", float_bytes);
    printf("  Ternary size:     %zu bytes (weights=%zu, biases+tau=%zu)\n",
           ternary_total, ternary_w_bytes, ternary_total - ternary_w_bytes);
    printf("  Compression:      %.1fx\n", (float)float_bytes / (float)ternary_total);

    if (ternary_mse < 0.1f) {
        printf("\n  TERNARY CfC LEARNED SINE PREDICTION.\n");
    } else if (ternary_mse < 0.5f) {
        printf("\n  Partial learning. Ternary approximation is rough.\n");
    } else {
        printf("\n  Ternary quantization destroyed the model.\n");
    }

    printf("===================================================\n");

    return 0;
}
