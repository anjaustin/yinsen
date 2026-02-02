/*
 * SOFTMAX_CHIP — Frozen Numerically Stable Softmax Primitive
 *
 * softmax(x)[i] = exp(x[i] - max(x)) / sum(exp(x[j] - max(x)))
 *
 * Used for: classification output layer after CfC hidden state.
 * The max-subtraction trick prevents overflow.
 *
 * Also includes ARGMAX for when you just need the class index.
 *
 * Created by: Tripp + Claude
 * Date: January 31, 2026
 */

#ifndef TRIX_SOFTMAX_CHIP_H
#define TRIX_SOFTMAX_CHIP_H

#include <math.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * SOFTMAX_CHIP — Numerically stable softmax
 *
 * @param x    Input logits [n]
 * @param out  Output probabilities [n] (can alias x for in-place)
 * @param n    Number of classes
 */
static inline void SOFTMAX_CHIP(const float* x, float* out, int n) {
    /* Pass 1: find max */
    float max_val = x[0];
    for (int i = 1; i < n; i++) {
        if (x[i] > max_val) max_val = x[i];
    }

    /* Pass 2: exp(x - max) and accumulate sum */
    float sum = 0.0f;
    for (int i = 0; i < n; i++) {
        out[i] = expf(x[i] - max_val);
        sum += out[i];
    }

    /* Pass 3: normalize */
    float inv_sum = 1.0f / sum;
    for (int i = 0; i < n; i++) {
        out[i] *= inv_sum;
    }
}

/**
 * SOFTMAX_CHIP_FAST — Using Schraudolph exp
 *
 * No libm. ~4% relative error per element, but the argmax
 * is almost always correct (same ranking).
 */
static inline void SOFTMAX_CHIP_FAST(const float* x, float* out, int n) {
    float max_val = x[0];
    for (int i = 1; i < n; i++) {
        if (x[i] > max_val) max_val = x[i];
    }

    float sum = 0.0f;
    for (int i = 0; i < n; i++) {
        float v = x[i] - max_val;
        if (v < -88.0f) v = -88.0f;
        union { float f; int32_t i; } u;
        u.i = (int32_t)(v * 12102203.0f + 1064866805.0f);
        out[i] = u.f;
        sum += out[i];
    }

    float inv_sum = 1.0f / sum;
    for (int i = 0; i < n; i++) {
        out[i] *= inv_sum;
    }
}

/**
 * ARGMAX_CHIP — Index of maximum element
 *
 * Skips the exp/normalize entirely when you just need the class.
 * O(n) scan, zero math beyond comparison.
 */
static inline int ARGMAX_CHIP(const float* x, int n) {
    int max_idx = 0;
    float max_val = x[0];
    for (int i = 1; i < n; i++) {
        if (x[i] > max_val) {
            max_val = x[i];
            max_idx = i;
        }
    }
    return max_idx;
}

/**
 * TOP_K_CHIP — Find top-K indices (selection sort, small K only)
 *
 * For K=1, use ARGMAX_CHIP instead.
 * For K <= ~5, this is fine. Don't use for K > 10.
 *
 * @param x        Input logits [n]
 * @param n        Number of elements
 * @param indices  Output: top-K indices [k], sorted descending
 * @param k        Number of top elements to find
 */
static inline void TOP_K_CHIP(const float* x, int n, int* indices, int k) {
    /* Selection sort for K elements */
    float used[k]; /* Track values we've already selected */
    (void)used;

    for (int t = 0; t < k; t++) {
        int best_idx = -1;
        float best_val = -1e30f;

        for (int i = 0; i < n; i++) {
            /* Check if already selected */
            int skip = 0;
            for (int s = 0; s < t; s++) {
                if (indices[s] == i) { skip = 1; break; }
            }
            if (skip) continue;

            if (x[i] > best_val) {
                best_val = x[i];
                best_idx = i;
            }
        }
        indices[t] = best_idx;
    }
}

#ifdef __cplusplus
}
#endif

#endif /* TRIX_SOFTMAX_CHIP_H */
