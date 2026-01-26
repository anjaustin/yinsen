/*
 * YINSEN ONNX Shapes
 *
 * Frozen shapes for neural network operations.
 * Verified: Numerical accuracy tests pass.
 */

#ifndef YINSEN_ONNX_SHAPES_H
#define YINSEN_ONNX_SHAPES_H

#include <math.h>
#include <string.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ============================================================================
 * ACTIVATION FUNCTIONS
 * ============================================================================ */

/* ReLU: max(0, x) */
static inline float yinsen_relu(float x) {
    return x > 0.0f ? x : 0.0f;
}

/* Sigmoid: 1 / (1 + exp(-x)) */
static inline float yinsen_sigmoid(float x) {
    return 1.0f / (1.0f + expf(-x));
}

/* Tanh: (exp(x) - exp(-x)) / (exp(x) + exp(-x)) */
static inline float yinsen_tanh(float x) {
    return tanhf(x);
}

/* GELU: x * sigmoid(1.702 * x) - fast approximation */
static inline float yinsen_gelu(float x) {
    return x * yinsen_sigmoid(1.702f * x);
}

/* SiLU / Swish: x * sigmoid(x) */
static inline float yinsen_silu(float x) {
    return x * yinsen_sigmoid(x);
}

/* ============================================================================
 * SOFTMAX - Numerically stable
 * ============================================================================ */

static inline void yinsen_softmax(const float* x, float* out, int n) {
    /* Subtract max for numerical stability */
    float max_val = x[0];
    for (int i = 1; i < n; i++) {
        if (x[i] > max_val) max_val = x[i];
    }

    /* Compute exp and sum */
    float sum = 0.0f;
    for (int i = 0; i < n; i++) {
        out[i] = expf(x[i] - max_val);
        sum += out[i];
    }

    /* Normalize */
    float inv_sum = 1.0f / sum;
    for (int i = 0; i < n; i++) {
        out[i] *= inv_sum;
    }
}

/* ============================================================================
 * MATRIX OPERATIONS
 * ============================================================================ */

/* MatMul: C[M,N] = A[M,K] @ B[K,N] */
static inline void yinsen_matmul(
    const float* a, const float* b, float* c,
    int M, int N, int K
) {
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float sum = 0.0f;
            for (int k = 0; k < K; k++) {
                sum += a[i * K + k] * b[k * N + j];
            }
            c[i * N + j] = sum;
        }
    }
}

/* GEMM: C = alpha * A @ B + beta * bias */
static inline void yinsen_gemm(
    const float* a, const float* b, const float* bias, float* c,
    int M, int N, int K,
    float alpha, float beta
) {
    yinsen_matmul(a, b, c, M, N, K);

    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            c[i * N + j] = alpha * c[i * N + j] + beta * bias[j];
        }
    }
}

/* ============================================================================
 * REDUCTIONS
 * ============================================================================ */

static inline float yinsen_reduce_sum(const float* x, int n) {
    float sum = 0.0f;
    for (int i = 0; i < n; i++) sum += x[i];
    return sum;
}

static inline float yinsen_reduce_mean(const float* x, int n) {
    return yinsen_reduce_sum(x, n) / (float)n;
}

static inline float yinsen_reduce_max(const float* x, int n) {
    float m = x[0];
    for (int i = 1; i < n; i++) {
        if (x[i] > m) m = x[i];
    }
    return m;
}

/* ============================================================================
 * LAYER NORMALIZATION
 * ============================================================================ */

static inline void yinsen_layer_norm(
    const float* x, const float* gamma, const float* beta,
    float* out, int n, float eps
) {
    /* Compute mean */
    float mean = yinsen_reduce_mean(x, n);

    /* Compute variance */
    float var = 0.0f;
    for (int i = 0; i < n; i++) {
        float d = x[i] - mean;
        var += d * d;
    }
    var /= (float)n;

    /* Normalize and scale */
    float inv_std = 1.0f / sqrtf(var + eps);
    for (int i = 0; i < n; i++) {
        out[i] = gamma[i] * (x[i] - mean) * inv_std + beta[i];
    }
}

#ifdef __cplusplus
}
#endif

#endif /* YINSEN_ONNX_SHAPES_H */
