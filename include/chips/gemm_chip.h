/*
 * GEMM_CHIP — Frozen General Matrix-Multiply Primitive
 *
 * C[M,N] = alpha * A[M,K] @ B[K,N] + beta * bias[N]
 *
 * This is the hottest instruction in any neural network.
 * The CfC cell calls it twice per step (gate + candidate).
 * Everything else is O(hidden_dim). This is O(hidden_dim * concat_dim).
 *
 * Three variants, same interface:
 *   GEMM_CHIP        — Full GEMM with alpha/beta scaling
 *   GEMM_CHIP_BIASED — Simplified: C = A @ B + bias (alpha=1, beta=1)
 *   GEMM_CHIP_BARE   — Minimal: C = A @ B (no bias, no scaling)
 *
 * For M=1 (matvec, the CfC case), the inner loop is a dot product.
 * The compiler can vectorize this with FMA on any modern target.
 *
 * Created by: Tripp + Claude
 * Date: January 31, 2026
 */

#ifndef TRIX_GEMM_CHIP_H
#define TRIX_GEMM_CHIP_H

#ifdef __cplusplus
extern "C" {
#endif

/**
 * GEMM_CHIP_BARE — C[M,N] = A[M,K] @ B[K,N]
 *
 * No bias. No scaling. Just the matmul.
 * 28 bytes of .text at M=1,N=8,K=12 on ARM64 -O2.
 */
static inline void GEMM_CHIP_BARE(
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

/**
 * GEMM_CHIP_BIASED — C[M,N] = A[M,K] @ B[K,N] + bias[N]
 *
 * The CfC case: alpha=1, beta=1, bias is per-column.
 * Fuses bias addition into the store to avoid a second pass.
 */
static inline void GEMM_CHIP_BIASED(
    const float* a, const float* b, const float* bias, float* c,
    int M, int N, int K
) {
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float sum = bias[j];
            for (int k = 0; k < K; k++) {
                sum += a[i * K + k] * b[k * N + j];
            }
            c[i * N + j] = sum;
        }
    }
}

/**
 * GEMM_CHIP — C[M,N] = alpha * A[M,K] @ B[K,N] + beta * bias[N]
 *
 * Full ONNX GEMM. Compatible with yinsen_gemm().
 * Use GEMM_CHIP_BIASED if alpha=1, beta=1 (saves 2 multiplies per element).
 */
static inline void GEMM_CHIP(
    const float* a, const float* b, const float* bias, float* c,
    int M, int N, int K,
    float alpha, float beta
) {
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float sum = 0.0f;
            for (int k = 0; k < K; k++) {
                sum += a[i * K + k] * b[k * N + j];
            }
            c[i * N + j] = alpha * sum + beta * bias[j];
        }
    }
}

/**
 * MATVEC_CHIP — y[N] = W[N,K] @ x[K] + bias[N]
 *
 * Specialized M=1 case. The inner loop is a pure dot product.
 * This is what CfC actually calls: one input vector, one weight matrix.
 *
 * Laid out for the compiler to auto-vectorize the K-loop.
 */
static inline void MATVEC_CHIP(
    const float* x, const float* W, const float* bias, float* y,
    int N, int K
) {
    for (int i = 0; i < N; i++) {
        const float* w_row = W + i * K;
        float sum = bias[i];
        for (int k = 0; k < K; k++) {
            sum += w_row[k] * x[k];
        }
        y[i] = sum;
    }
}

#ifdef __cplusplus
}
#endif

#endif /* TRIX_GEMM_CHIP_H */
