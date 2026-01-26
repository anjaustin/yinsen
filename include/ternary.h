/*
 * YINSEN Ternary - Three-Valued Weight Computation
 *
 * Weights are {-1, 0, +1}, not floating point.
 * Matmul becomes add/subtract - no multiplication needed.
 *
 * Benefits:
 *   - Deterministic: integer arithmetic, no float variance
 *   - Tiny: 2 bits per weight (vs 32 for float)
 *   - Fast: add/subtract only, no multiply
 *   - Hardware-friendly: works on anything with an ALU
 *
 * Encoding (2 bits per trit):
 *   00 = 0  (skip)
 *   01 = +1 (add)
 *   11 = -1 (subtract)
 *   10 = reserved
 *
 * Verification status: See test_ternary.c
 */

#ifndef YINSEN_TERNARY_H
#define YINSEN_TERNARY_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ============================================================================
 * TRIT ENCODING
 *
 * A "trit" is a ternary digit: {-1, 0, +1}
 * We pack 4 trits per byte (2 bits each)
 * ============================================================================ */

#define TRIT_ZERO  0x0  /* 00 */
#define TRIT_POS   0x1  /* 01 */
#define TRIT_NEG   0x3  /* 11 */

/* Extract trit at position (0-3) from packed byte */
static inline int8_t trit_unpack(uint8_t packed, int pos) {
    uint8_t bits = (packed >> (pos * 2)) & 0x3;
    if (bits == TRIT_ZERO) return 0;
    if (bits == TRIT_POS)  return 1;
    if (bits == TRIT_NEG)  return -1;
    return 0;  /* Reserved encoding treated as zero */
}

/* Pack trit (-1, 0, +1) into 2-bit encoding */
static inline uint8_t trit_encode(int8_t val) {
    if (val > 0)  return TRIT_POS;
    if (val < 0)  return TRIT_NEG;
    return TRIT_ZERO;
}

/* Pack 4 trits into one byte */
static inline uint8_t trit_pack4(int8_t t0, int8_t t1, int8_t t2, int8_t t3) {
    return (trit_encode(t0) << 0) |
           (trit_encode(t1) << 2) |
           (trit_encode(t2) << 4) |
           (trit_encode(t3) << 6);
}

/* ============================================================================
 * TERNARY VECTOR OPERATIONS
 *
 * Input: float vector x
 * Weights: packed trits
 * Output: float (accumulated sum)
 *
 * y = sum(x[i] where w[i]=+1) - sum(x[i] where w[i]=-1)
 * No multiplication. Just conditional add/subtract.
 * ============================================================================ */

/*
 * Ternary dot product: y = w . x
 *
 * w: packed trit weights (2 bits per weight, 4 weights per byte)
 * x: input vector (float)
 * n: vector length
 *
 * Returns: sum of x[i] where w[i]=+1, minus sum of x[i] where w[i]=-1
 */
static inline float ternary_dot(
    const uint8_t* w_packed,
    const float* x,
    int n
) {
    float sum = 0.0f;
    int byte_idx = 0;
    int bit_pos = 0;

    for (int i = 0; i < n; i++) {
        int8_t trit = trit_unpack(w_packed[byte_idx], bit_pos);

        if (trit > 0) {
            sum += x[i];
        } else if (trit < 0) {
            sum -= x[i];
        }
        /* trit == 0: skip (sparse) */

        bit_pos++;
        if (bit_pos == 4) {
            bit_pos = 0;
            byte_idx++;
        }
    }

    return sum;
}

/*
 * Ternary matrix-vector multiply: y = W @ x
 *
 * W: packed trit matrix [M x N], row-major, 4 trits per byte
 * x: input vector [N]
 * y: output vector [M]
 *
 * Each row of W is a packed trit vector.
 * Bytes per row = ceil(N / 4)
 */
static inline void ternary_matvec(
    const uint8_t* W_packed,
    const float* x,
    float* y,
    int M,
    int N
) {
    int bytes_per_row = (N + 3) / 4;

    for (int i = 0; i < M; i++) {
        y[i] = ternary_dot(W_packed + i * bytes_per_row, x, N);
    }
}

/*
 * Ternary matrix-vector multiply with bias: y = W @ x + b
 */
static inline void ternary_matvec_bias(
    const uint8_t* W_packed,
    const float* x,
    const float* bias,
    float* y,
    int M,
    int N
) {
    ternary_matvec(W_packed, x, y, M, N);
    for (int i = 0; i < M; i++) {
        y[i] += bias[i];
    }
}

/* ============================================================================
 * TERNARY WEIGHT UTILITIES
 * ============================================================================ */

/* Bytes needed to store n trits */
static inline size_t ternary_bytes(int n) {
    return (n + 3) / 4;
}

/* Bytes needed to store an M x N ternary matrix */
static inline size_t ternary_matrix_bytes(int M, int N) {
    return M * ternary_bytes(N);
}

/*
 * Pack float weights to ternary.
 *
 * Quantization: val > threshold  -> +1
 *               val < -threshold -> -1
 *               otherwise        -> 0
 */
static inline void ternary_quantize(
    const float* weights,
    uint8_t* packed,
    int n,
    float threshold
) {
    int byte_idx = 0;
    uint8_t current_byte = 0;

    for (int i = 0; i < n; i++) {
        int8_t trit;
        if (weights[i] > threshold) {
            trit = 1;
        } else if (weights[i] < -threshold) {
            trit = -1;
        } else {
            trit = 0;
        }

        int bit_pos = i % 4;
        current_byte |= (trit_encode(trit) << (bit_pos * 2));

        if (bit_pos == 3 || i == n - 1) {
            packed[byte_idx++] = current_byte;
            current_byte = 0;
        }
    }
}

/*
 * Unpack ternary weights to float (-1.0, 0.0, +1.0)
 */
static inline void ternary_unpack_to_float(
    const uint8_t* packed,
    float* weights,
    int n
) {
    int byte_idx = 0;
    int bit_pos = 0;

    for (int i = 0; i < n; i++) {
        weights[i] = (float)trit_unpack(packed[byte_idx], bit_pos);

        bit_pos++;
        if (bit_pos == 4) {
            bit_pos = 0;
            byte_idx++;
        }
    }
}

/* ============================================================================
 * STATISTICS
 * ============================================================================ */

/* Count non-zero trits (sparsity measure) */
static inline int ternary_count_nonzero(const uint8_t* packed, int n) {
    int count = 0;
    int byte_idx = 0;
    int bit_pos = 0;

    for (int i = 0; i < n; i++) {
        int8_t trit = trit_unpack(packed[byte_idx], bit_pos);
        if (trit != 0) count++;

        bit_pos++;
        if (bit_pos == 4) {
            bit_pos = 0;
            byte_idx++;
        }
    }

    return count;
}

/* Memory comparison: ternary vs float */
static inline void ternary_memory_stats(
    int n,
    size_t* ternary_bytes_out,
    size_t* float_bytes_out,
    float* compression_ratio
) {
    *ternary_bytes_out = ternary_bytes(n);
    *float_bytes_out = n * sizeof(float);
    *compression_ratio = (float)(*float_bytes_out) / (float)(*ternary_bytes_out);
}

#ifdef __cplusplus
}
#endif

#endif /* YINSEN_TERNARY_H */
