/*
 * YINSEN Ternary - 1.58-bit Weight Computation
 *
 * Weights are {-1, 0, +1}, requiring log2(3) = 1.58 bits per weight.
 * Matmul becomes add/subtract - no multiplication needed.
 *
 * Benefits:
 *   - Deterministic: integer arithmetic, no float variance
 *   - Tiny: 2 bits per weight storage (vs 32 for float)
 *   - Fast: add/subtract only, no multiply
 *   - Hardware-friendly: works on anything with an ALU
 *   - Auditable: every weight is inspectable as -1, 0, or +1
 *
 * Encoding (2 bits per trit):
 *   00 = 0  (skip - explicit feature filtering)
 *   01 = +1 (add)
 *   11 = -1 (subtract)
 *   10 = reserved
 *
 * Note: Zero is not "missing" - it's explicit "ignore this input".
 * This enables feature filtering (BitNet b1.58 insight).
 *
 * Verification status: See test_ternary.c
 */

#ifndef YINSEN_TERNARY_H
#define YINSEN_TERNARY_H

#include <stdint.h>
#include <stddef.h>
#include <math.h>

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
 * Pack float weights to ternary using threshold.
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
 * Pack float weights to ternary using absmean quantization.
 * (BitNet b1.58 method)
 *
 * Algorithm:
 *   1. Compute γ = mean(|W|)
 *   2. Scale: W_scaled = W / γ
 *   3. Round to nearest in {-1, 0, +1}
 *
 * This adapts to the weight distribution automatically.
 */
static inline void ternary_quantize_absmean(
    const float* weights,
    uint8_t* packed,
    int n
) {
    /* Step 1: Compute absmean (γ) */
    float sum = 0.0f;
    for (int i = 0; i < n; i++) {
        sum += fabsf(weights[i]);
    }
    float gamma = sum / (float)n;
    float eps = 1e-8f;
    
    /* Step 2: Quantize using RoundClip(W/γ, -1, 1) */
    int byte_idx = 0;
    uint8_t current_byte = 0;
    
    for (int i = 0; i < n; i++) {
        float scaled = weights[i] / (gamma + eps);
        
        /* Round to nearest integer, clip to [-1, 1] */
        int rounded = (int)roundf(scaled);
        if (rounded > 1) rounded = 1;
        if (rounded < -1) rounded = -1;
        
        int8_t trit = (int8_t)rounded;
        
        int bit_pos = i % 4;
        current_byte |= (trit_encode(trit) << (bit_pos * 2));
        
        if (bit_pos == 3 || i == n - 1) {
            packed[byte_idx++] = current_byte;
            current_byte = 0;
        }
    }
}

/*
 * Get the absmean scale factor for a weight array.
 * Useful for dequantization or debugging.
 */
static inline float ternary_absmean_scale(const float* weights, int n) {
    float sum = 0.0f;
    for (int i = 0; i < n; i++) {
        sum += fabsf(weights[i]);
    }
    return sum / (float)n;
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
 * INT8 ACTIVATION QUANTIZATION
 *
 * For fully integer forward pass (except nonlinearities).
 * Based on BitNet b1.58's activation handling.
 * ============================================================================ */

typedef struct {
    float scale;       /* Multiply int result by this to get float */
    int8_t zero_point; /* Always 0 for symmetric quantization */
} TernaryQuantParams;

/*
 * Quantize float activations to int8 (symmetric, per-tensor).
 * Range: [-127, 127]
 */
static inline void ternary_quantize_activations(
    const float* x,
    int8_t* x_q,
    int n,
    TernaryQuantParams* params
) {
    /* Find absmax */
    float absmax = 0.0f;
    for (int i = 0; i < n; i++) {
        float abs_val = fabsf(x[i]);
        if (abs_val > absmax) absmax = abs_val;
    }
    
    /* Compute scale */
    params->scale = absmax / 127.0f;
    params->zero_point = 0;
    
    /* Quantize */
    if (absmax < 1e-8f) {
        /* All zeros */
        for (int i = 0; i < n; i++) {
            x_q[i] = 0;
        }
    } else {
        for (int i = 0; i < n; i++) {
            float scaled = x[i] / params->scale;
            int rounded = (int)roundf(scaled);
            if (rounded > 127) rounded = 127;
            if (rounded < -127) rounded = -127;
            x_q[i] = (int8_t)rounded;
        }
    }
}

/*
 * Dequantize int8 back to float.
 */
static inline void ternary_dequantize_activations(
    const int8_t* x_q,
    float* x,
    int n,
    const TernaryQuantParams* params
) {
    for (int i = 0; i < n; i++) {
        x[i] = (float)x_q[i] * params->scale;
    }
}

/*
 * Integer ternary dot product.
 *
 * w: packed trit weights
 * x_q: quantized int8 activations
 * n: vector length
 *
 * Returns: int32 accumulator (multiply by scale later)
 *
 * This is the core BitNet insight: matmul becomes integer add/subtract.
 */
static inline int32_t ternary_dot_int8(
    const uint8_t* w_packed,
    const int8_t* x_q,
    int n
) {
    int32_t sum = 0;
    int byte_idx = 0;
    int bit_pos = 0;
    
    for (int i = 0; i < n; i++) {
        int8_t trit = trit_unpack(w_packed[byte_idx], bit_pos);
        
        if (trit > 0) {
            sum += x_q[i];
        } else if (trit < 0) {
            sum -= x_q[i];
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
 * Integer ternary matvec: y_int = W @ x_q
 * Output is int32, needs dequantization.
 */
static inline void ternary_matvec_int8(
    const uint8_t* W_packed,
    const int8_t* x_q,
    int32_t* y_int,
    int M,
    int N
) {
    int bytes_per_row = (N + 3) / 4;
    
    for (int i = 0; i < M; i++) {
        y_int[i] = ternary_dot_int8(W_packed + i * bytes_per_row, x_q, N);
    }
}

/* ============================================================================
 * ENERGY ESTIMATION
 *
 * Based on Horowitz (2014) energy model for 7nm process.
 * INT8 ADD: ~0.03 pJ
 * FP16 MUL: ~0.9 pJ
 * FP16 ADD: ~0.4 pJ
 * ============================================================================ */

#define YINSEN_ENERGY_INT8_ADD_PJ   0.03f
#define YINSEN_ENERGY_FP16_MUL_PJ   0.9f
#define YINSEN_ENERGY_FP16_ADD_PJ   0.4f

/*
 * Estimate energy for ternary matvec (integer path).
 * Assumes all ops are int8 adds (conservative).
 */
static inline float ternary_matvec_energy_pj(int M, int N) {
    /* N additions per row, M rows */
    return (float)(M * N) * YINSEN_ENERGY_INT8_ADD_PJ;
}

/*
 * Estimate energy for float matvec (standard path).
 * Each element: 1 mul + 1 add.
 */
static inline float float_matvec_energy_pj(int M, int N) {
    return (float)(M * N) * (YINSEN_ENERGY_FP16_MUL_PJ + YINSEN_ENERGY_FP16_ADD_PJ);
}

/*
 * Energy savings ratio: float / ternary
 */
static inline float ternary_energy_savings_ratio(int M, int N) {
    return float_matvec_energy_pj(M, N) / ternary_matvec_energy_pj(M, N);
}

/* ============================================================================
 * STATISTICS
 * ============================================================================ */

/* Count non-zero trits (density measure) */
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

/* Count zeros (sparsity measure - zeros are explicit "ignore" signals) */
static inline int ternary_count_zeros(const uint8_t* packed, int n) {
    return n - ternary_count_nonzero(packed, n);
}

/* Sparsity ratio: fraction of weights that are zero */
static inline float ternary_sparsity(const uint8_t* packed, int n) {
    return (float)ternary_count_zeros(packed, n) / (float)n;
}

/* Count positive weights */
static inline int ternary_count_positive(const uint8_t* packed, int n) {
    int count = 0;
    int byte_idx = 0;
    int bit_pos = 0;

    for (int i = 0; i < n; i++) {
        int8_t trit = trit_unpack(packed[byte_idx], bit_pos);
        if (trit > 0) count++;

        bit_pos++;
        if (bit_pos == 4) {
            bit_pos = 0;
            byte_idx++;
        }
    }

    return count;
}

/* Count negative weights */
static inline int ternary_count_negative(const uint8_t* packed, int n) {
    int count = 0;
    int byte_idx = 0;
    int bit_pos = 0;

    for (int i = 0; i < n; i++) {
        int8_t trit = trit_unpack(packed[byte_idx], bit_pos);
        if (trit < 0) count++;

        bit_pos++;
        if (bit_pos == 4) {
            bit_pos = 0;
            byte_idx++;
        }
    }

    return count;
}

/* Full weight distribution statistics */
typedef struct {
    int total;
    int positive;     /* Count of +1 */
    int negative;     /* Count of -1 */
    int zeros;        /* Count of 0 */
    float sparsity;   /* zeros / total */
} TernaryStats;

static inline void ternary_stats(const uint8_t* packed, int n, TernaryStats* stats) {
    stats->total = n;
    stats->positive = ternary_count_positive(packed, n);
    stats->negative = ternary_count_negative(packed, n);
    stats->zeros = n - stats->positive - stats->negative;
    stats->sparsity = (float)stats->zeros / (float)n;
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
