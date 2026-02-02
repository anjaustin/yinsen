/*
 * ternary_sme.h - SME 16x16 Ternary Matrix Operations for Apple M4
 *
 * This header provides the public API for ternary matrix operations
 * using ARM's Scalable Matrix Extension (SME) available on M4 chips.
 *
 * Key insight: M4 has 512-bit SME with ZA tile storage (16x16 x 32-bit).
 * Instead of unpacking 2-bit trits to floats, we use predicates to mask
 * FMOPA operations: +1 triggers positive accumulation, -1 triggers
 * negative accumulation (via negated activations), 0 is skipped.
 *
 * Weight encoding (canonical - see include/trit_encoding.h):
 *   0b00 = 0  (zero)
 *   0b01 = +1 (positive)
 *   0b10 = -1 (negative)
 *   0b11 = reserved (treated as 0)
 *
 * Copyright 2026 Trix Research
 */

#ifndef TERNARY_SME_H
#define TERNARY_SME_H

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Check if SME is available and usable at runtime.
 *
 * NOTE: As of macOS 15 (Jan 2026), Apple has NOT enabled SME for userspace.
 * The M4 hardware has SME, but the OS doesn't allow user applications to use
 * streaming mode. This function returns false on macOS until Apple enables SME.
 *
 * When sme_available() returns false, the API uses optimized scalar fallbacks.
 * These are still fast (~5-7 Gop/s on M4) but not as fast as true SME would be.
 *
 * Returns: true if SME can be used, false otherwise
 */
bool sme_available(void);

/*
 * Check if SME hardware is present (independent of OS enablement).
 * Use this to report hardware capabilities, not to decide code paths.
 */
bool sme_hardware_present(void);

/*
 * Ternary dot product of two 16-element vectors.
 *
 * Computes: sum(activations[i] * decode(weights[i])) for i in 0..15
 *
 * Parameters:
 *   activations - 16 float32 values (must be 64-byte aligned for optimal performance)
 *   weights     - 32 bits containing 16 x 2-bit ternary weights (packed)
 *
 * Returns: scalar dot product result
 *
 * Weight packing: bits [1:0] = weight[0], bits [3:2] = weight[1], etc.
 */
float sme_dot16(const float* activations, uint32_t weights);

/*
 * Ternary matrix-vector multiplication using SME 16x16 tiles.
 *
 * Computes: output[i] = sum(weights[i,j] * input[j]) for j in 0..K-1
 *
 * Parameters:
 *   output  - M float32 values (output vector)
 *   weights - Packed ternary weights in SME column-interleaved format
 *             Layout: column-slice major for ld1rw broadcast
 *             Size: ceil(M/16) * ceil(K/16) * 16 * sizeof(uint32_t) bytes
 *   input   - K float32 values (input vector)
 *   M       - Number of output rows (must be multiple of 16, or will be padded)
 *   K       - Number of input columns (must be multiple of 16, or will be padded)
 *
 * Note: This function internally handles non-aligned dimensions by padding.
 *       For best performance, use dimensions that are multiples of 16.
 *       Optimized ASM kernels exist for 16x16 and 32x32 (dispatched automatically).
 */
void sme_matvec(float* output, const uint32_t* weights, const float* input, 
                size_t M, size_t K);

/*
 * Batched ternary matrix-vector multiplication.
 *
 * Computes output[b,i] = sum(weights[i,j] * input[b,j]) for all batches.
 *
 * Parameters:
 *   output     - batch_size x M float32 values (row-major)
 *   weights    - Packed ternary weights in SME format
 *   input      - batch_size x K float32 values (row-major)
 *   M          - Number of output rows per batch
 *   K          - Number of input columns per batch
 *   batch_size - Number of vectors to process
 *
 * This is where SME shines - batched operations amortize setup cost.
 * Recommended batch_size >= 16 for optimal throughput.
 */
void sme_matvec_batch(float* output, const uint32_t* weights, const float* input,
                      size_t M, size_t K, size_t batch_size);

/*
 * Full matrix-matrix multiplication with ternary weights.
 *
 * Computes: C[i,j] = sum(A[i,k] * decode(W[k,j])) for k in 0..K-1
 *
 * Parameters:
 *   C       - M x N output matrix (row-major)
 *   A       - M x K input matrix (row-major, float32)
 *   W       - K x N ternary weight matrix (SME column-interleaved format)
 *   M, K, N - Matrix dimensions
 *
 * This is the highest-throughput operation, using full ZA tile accumulation.
 */
void sme_matmul(float* C, const float* A, const uint32_t* W,
                size_t M, size_t K, size_t N);

/*
 * Convert weights from row-major packed format to SME column-interleaved format.
 *
 * Parameters:
 *   dst      - Output buffer for SME-formatted weights
 *              Size: sme_weight_buffer_size(rows, cols) bytes
 *   src      - Input weights in row-major packed format
 *              (same format as Metal: 2 bits per weight, row-major)
 *   rows     - Number of rows in weight matrix
 *   cols     - Number of columns in weight matrix
 *
 * Call sme_weight_buffer_size() first to determine required dst buffer size.
 */
void sme_pack_weights(uint32_t* dst, const uint8_t* src, size_t rows, size_t cols);

/*
 * Calculate required buffer size for SME-formatted weights.
 *
 * Parameters:
 *   rows - Number of rows in weight matrix
 *   cols - Number of columns in weight matrix
 *
 * Returns: Required buffer size in bytes
 */
size_t sme_weight_buffer_size(size_t rows, size_t cols);

/*
 * Reference implementation for verification (CPU scalar).
 * This is NOT optimized - use only for testing.
 */
float sme_dot16_ref(const float* activations, uint32_t weights);
void sme_matvec_ref(float* output, const uint32_t* weights, const float* input,
                    size_t M, size_t K);

#ifdef __cplusplus
}
#endif

#endif /* TERNARY_SME_H */
