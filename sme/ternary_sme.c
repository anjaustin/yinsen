/*
 * ternary_sme.c - SME 16x16 Ternary Matrix Operations for Apple M4
 *
 * Implementation using ARM SME (Scalable Matrix Extension) intrinsics.
 * This targets the M4 chip which has 512-bit SME with 16x16 ZA tiles.
 *
 * The key insight from the associate: instead of unpacking 2-bit trits
 * to floats, use predicates to mask FMOPA operations:
 *   - Weight == +1 (0b01): accumulate activation
 *   - Weight == -1 (0b10): accumulate negated activation  
 *   - Weight == 0  (0b00 or 0b11): skip (predicate false)
 *
 * Copyright 2026 Trix Research
 */

#include "ternary_sme.h"
#include <string.h>
#include <stdlib.h>

#ifdef __APPLE__
#include <sys/sysctl.h>
#endif

/* ============================================================================
 * Hardware Detection
 * ============================================================================
 *
 * NOTE: As of macOS 15 (Jan 2026), Apple has NOT enabled SME for userspace.
 * The hardware has SME, but the OS doesn't allow user applications to use
 * streaming mode. This function checks hardware presence, but the actual
 * SME code path will cause SIGILL until Apple enables it.
 *
 * For now, we always use the reference implementation on macOS.
 * When Apple enables SME, we can flip the sme_runtime_available() flag.
 */

/* Check if SME hardware is present (doesn't mean OS allows it) */
bool sme_hardware_present(void) {
#ifdef __APPLE__
    // Try the feature flag first (macOS 14+)
    int64_t sme = 0;
    size_t size = sizeof(sme);
    
    if (sysctlbyname("hw.optional.arm.FEAT_SME", &sme, &size, NULL, 0) == 0) {
        return sme != 0;
    }
    
    // Fallback: check CPU brand for M4
    char brand[256] = {0};
    size = sizeof(brand);
    if (sysctlbyname("machdep.cpu.brand_string", brand, &size, NULL, 0) == 0) {
        // M4, M4 Pro, M4 Max, M4 Ultra all have SME
        if (strstr(brand, "M4") != NULL) return true;
        // Future chips (M5+) will also have SME
        if (strstr(brand, "M5") != NULL) return true;
        if (strstr(brand, "M6") != NULL) return true;
    }
    
    return false;
#else
    // Non-Apple platforms: check /proc/cpuinfo or similar
    return false;
#endif
}

/* Check if SME is usable at runtime (OS must allow streaming mode) */
static bool sme_runtime_available(void) {
    /* 
     * TODO: When Apple enables SME for userspace, change this to:
     *   return sme_hardware_present();
     * 
     * For now, always return false to use reference implementation.
     * Attempting to use SME streaming mode on macOS causes SIGILL.
     */
    return false;
}

bool sme_available(void) {
    return sme_runtime_available();
}

/* ============================================================================
 * Reference Implementations (Scalar CPU)
 * ============================================================================ */

/*
 * Decode a 2-bit trit to its value.
 * 0b00 -> 0, 0b01 -> +1, 0b10 -> -1, 0b11 -> 0 (reserved)
 */
static inline int decode_trit(uint32_t trit) {
    // trit is 2 bits
    if (trit == 1) return 1;
    if (trit == 2) return -1;
    return 0;
}

float sme_dot16_ref(const float* activations, uint32_t weights) {
    float sum = 0.0f;
    for (int i = 0; i < 16; i++) {
        int w = decode_trit((weights >> (i * 2)) & 0x3);
        sum += activations[i] * (float)w;
    }
    return sum;
}

void sme_matvec_ref(float* output, const uint32_t* weights, const float* input,
                    size_t M, size_t K) {
    // Reference implementation using row-major packed weights
    // Note: This expects weights in reference format, not SME-interleaved format
    
    size_t M_aligned = (M + 15) & ~15;
    size_t K_aligned = (K + 15) & ~15;
    
    for (size_t i = 0; i < M; i++) {
        float sum = 0.0f;
        for (size_t j = 0; j < K; j++) {
            // Calculate weight position in packed format
            size_t tile_row = i / 16;
            size_t tile_col = j / 16;
            size_t in_tile_row = i % 16;
            size_t in_tile_col = j % 16;
            
            // Each 16x16 tile has 16 uint32_t (one per row, 16 weights each)
            size_t tiles_per_row = K_aligned / 16;
            size_t tile_idx = tile_row * tiles_per_row + tile_col;
            size_t weight_idx = tile_idx * 16 + in_tile_row;
            
            uint32_t packed = weights[weight_idx];
            int w = decode_trit((packed >> (in_tile_col * 2)) & 0x3);
            
            float inp = (j < K) ? input[j] : 0.0f;
            sum += inp * (float)w;
        }
        output[i] = sum;
    }
}

/* ============================================================================
 * Weight Packing
 * ============================================================================ */

size_t sme_weight_buffer_size(size_t rows, size_t cols) {
    // Round up to multiples of 16
    size_t rows_aligned = (rows + 15) & ~15;
    size_t cols_aligned = (cols + 15) & ~15;
    
    // Each 16x16 tile stores 16 uint32_t (16 weights per row, 2 bits each = 32 bits)
    size_t num_tiles = (rows_aligned / 16) * (cols_aligned / 16);
    
    // Each tile needs 16 x uint32_t in SME column-interleaved format
    // Actually we store row-major within tiles, just tiles are arranged for SME access
    return num_tiles * 16 * sizeof(uint32_t);
}

void sme_pack_weights(uint32_t* dst, const uint8_t* src, size_t rows, size_t cols) {
    // Convert from flat row-major 2-bit packed format to SME tile format
    //
    // Input: src contains ceil(rows * cols / 4) bytes, row-major
    //        Each byte has 4 weights: bits [1:0]=w0, [3:2]=w1, [5:4]=w2, [7:6]=w3
    //
    // Output: dst organized as 16x16 tiles, each tile is 16 x uint32_t
    //         Tile order is row-major (tile[0,0], tile[0,1], ..., tile[1,0], ...)
    //         Within tile: row-major, each uint32_t has 16 weights for one row
    
    size_t rows_aligned = (rows + 15) & ~15;
    size_t cols_aligned = (cols + 15) & ~15;
    size_t tiles_per_row = cols_aligned / 16;
    
    // Zero the output first
    memset(dst, 0, sme_weight_buffer_size(rows, cols));
    
    for (size_t r = 0; r < rows; r++) {
        for (size_t c = 0; c < cols; c++) {
            // Find the weight in source
            size_t src_bit_idx = (r * cols + c) * 2;
            size_t src_byte = src_bit_idx / 8;
            size_t src_bit_offset = src_bit_idx % 8;
            uint8_t trit = (src[src_byte] >> src_bit_offset) & 0x3;
            
            // Find destination position
            size_t tile_row = r / 16;
            size_t tile_col = c / 16;
            size_t in_tile_row = r % 16;
            size_t in_tile_col = c % 16;
            
            size_t tile_idx = tile_row * tiles_per_row + tile_col;
            size_t dst_idx = tile_idx * 16 + in_tile_row;
            
            // Set the trit in the destination
            dst[dst_idx] |= ((uint32_t)trit << (in_tile_col * 2));
        }
    }
}

/* ============================================================================
 * SME Kernel Implementations
 * ============================================================================
 *
 * On systems without SME support (or when compiling without SME intrinsics),
 * we fall back to the reference implementations.
 *
 * When SME is available, we use the ARM ACLE SME intrinsics.
 */

#if defined(__ARM_FEATURE_SME) && __ARM_FEATURE_SME

#include <arm_sme.h>

/* 
 * SME streaming mode kernel for 16-element dot product.
 * Uses predicate-driven accumulation to avoid unpacking trits.
 */
__arm_locally_streaming __arm_new("za")
float sme_dot16_sme_impl(const float* activations, uint32_t weights) {
    // Generate predicates for +1 and -1 weights
    // P_pos: true where weight == 0b01 (+1)
    // P_neg: true where weight == 0b10 (-1)
    
    svbool_t p_all = svptrue_b32();
    
    // Load activations as SVE vector (16 x float32)
    svfloat32_t act = svld1_f32(p_all, activations);
    
    // Broadcast weights to SVE vector and extract individual trits
    // We need to compare each lane's corresponding 2 bits
    
    // Create lane indices: 0, 1, 2, ..., 15
    svuint32_t indices = svindex_u32(0, 1);
    
    // Calculate bit shift for each lane: 0, 2, 4, ..., 30
    svuint32_t shifts = svlsl_n_u32_x(p_all, indices, 1);
    
    // Broadcast weights and shift/mask to get each trit
    svuint32_t w_broadcast = svdup_u32(weights);
    svuint32_t trits = svand_n_u32_x(p_all, svlsr_u32_x(p_all, w_broadcast, shifts), 0x3);
    
    // Generate predicates
    svbool_t p_pos = svcmpeq_n_u32(p_all, trits, 1);  // +1
    svbool_t p_neg = svcmpeq_n_u32(p_all, trits, 2);  // -1
    
    // Sum positives (where weight == +1)
    float sum_pos = svaddv_f32(p_pos, act);
    
    // Sum negatives (where weight == -1)  
    float sum_neg = svaddv_f32(p_neg, act);
    
    return sum_pos - sum_neg;
}

/*
 * SME streaming mode kernel for matrix-vector multiplication.
 * Processes 16x16 tiles using ZA accumulator.
 */
__arm_locally_streaming __arm_new("za")
void sme_matvec_sme_impl(float* output, const uint32_t* weights, const float* input,
                         size_t M, size_t K) {
    size_t M_aligned = (M + 15) & ~15;
    size_t K_aligned = (K + 15) & ~15;
    size_t tiles_per_row = K_aligned / 16;
    
    svbool_t p_all = svptrue_b32();
    
    // Process in row-tiles (16 rows at a time)
    for (size_t tile_row = 0; tile_row < M_aligned / 16; tile_row++) {
        // Zero the ZA accumulator for this row-tile
        svzero_za();
        
        // Accumulate across column-tiles
        for (size_t tile_col = 0; tile_col < tiles_per_row; tile_col++) {
            size_t tile_idx = tile_row * tiles_per_row + tile_col;
            const uint32_t* tile_weights = weights + tile_idx * 16;
            
            // Load 16 input values for this column tile
            const float* inp_ptr = input + tile_col * 16;
            svfloat32_t inp = svld1_f32(p_all, inp_ptr);
            
            // Create lane indices for trit extraction
            svuint32_t indices = svindex_u32(0, 1);
            svuint32_t shifts = svlsl_n_u32_x(p_all, indices, 1);
            
            // Process each row in the tile
            for (int r = 0; r < 16; r++) {
                uint32_t row_weights = tile_weights[r];
                
                // Extract trits for this row
                svuint32_t w_broadcast = svdup_u32(row_weights);
                svuint32_t trits = svand_n_u32_x(p_all, 
                    svlsr_u32_x(p_all, w_broadcast, shifts), 0x3);
                
                // Generate predicates
                svbool_t p_pos = svcmpeq_n_u32(p_all, trits, 1);
                svbool_t p_neg = svcmpeq_n_u32(p_all, trits, 2);
                
                // Accumulate: positive weights add, negative weights subtract
                float sum_pos = svaddv_f32(p_pos, inp);
                float sum_neg = svaddv_f32(p_neg, inp);
                
                // Store to ZA (we're treating ZA as a register file here)
                // In a more optimized version, we'd use FMOPA directly
                size_t out_row = tile_row * 16 + r;
                if (out_row < M) {
                    output[out_row] += sum_pos - sum_neg;
                }
            }
        }
    }
}

/* SME wrapper functions */
float sme_dot16(const float* activations, uint32_t weights) {
    return sme_dot16_sme_impl(activations, weights);
}

void sme_matvec(float* output, const uint32_t* weights, const float* input,
                size_t M, size_t K) {
    // Zero output
    memset(output, 0, M * sizeof(float));
    sme_matvec_sme_impl(output, weights, input, M, K);
}

#else /* No SME support - use reference implementations */

float sme_dot16(const float* activations, uint32_t weights) {
    return sme_dot16_ref(activations, weights);
}

void sme_matvec(float* output, const uint32_t* weights, const float* input,
                size_t M, size_t K) {
    memset(output, 0, M * sizeof(float));
    sme_matvec_ref(output, weights, input, M, K);
}

#endif /* __ARM_FEATURE_SME */

/* ============================================================================
 * Batched Operations
 * ============================================================================ */

void sme_matvec_batch(float* output, const uint32_t* weights, const float* input,
                      size_t M, size_t K, size_t batch_size) {
    // Process each batch element
    // TODO: Optimize by processing multiple batch elements in parallel with FMOPA
    for (size_t b = 0; b < batch_size; b++) {
        sme_matvec(output + b * M, weights, input + b * K, M, K);
    }
}

void sme_matmul(float* C, const float* A, const uint32_t* W,
                size_t M, size_t K, size_t N) {
    // Full matrix-matrix multiplication
    // C[M,N] = A[M,K] * W[K,N]
    //
    // Treat as M independent vector-matrix products
    // TODO: Optimize with proper tiling and ZA accumulation
    
    // Zero output
    memset(C, 0, M * N * sizeof(float));
    
    // For now, treat each row of A as a vector and multiply by W
    // This is not optimal but correct
    for (size_t m = 0; m < M; m++) {
        // C[m,:] = A[m,:] * W
        // This is a 1xK vector times KxN matrix = 1xN vector
        
        // Actually we need to transpose the operation conceptually
        // C[m,n] = sum_k A[m,k] * W[k,n]
        
        // Process in tiles of 16
        size_t K_aligned = (K + 15) & ~15;
        size_t N_aligned = (N + 15) & ~15;
        
        for (size_t n_tile = 0; n_tile < N_aligned / 16; n_tile++) {
            float tile_out[16] = {0};
            
            for (size_t k_tile = 0; k_tile < K_aligned / 16; k_tile++) {
                // Load input activations for this K tile
                float inp[16];
                for (int i = 0; i < 16; i++) {
                    size_t k = k_tile * 16 + i;
                    inp[i] = (k < K) ? A[m * K + k] : 0.0f;
                }
                
                // Get weight tile: rows k_tile*16..(k_tile+1)*16, cols n_tile*16..(n_tile+1)*16
                // Weight layout: tiles are stored in row-major order within the weight matrix
                // But our sme_pack_weights uses (row, col) as (M, K) so we need (K, N) here
                
                // For matmul, weights are KxN, so:
                size_t tiles_per_row_w = N_aligned / 16;
                size_t tile_idx = k_tile * tiles_per_row_w + n_tile;
                const uint32_t* tile_w = W + tile_idx * 16;
                
                // Each row of tile_w corresponds to one K value
                // Each column of tile_w (bit pair) corresponds to one N value
                for (int k_in_tile = 0; k_in_tile < 16; k_in_tile++) {
                    uint32_t row_weights = tile_w[k_in_tile];
                    float a_val = inp[k_in_tile];
                    
                    for (int n_in_tile = 0; n_in_tile < 16; n_in_tile++) {
                        int w = decode_trit((row_weights >> (n_in_tile * 2)) & 0x3);
                        tile_out[n_in_tile] += a_val * (float)w;
                    }
                }
            }
            
            // Store output tile
            for (int i = 0; i < 16; i++) {
                size_t n = n_tile * 16 + i;
                if (n < N) {
                    C[m * N + n] = tile_out[i];
                }
            }
        }
    }
}
