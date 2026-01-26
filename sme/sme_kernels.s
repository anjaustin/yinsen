/*
 * sme_kernels.s - ARM64 assembly kernels for SME on Apple M4
 *
 * These kernels implement ternary matrix operations using SME streaming mode.
 * Key insight: SME mode transitions (SMSTART/SMSTOP) are expensive (~50 cycles).
 * Batch operations to amortize this cost.
 *
 * IMPORTANT: NEON/FP registers are NOT preserved across mode transitions.
 * Results must be saved to GPRs or memory before SMSTOP.
 *
 * Copyright 2026 Trix Research
 */

.global _sme_dot16_asm
.global _sme_dot16_batch_asm
.global _sme_matvec_16x16_asm
.p2align 4

/*
 * float sme_dot16_asm(const float* activations, uint32_t weights)
 *
 * Computes ternary dot product of 16 elements.
 * Weight encoding: 0b01=+1, 0b10=-1, 0b00/0b11=0
 *
 * Inputs:
 *   x0 = pointer to 16 floats (64-byte aligned preferred)
 *   w1 = packed weights (16 x 2-bit trits)
 *
 * Returns:
 *   s0 = dot product result
 *
 * Note: This function has ~65ns overhead from mode transitions.
 * Use sme_dot16_batch_asm for better throughput.
 */
_sme_dot16_asm:
    smstart sm                      // Enter streaming mode
    
    mov x8, #16
    whilelo p0.s, xzr, x8           // p0 = predicate for 16 elements
    
    ld1w {z0.s}, p0/z, [x0]         // Load 16 floats
    
    index z1.s, #0, #1              // z1 = {0, 1, 2, ..., 15}
    lsl z1.s, z1.s, #1              // z1 = {0, 2, 4, ..., 30} bit positions
    
    mov z2.s, w1                    // Broadcast weights
    lsrr z1.s, p0/m, z1.s, z2.s     // Extract trits
    and z1.s, z1.s, #3
    
    cmpeq p1.s, p0/z, z1.s, #1      // p1 = where weight is +1
    cmpeq p2.s, p0/z, z1.s, #2      // p2 = where weight is -1
    
    faddv s1, p1, z0.s              // Sum positives
    faddv s2, p2, z0.s              // Sum negatives
    fsub s0, s1, s2                 // Result = pos - neg
    
    fmov w9, s0                     // Save to GPR before mode exit
    
    smstop sm                       // Exit streaming mode
    
    fmov s0, w9                     // Return result
    ret


/*
 * void sme_dot16_batch_asm(float* results, const float* activations, 
 *                          const uint32_t* weights, uint32_t count)
 *
 * Computes 'count' ternary dot products in a single streaming session.
 * Much faster than calling sme_dot16_asm in a loop.
 *
 * Inputs:
 *   x0 = output array (count floats)
 *   x1 = input activations (count * 16 floats, 64-byte aligned preferred)
 *   x2 = packed weights (count uint32s)
 *   w3 = number of dot products
 *
 * Throughput: ~14ns per dot16 (vs 65ns single)
 */
_sme_dot16_batch_asm:
    cbz w3, .Lbatch_done            // Early exit if count == 0
    
    stp x19, x20, [sp, #-32]!
    stp x21, x22, [sp, #16]
    
    mov x19, x0                     // results
    mov x20, x1                     // activations
    mov x21, x2                     // weights
    mov w22, w3                     // count
    
    smstart sm
    
    mov x8, #16
    whilelo p0.s, xzr, x8           // p0 = first 16 elements
    
.Lbatch_loop:
    ld1w {z0.s}, p0/z, [x20]        // Load 16 floats
    ldr w9, [x21]                   // Load weight
    
    // Extract and compare trits
    index z1.s, #0, #1
    lsl z1.s, z1.s, #1
    mov z2.s, w9
    lsrr z1.s, p0/m, z1.s, z2.s
    and z1.s, z1.s, #3
    
    cmpeq p1.s, p0/z, z1.s, #1      // +1 positions
    cmpeq p2.s, p0/z, z1.s, #2      // -1 positions
    
    faddv s1, p1, z0.s
    faddv s2, p2, z0.s
    fsub s0, s1, s2
    
    str s0, [x19]                   // Store result
    
    add x20, x20, #64               // Next 16 floats
    add x21, x21, #4                // Next weight
    add x19, x19, #4                // Next result
    
    subs w22, w22, #1
    b.ne .Lbatch_loop
    
    smstop sm
    
    ldp x21, x22, [sp, #16]
    ldp x19, x20, [sp], #32
    
.Lbatch_done:
    ret


/*
 * void sme_matvec_16x16_asm(float* output, const uint32_t* weights,
 *                           const float* input)
 *
 * Computes 16x16 matrix-vector multiplication with ternary weights.
 * output[i] = sum_j(decode(weights[i,j]) * input[j])
 *
 * Inputs:
 *   x0 = output array (16 floats)
 *   x1 = weights (16 uint32s, each with 16 x 2-bit trits)
 *   x2 = input vector (16 floats)
 *
 * This processes all 16 rows in a single streaming session.
 */
_sme_matvec_16x16_asm:
    stp x19, x20, [sp, #-32]!
    stp x21, x22, [sp, #16]
    
    mov x19, x0                     // output
    mov x20, x1                     // weights
    mov x21, x2                     // input
    mov w22, #16                    // row count
    
    smstart sm
    
    mov x8, #16
    whilelo p0.s, xzr, x8
    
    // Load input vector once
    ld1w {z3.s}, p0/z, [x21]
    
    // Pre-compute index and shift vectors
    index z4.s, #0, #1              // {0,1,2,...,15}
    lsl z4.s, z4.s, #1              // {0,2,4,...,30}
    
.Lmv_loop:
    ldr w9, [x20]                   // Load row weights
    
    mov z2.s, w9                    // Broadcast
    mov z1.d, z4.d                  // Copy shift pattern (use .d for full copy)
    lsrr z1.s, p0/m, z1.s, z2.s     // Extract trits
    and z1.s, z1.s, #3
    
    cmpeq p1.s, p0/z, z1.s, #1
    cmpeq p2.s, p0/z, z1.s, #2
    
    faddv s1, p1, z3.s              // Sum where +1
    faddv s2, p2, z3.s              // Sum where -1
    fsub s0, s1, s2
    
    str s0, [x19]                   // Store output row
    
    add x20, x20, #4                // Next row weights
    add x19, x19, #4                // Next output
    
    subs w22, w22, #1
    b.ne .Lmv_loop
    
    smstop sm
    
    ldp x21, x22, [sp, #16]
    ldp x19, x20, [sp], #32
    ret
