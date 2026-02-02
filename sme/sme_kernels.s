/*
 * sme_kernels.s - ARM64 assembly kernels for SME on Apple M4
 *
 * These kernels implement ternary matrix operations using SME streaming mode.
 * Key insight: SME mode transitions (SMSTART/SMSTOP) are expensive (~50 cycles).
 * Batch operations to amortize this cost.
 *
 * IMPORTANT: NEON/FP registers are NOT preserved across mode transitions.
 * SMSTART SM zeroes all Z/V/FP registers (including callee-saved q8-q15).
 * SMSTOP SM zeroes them again. We must save q8-q15 before SMSTART and
 * restore after SMSTOP to comply with AAPCS64 calling convention.
 *
 * Copyright 2026 Trix Research
 */

.global _sme_dot16_asm
.global _sme_dot16_batch_asm
.global _sme_matvec_16x16_asm
.global _sme_matvec_32x32_asm
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
 */
_sme_dot16_asm:
    /* Save callee-saved NEON regs (destroyed by smstart/smstop) */
    stp q8, q9, [sp, #-128]!
    stp q10, q11, [sp, #32]
    stp q12, q13, [sp, #64]
    stp q14, q15, [sp, #96]

    smstart sm

    mov x8, #16
    whilelo p0.s, xzr, x8

    ld1w {z0.s}, p0/z, [x0]

    index z1.s, #0, #1
    lsl z1.s, z1.s, #1

    mov z2.s, w1
    lsrr z1.s, p0/m, z1.s, z2.s
    and z1.s, z1.s, #3

    cmpeq p1.s, p0/z, z1.s, #1
    cmpeq p2.s, p0/z, z1.s, #2

    faddv s1, p1, z0.s
    faddv s2, p2, z0.s
    fsub s0, s1, s2

    fmov w9, s0

    smstop sm

    /* Restore callee-saved NEON regs */
    ldp q14, q15, [sp, #96]
    ldp q12, q13, [sp, #64]
    ldp q10, q11, [sp, #32]
    ldp q8, q9, [sp], #128

    fmov s0, w9
    ret


/*
 * void sme_dot16_batch_asm(float* results, const float* activations,
 *                          const uint32_t* weights, uint32_t count)
 *
 * Computes 'count' ternary dot products in a single streaming session.
 *
 * Inputs:
 *   x0 = output array (count floats)
 *   x1 = input activations (count * 16 floats)
 *   x2 = packed weights (count uint32s)
 *   w3 = number of dot products
 */
_sme_dot16_batch_asm:
    cbz w3, .Lbatch_done

    stp x19, x20, [sp, #-160]!
    stp x21, x22, [sp, #16]
    stp q8, q9, [sp, #32]
    stp q10, q11, [sp, #64]
    stp q12, q13, [sp, #96]
    stp q14, q15, [sp, #128]

    mov x19, x0
    mov x20, x1
    mov x21, x2
    mov w22, w3

    smstart sm

    mov x8, #16
    whilelo p0.s, xzr, x8

.Lbatch_loop:
    ld1w {z0.s}, p0/z, [x20]
    ldr w9, [x21]

    index z1.s, #0, #1
    lsl z1.s, z1.s, #1
    mov z2.s, w9
    lsrr z1.s, p0/m, z1.s, z2.s
    and z1.s, z1.s, #3

    cmpeq p1.s, p0/z, z1.s, #1
    cmpeq p2.s, p0/z, z1.s, #2

    faddv s1, p1, z0.s
    faddv s2, p2, z0.s
    fsub s0, s1, s2

    str s0, [x19]

    add x20, x20, #64
    add x21, x21, #4
    add x19, x19, #4

    subs w22, w22, #1
    b.ne .Lbatch_loop

    smstop sm

    ldp q14, q15, [sp, #128]
    ldp q12, q13, [sp, #96]
    ldp q10, q11, [sp, #64]
    ldp q8, q9, [sp, #32]
    ldp x21, x22, [sp, #16]
    ldp x19, x20, [sp], #160

.Lbatch_done:
    ret


/*
 * void sme_matvec_16x16_asm(float* output, const uint32_t* weights,
 *                           const float* input)
 *
 * Computes 16x16 matrix-vector multiplication with ternary weights.
 * All 16 rows processed in a single streaming session.
 *
 * Inputs:
 *   x0 = output array (16 floats)
 *   x1 = weights (16 uint32s, each with 16 x 2-bit trits)
 *   x2 = input vector (16 floats)
 */
_sme_matvec_16x16_asm:
    stp x19, x20, [sp, #-160]!
    stp x21, x22, [sp, #16]
    stp q8, q9, [sp, #32]
    stp q10, q11, [sp, #64]
    stp q12, q13, [sp, #96]
    stp q14, q15, [sp, #128]

    mov x19, x0
    mov x20, x1
    mov x21, x2
    mov w22, #16

    smstart sm

    mov x8, #16
    whilelo p0.s, xzr, x8

    ld1w {z3.s}, p0/z, [x21]

    index z4.s, #0, #1
    lsl z4.s, z4.s, #1

.Lmv_loop:
    ldr w9, [x20]

    mov z2.s, w9
    mov z1.d, z4.d
    lsrr z1.s, p0/m, z1.s, z2.s
    and z1.s, z1.s, #3

    cmpeq p1.s, p0/z, z1.s, #1
    cmpeq p2.s, p0/z, z1.s, #2

    faddv s1, p1, z3.s
    faddv s2, p2, z3.s
    fsub s0, s1, s2

    str s0, [x19]

    add x20, x20, #4
    add x19, x19, #4

    subs w22, w22, #1
    b.ne .Lmv_loop

    smstop sm

    ldp q14, q15, [sp, #128]
    ldp q12, q13, [sp, #96]
    ldp q10, q11, [sp, #64]
    ldp q8, q9, [sp, #32]
    ldp x21, x22, [sp, #16]
    ldp x19, x20, [sp], #160
    ret


/*
 * void sme_matvec_32x32_asm(float* output, const uint32_t* weights,
 *                           const float* input)
 *
 * Computes 32x32 matrix-vector multiplication with ternary weights.
 * Tiled as 2x2 grid of 16x16 tiles, single streaming session.
 *
 * Weight layout: [T00 T01 T10 T11], each tile 16 uint32s.
 *   T00 = rows 0-15,  cols 0-15
 *   T01 = rows 0-15,  cols 16-31
 *   T10 = rows 16-31, cols 0-15
 *   T11 = rows 16-31, cols 16-31
 *
 * Register plan:
 *   z3 = input[0:15]   (persistent, accessed via z6 copy to avoid
 *        scalar write clobber — faddv scalar dest zeroes full Z reg)
 *   z4 = shift pattern  (persistent)
 *   z5 = input[16:31]  (persistent, same z6 copy rule)
 *   z0, z1, z2, z6 = temporaries
 *   p0 = all-16 predicate (persistent)
 *   p1, p2 = +1/-1 predicates (per-row)
 */
_sme_matvec_32x32_asm:
    stp x19, x20, [sp, #-176]!
    stp x21, x22, [sp, #16]
    stp x23, x24, [sp, #32]
    stp q8, q9, [sp, #48]
    stp q10, q11, [sp, #80]
    stp q12, q13, [sp, #112]
    stp q14, q15, [sp, #144]

    mov x19, x0                     // output
    mov x20, x1                     // weights base (T00)
    mov x21, x2                     // input

    smstart sm

    mov x8, #16
    whilelo p0.s, xzr, x8

    // Load both input halves — persistent across all 32 rows
    ld1w {z3.s}, p0/z, [x21]        // input[0:15]
    add x9, x21, #64
    ld1w {z5.s}, p0/z, [x9]         // input[16:31]

    // Pre-compute shift pattern — persistent
    index z4.s, #0, #1
    lsl z4.s, z4.s, #1

    // ─── Rows 0-15: T00 (left) + T01 (right) ───
    add x23, x20, #64               // T01
    mov w22, #16

.Lmv32_upper_loop:
    // Left half: dot16(input[0:15], T00[row])
    ldr w9, [x20]
    mov z2.s, w9
    mov z1.d, z4.d
    lsrr z1.s, p0/m, z1.s, z2.s
    and z1.s, z1.s, #3
    cmpeq p1.s, p0/z, z1.s, #1
    cmpeq p2.s, p0/z, z1.s, #2
    mov z6.d, z3.d                  // Copy input left (protect z3)
    faddv s0, p1, z6.s
    mov z6.d, z3.d                  // Re-copy
    faddv s1, p2, z6.s
    fsub s0, s0, s1
    fmov w10, s0                    // Save partial_left to GPR

    // Right half: dot16(input[16:31], T01[row])
    ldr w9, [x23]
    mov z2.s, w9
    mov z1.d, z4.d
    lsrr z1.s, p0/m, z1.s, z2.s
    and z1.s, z1.s, #3
    cmpeq p1.s, p0/z, z1.s, #1
    cmpeq p2.s, p0/z, z1.s, #2
    mov z6.d, z5.d                  // Copy input right (protect z5)
    faddv s0, p1, z6.s
    mov z6.d, z5.d
    faddv s1, p2, z6.s
    fsub s0, s0, s1

    // Sum and store
    fmov s1, w10
    fadd s0, s1, s0
    str s0, [x19]

    add x20, x20, #4
    add x23, x23, #4
    add x19, x19, #4
    subs w22, w22, #1
    b.ne .Lmv32_upper_loop

    // ─── Rows 16-31: T10 (left) + T11 (right) ───
    mov x20, x1
    add x20, x20, #128              // T10 = weights + 32*4
    add x23, x20, #64               // T11 = T10 + 16*4
    mov w22, #16

.Lmv32_lower_loop:
    ldr w9, [x20]
    mov z2.s, w9
    mov z1.d, z4.d
    lsrr z1.s, p0/m, z1.s, z2.s
    and z1.s, z1.s, #3
    cmpeq p1.s, p0/z, z1.s, #1
    cmpeq p2.s, p0/z, z1.s, #2
    mov z6.d, z3.d
    faddv s0, p1, z6.s
    mov z6.d, z3.d
    faddv s1, p2, z6.s
    fsub s0, s0, s1
    fmov w10, s0

    ldr w9, [x23]
    mov z2.s, w9
    mov z1.d, z4.d
    lsrr z1.s, p0/m, z1.s, z2.s
    and z1.s, z1.s, #3
    cmpeq p1.s, p0/z, z1.s, #1
    cmpeq p2.s, p0/z, z1.s, #2
    mov z6.d, z5.d
    faddv s0, p1, z6.s
    mov z6.d, z5.d
    faddv s1, p2, z6.s
    fsub s0, s0, s1

    fmov s1, w10
    fadd s0, s1, s0
    str s0, [x19]

    add x20, x20, #4
    add x23, x23, #4
    add x19, x19, #4
    subs w22, w22, #1
    b.ne .Lmv32_lower_loop

    smstop sm

    ldp q14, q15, [sp, #144]
    ldp q12, q13, [sp, #112]
    ldp q10, q11, [sp, #80]
    ldp q8, q9, [sp, #48]
    ldp x23, x24, [sp, #32]
    ldp x21, x22, [sp, #16]
    ldp x19, x20, [sp], #176
    ret
