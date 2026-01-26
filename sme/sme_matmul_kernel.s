/*
 * sme_matmul_kernel.s - Ternary MatMul using SVE (Z registers only)
 *
 * THE COMBUSTION CHAMBER (macOS-compatible version)
 *
 * On real SME hardware, we'd use:
 *   zero {za0.s}
 *   fmopa za0.s, p1/m, p1/m, z0.s, z31.s
 *
 * But macOS BLOCKS ZA tile access, so we accumulate in Z registers instead.
 * This loses the ZA/Z parallelism but still gets 512-bit SIMD in streaming mode.
 *
 * Strategy: "Predicate Switchboard"
 * - Load 16 weights in one 32-bit int (16 x 2-bit trits)
 * - Use shift vector to extract each trit
 * - Generate predicates: P1 = +1 lanes, P2 = -1 lanes
 * - Accumulate: acc += act where P1, acc -= act where P2
 *
 * Weight encoding: 0b01 = +1, 0b10 = -1, 0b00/0b11 = 0
 *
 * Copyright 2026 Trix Research
 */

.global _sme_ternary_matmul_16x16
.global _sme_ternary_matmul_16xK
.p2align 4

/*
 * void sme_ternary_matmul_16x16(
 *     float* out,             // x0: Output [16, 16] - 16 output channels x 16 tokens
 *     const float* act,       // x1: Activations [16, 16] - column-major (K=16, Tokens=16)
 *     const uint32_t* wgt,    // x2: Packed weights [16] - 16 vertical slices
 *     int K                   // w3: Input channels (must be 16 for this version)
 * )
 *
 * Computes: out[oc, tok] = sum_k(act[k, tok] * decode(wgt[k, oc]))
 *
 * Memory layout:
 *   act: Column-major [K, 16] - act[k*16 + tok] = activation for channel k, token tok
 *   wgt: Each uint32 contains 16 trits for one input channel across all 16 output channels
 *   out: Row-major [16, 16] - out[oc*16 + tok]
 */
_sme_ternary_matmul_16x16:
    stp x19, x20, [sp, #-64]!
    stp x21, x22, [sp, #16]
    stp x23, x24, [sp, #32]
    stp x25, x26, [sp, #48]
    
    mov x19, x0                     // out
    mov x20, x1                     // act
    mov x21, x2                     // wgt
    mov w22, w3                     // K
    
    smstart sm
    
    // Setup predicates and constants
    mov x8, #16
    whilelo p0.s, xzr, x8           // All 16 lanes active
    
    // Shift vector for bit extraction: {0, 2, 4, ..., 30}
    index z30.s, #0, #2
    
    // Zero the accumulators (16 output channels x 16 tokens)
    // We use z16-z31 as accumulators for 16 output channels
    // But we only have 32 Z registers, and we need working regs too
    // So we accumulate one output channel at a time and store
    
    // Actually, let's do it differently:
    // For each output channel (oc = 0..15):
    //   acc = 0
    //   for each input channel (k = 0..K-1):
    //     load act[k, :] (16 tokens)
    //     extract weight[k, oc] 
    //     if +1: acc += act
    //     if -1: acc -= act
    //   store out[oc, :]
    
    mov w23, #0                     // oc = 0 (output channel counter)
    
.Loc_loop:
    // Zero accumulator for this output channel
    mov z0.s, #0                    // acc for 16 tokens
    
    // Reset pointers for this output channel
    mov x24, x20                    // act_ptr = act base
    mov x25, x21                    // wgt_ptr = wgt base
    mov w26, w22                    // k = K
    
.Lk_loop:
    // A. Load activations for this input channel (16 tokens)
    ld1w {z1.s}, p0/z, [x24]
    add x24, x24, #64               // Next K row (16 floats = 64 bytes)
    
    // B. Load packed weights (16 output channels in one uint32)
    ldr w10, [x25]
    add x25, x25, #4                // Next weight row
    
    // C. Extract the trit for THIS output channel (oc)
    //    trit = (packed >> (oc * 2)) & 3
    lsl w11, w23, #1                // w11 = oc * 2
    lsr w10, w10, w11               // w10 = packed >> (oc * 2)
    and w10, w10, #3                // w10 = trit for this oc
    
    // D. Branch based on trit value
    cmp w10, #1
    b.eq .Ladd_act
    cmp w10, #2
    b.eq .Lsub_act
    b .Lnext_k                      // trit = 0 or 3, skip
    
.Ladd_act:
    fadd z0.s, z0.s, z1.s           // acc += act
    b .Lnext_k
    
.Lsub_act:
    fsub z0.s, z0.s, z1.s           // acc -= act
    
.Lnext_k:
    subs w26, w26, #1
    b.ne .Lk_loop
    
    // Store result for this output channel
    // out[oc * 16 : oc * 16 + 16]
    lsl x10, x23, #6                // x10 = oc * 64 (bytes)
    add x10, x19, x10
    st1w {z0.s}, p0, [x10]
    
    // Next output channel
    add w23, w23, #1
    cmp w23, #16
    b.lt .Loc_loop
    
    smstop sm
    
    ldp x25, x26, [sp, #48]
    ldp x23, x24, [sp, #32]
    ldp x21, x22, [sp, #16]
    ldp x19, x20, [sp], #64
    ret


/*
 * Optimized version: Process all 16 output channels in parallel using predicates
 * This is closer to the original "Combustion Chamber" design but uses Z accumulators
 */
_sme_ternary_matmul_16xK:
    stp x19, x20, [sp, #-64]!
    stp x21, x22, [sp, #16]
    stp x23, x24, [sp, #32]
    stp x25, x26, [sp, #48]
    
    mov x19, x0                     // out
    mov x20, x1                     // act  
    mov x21, x2                     // wgt
    mov w22, w3                     // K
    
    cbz w22, .Learly_exit           // K == 0, nothing to do
    
    smstart sm
    
    // Predicates
    mov x8, #16
    whilelo p0.s, xzr, x8
    
    // Shift vector: {0, 2, 4, ..., 30}
    index z30.s, #0, #2
    
    // Constant 1.0 for masking
    fmov z31.s, #1.0
    
    // Zero accumulators for all 16 output channels
    // z0-z15 will hold accumulators for output channels 0-15
    mov z0.s, #0
    mov z1.s, #0
    mov z2.s, #0
    mov z3.s, #0
    mov z4.s, #0
    mov z5.s, #0
    mov z6.s, #0
    mov z7.s, #0
    mov z8.s, #0
    mov z9.s, #0
    mov z10.s, #0
    mov z11.s, #0
    mov z12.s, #0
    mov z13.s, #0
    mov z14.s, #0
    mov z15.s, #0
    
    // Main K loop
.Lk_loop_par:
    // A. Load activations (16 tokens for this input channel k)
    ld1w {z16.s}, p0/z, [x20]
    add x20, x20, #64
    
    // B. Load packed weights and broadcast
    ld1rw {z17.s}, p0/z, [x21]
    add x21, x21, #4
    
    // C. Extract all 16 trits in parallel
    //    z18[i] = (z17 >> (i*2)) & 3
    //    Use LSRR (reverse shift) since we want z17 >> z30
    mov z18.d, z30.d                // Copy shift amounts
    lsrr z18.s, p0/m, z18.s, z17.s  // z18 = z17 >> z30 (element-wise)
    and z18.s, z18.s, #3            // Mask to 2 bits
    
    // D. Generate predicates for +1 and -1
    mov z19.s, #1
    mov z20.s, #2
    cmpeq p1.s, p0/z, z18.s, z19.s  // P1 = where trit == 1 (+1)
    cmpeq p2.s, p0/z, z18.s, z20.s  // P2 = where trit == 2 (-1)
    
    // E. Create masked activation vectors
    //    For +1 channels: use activation as-is
    //    For -1 channels: negate activation
    //    For 0 channels: zero
    
    // Positive contribution: act where P1, else 0
    mov z21.s, #0
    sel z21.s, p1, z16.s, z21.s
    
    // Negative contribution: -act where P2, else 0
    fneg z22.s, p0/m, z16.s
    mov z23.s, #0
    sel z23.s, p2, z22.s, z23.s
    
    // F. Accumulate into each output channel
    //    This is the tricky part - we need to scatter the activation
    //    to different output channels based on which lanes have +1/-1
    //
    //    The issue: z16 has 16 TOKEN values, but we need to add them
    //    to 16 different OUTPUT CHANNEL accumulators based on the weight.
    //
    //    In ZA mode: fmopa does outer product, handling this naturally
    //    In Z-only mode: we need a different approach
    
    // For each output channel, we need to reduce z16 based on weight[k, oc]
    // This requires extracting weight bits and doing conditional adds
    
    // Actually, let's reconsider the data layout...
    // The original kernel does: ZA += outer_product(act, weights)
    // Where act is [16 tokens] and weights is [16 output channels]
    //
    // For Z-only: we accumulate each output channel separately
    // acc[oc] += act * weight[oc]  (where weight[oc] is +1, -1, or 0)
    //
    // Since weight[oc] is scalar, this is: acc[oc] += act * scalar
    // which is a simple vector-scalar multiply-add
    
    // Extract weight for each output channel and accumulate
    // z18 already has the decoded trits for all 16 output channels
    
    // For oc=0: weight is in z18.s[0]
    // We need to broadcast z18.s[0] and multiply with z16
    
    // Use DUP to broadcast each element and accumulate
    // This is 16 iterations but each is simple
    
    // Output channel 0
    dup z24.s, z18.s[0]
    cmpeq p3.s, p0/z, z24.s, z19.s
    cmpeq p4.s, p0/z, z24.s, z20.s
    fadd z0.s, p3/m, z0.s, z16.s
    fsub z0.s, p4/m, z0.s, z16.s
    
    // Output channel 1
    dup z24.s, z18.s[1]
    cmpeq p3.s, p0/z, z24.s, z19.s
    cmpeq p4.s, p0/z, z24.s, z20.s
    fadd z1.s, p3/m, z1.s, z16.s
    fsub z1.s, p4/m, z1.s, z16.s
    
    // Output channel 2
    dup z24.s, z18.s[2]
    cmpeq p3.s, p0/z, z24.s, z19.s
    cmpeq p4.s, p0/z, z24.s, z20.s
    fadd z2.s, p3/m, z2.s, z16.s
    fsub z2.s, p4/m, z2.s, z16.s
    
    // Output channel 3
    dup z24.s, z18.s[3]
    cmpeq p3.s, p0/z, z24.s, z19.s
    cmpeq p4.s, p0/z, z24.s, z20.s
    fadd z3.s, p3/m, z3.s, z16.s
    fsub z3.s, p4/m, z3.s, z16.s
    
    // Output channel 4
    dup z24.s, z18.s[4]
    cmpeq p3.s, p0/z, z24.s, z19.s
    cmpeq p4.s, p0/z, z24.s, z20.s
    fadd z4.s, p3/m, z4.s, z16.s
    fsub z4.s, p4/m, z4.s, z16.s
    
    // Output channel 5
    dup z24.s, z18.s[5]
    cmpeq p3.s, p0/z, z24.s, z19.s
    cmpeq p4.s, p0/z, z24.s, z20.s
    fadd z5.s, p3/m, z5.s, z16.s
    fsub z5.s, p4/m, z5.s, z16.s
    
    // Output channel 6
    dup z24.s, z18.s[6]
    cmpeq p3.s, p0/z, z24.s, z19.s
    cmpeq p4.s, p0/z, z24.s, z20.s
    fadd z6.s, p3/m, z6.s, z16.s
    fsub z6.s, p4/m, z6.s, z16.s
    
    // Output channel 7
    dup z24.s, z18.s[7]
    cmpeq p3.s, p0/z, z24.s, z19.s
    cmpeq p4.s, p0/z, z24.s, z20.s
    fadd z7.s, p3/m, z7.s, z16.s
    fsub z7.s, p4/m, z7.s, z16.s
    
    // Output channel 8
    dup z24.s, z18.s[8]
    cmpeq p3.s, p0/z, z24.s, z19.s
    cmpeq p4.s, p0/z, z24.s, z20.s
    fadd z8.s, p3/m, z8.s, z16.s
    fsub z8.s, p4/m, z8.s, z16.s
    
    // Output channel 9
    dup z24.s, z18.s[9]
    cmpeq p3.s, p0/z, z24.s, z19.s
    cmpeq p4.s, p0/z, z24.s, z20.s
    fadd z9.s, p3/m, z9.s, z16.s
    fsub z9.s, p4/m, z9.s, z16.s
    
    // Output channel 10
    dup z24.s, z18.s[10]
    cmpeq p3.s, p0/z, z24.s, z19.s
    cmpeq p4.s, p0/z, z24.s, z20.s
    fadd z10.s, p3/m, z10.s, z16.s
    fsub z10.s, p4/m, z10.s, z16.s
    
    // Output channel 11
    dup z24.s, z18.s[11]
    cmpeq p3.s, p0/z, z24.s, z19.s
    cmpeq p4.s, p0/z, z24.s, z20.s
    fadd z11.s, p3/m, z11.s, z16.s
    fsub z11.s, p4/m, z11.s, z16.s
    
    // Output channel 12
    dup z24.s, z18.s[12]
    cmpeq p3.s, p0/z, z24.s, z19.s
    cmpeq p4.s, p0/z, z24.s, z20.s
    fadd z12.s, p3/m, z12.s, z16.s
    fsub z12.s, p4/m, z12.s, z16.s
    
    // Output channel 13
    dup z24.s, z18.s[13]
    cmpeq p3.s, p0/z, z24.s, z19.s
    cmpeq p4.s, p0/z, z24.s, z20.s
    fadd z13.s, p3/m, z13.s, z16.s
    fsub z13.s, p4/m, z13.s, z16.s
    
    // Output channel 14
    dup z24.s, z18.s[14]
    cmpeq p3.s, p0/z, z24.s, z19.s
    cmpeq p4.s, p0/z, z24.s, z20.s
    fadd z14.s, p3/m, z14.s, z16.s
    fsub z14.s, p4/m, z14.s, z16.s
    
    // Output channel 15
    dup z24.s, z18.s[15]
    cmpeq p3.s, p0/z, z24.s, z19.s
    cmpeq p4.s, p0/z, z24.s, z20.s
    fadd z15.s, p3/m, z15.s, z16.s
    fsub z15.s, p4/m, z15.s, z16.s
    
    // Loop control
    subs w22, w22, #1
    b.ne .Lk_loop_par
    
    // Store all 16 output channels (each is 16 floats = 64 bytes)
    st1w {z0.s}, p0, [x19]
    add x19, x19, #64
    st1w {z1.s}, p0, [x19]
    add x19, x19, #64
    st1w {z2.s}, p0, [x19]
    add x19, x19, #64
    st1w {z3.s}, p0, [x19]
    add x19, x19, #64
    st1w {z4.s}, p0, [x19]
    add x19, x19, #64
    st1w {z5.s}, p0, [x19]
    add x19, x19, #64
    st1w {z6.s}, p0, [x19]
    add x19, x19, #64
    st1w {z7.s}, p0, [x19]
    add x19, x19, #64
    st1w {z8.s}, p0, [x19]
    add x19, x19, #64
    st1w {z9.s}, p0, [x19]
    add x19, x19, #64
    st1w {z10.s}, p0, [x19]
    add x19, x19, #64
    st1w {z11.s}, p0, [x19]
    add x19, x19, #64
    st1w {z12.s}, p0, [x19]
    add x19, x19, #64
    st1w {z13.s}, p0, [x19]
    add x19, x19, #64
    st1w {z14.s}, p0, [x19]
    add x19, x19, #64
    st1w {z15.s}, p0, [x19]
    
    smstop sm
    
.Learly_exit:
    ldp x25, x26, [sp, #48]
    ldp x23, x24, [sp, #32]
    ldp x21, x22, [sp, #16]
    ldp x19, x20, [sp], #64
    ret
