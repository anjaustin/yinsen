/*
 * YINSEN APU - Arithmetic Processing Unit
 *
 * Core primitives for frozen computation.
 * Verified: Logic shapes produce exact truth tables for binary inputs.
 *
 * The 5 Primes: ADD, MUL, EXP, MAX, CONST
 * Everything else is composition.
 */

#ifndef YINSEN_APU_H
#define YINSEN_APU_H

#include <stdint.h>
#include <math.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ============================================================================
 * FROZEN LOGIC SHAPES
 *
 * These are mathematically exact for binary inputs {0, 1}.
 * Verified by exhaustive truth table testing.
 * ============================================================================ */

/* XOR: a ^ b = a + b - 2ab */
static inline float yinsen_xor(float a, float b) {
    return a + b - 2.0f * a * b;
}

/* AND: a & b = ab */
static inline float yinsen_and(float a, float b) {
    return a * b;
}

/* OR: a | b = a + b - ab */
static inline float yinsen_or(float a, float b) {
    return a + b - a * b;
}

/* NOT: ~a = 1 - a */
static inline float yinsen_not(float a) {
    return 1.0f - a;
}

/* NAND: ~(a & b) = 1 - ab */
static inline float yinsen_nand(float a, float b) {
    return 1.0f - a * b;
}

/* NOR: ~(a | b) = 1 - a - b + ab */
static inline float yinsen_nor(float a, float b) {
    return 1.0f - a - b + a * b;
}

/* XNOR: ~(a ^ b) = 1 - a - b + 2ab */
static inline float yinsen_xnor(float a, float b) {
    return 1.0f - a - b + 2.0f * a * b;
}

/* ============================================================================
 * FULL ADDER - Atomic unit of arithmetic
 *
 * sum = a XOR b XOR c
 * carry = (a AND b) OR ((a XOR b) AND c)
 *
 * Polynomial form:
 *   sum = a + b + c - 2ab - 2ac - 2bc + 4abc
 *   carry = ab + ac + bc - 2abc
 *
 * Verified: Exhaustive test of all 8 input combinations.
 * ============================================================================ */

static inline void yinsen_full_adder(
    float a, float b, float c,
    float* sum, float* carry
) {
    /* XOR chain for sum */
    float ab_xor = a + b - 2.0f * a * b;
    *sum = ab_xor + c - 2.0f * ab_xor * c;

    /* AND-OR chain for carry */
    float ab_and = a * b;
    float ab_xor_c = ab_xor * c;
    *carry = ab_and + ab_xor_c - ab_and * ab_xor_c;
}

/* ============================================================================
 * 8-BIT RIPPLE CARRY ADDER
 *
 * Verified: Exhaustive test of all 65,536 input combinations (256 x 256).
 * ============================================================================ */

static inline void yinsen_ripple_add_8bit(
    const float* a,    /* 8 bits, LSB first */
    const float* b,    /* 8 bits, LSB first */
    float c_in,
    float* result,     /* 8 bits, LSB first */
    float* c_out
) {
    float carry = c_in;
    for (int i = 0; i < 8; i++) {
        float sum;
        yinsen_full_adder(a[i], b[i], carry, &sum, &carry);
        result[i] = sum;
    }
    *c_out = carry;
}

/* ============================================================================
 * HAMMING DISTANCE
 *
 * hamming(a, b) = popcount(a XOR b)
 * Used for signature matching in routing.
 * ============================================================================ */

static inline float yinsen_hamming(const float* a, const float* b, int len) {
    float count = 0.0f;
    for (int i = 0; i < len; i++) {
        float xor_bit = a[i] + b[i] - 2.0f * a[i] * b[i];
        count += xor_bit;
    }
    return count;
}

#ifdef __cplusplus
}
#endif

#endif /* YINSEN_APU_H */
