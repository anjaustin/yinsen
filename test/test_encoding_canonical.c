/*
 * YINSEN Canonical Encoding Verification
 *
 * Verifies that the 2-bit trit encoding is consistent across backends.
 * Tests all 256 possible packed bytes (exhaustive for 4-trit packing).
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include "../include/ternary.h"

static int tests_run = 0;
static int tests_passed = 0;

#define TEST(cond, name) do { \
    tests_run++; \
    if (cond) { tests_passed++; } \
    else { printf("  FAIL: %s\n", name); } \
} while(0)

/* The NEON TBL decode table: index -> trit value */
static const int8_t NEON_DECODE[4] = {0, 1, -1, 0};

/* Reference decode matching canonical encoding */
static int8_t canonical_decode(uint8_t bits) {
    if (bits == 1) return 1;   /* 01 = +1 */
    if (bits == 2) return -1;  /* 10 = -1 */
    return 0;                  /* 00 = 0, 11 = reserved/0 */
}

void test_encoding_constants(void) {
    printf("\n=== Encoding Constants ===\n");

    TEST(TRIT_ZERO == 0x0, "TRIT_ZERO == 0x0");
    TEST(TRIT_POS  == 0x1, "TRIT_POS  == 0x1");
    TEST(TRIT_NEG  == 0x2, "TRIT_NEG  == 0x2");
    TEST(TRIT_RESERVED == 0x3, "TRIT_RESERVED == 0x3");
}

void test_encode_roundtrip(void) {
    printf("\n=== Encode/Decode Roundtrip ===\n");

    TEST(trit_encode(1)  == TRIT_POS,  "encode(+1) == TRIT_POS (0x1)");
    TEST(trit_encode(-1) == TRIT_NEG,  "encode(-1) == TRIT_NEG (0x2)");
    TEST(trit_encode(0)  == TRIT_ZERO, "encode(0)  == TRIT_ZERO (0x0)");

    /* Roundtrip all three values */
    for (int v = -1; v <= 1; v++) {
        uint8_t encoded = trit_encode((int8_t)v);
        uint8_t packed = encoded;  /* single trit at position 0 */
        int8_t decoded = trit_unpack(packed, 0);
        TEST(decoded == v, "roundtrip trit value");
    }
}

void test_neon_decode_table_matches(void) {
    printf("\n=== NEON Decode Table Consistency ===\n");

    /* The NEON TBL table {0, 1, -1, 0} must agree with canonical_decode
     * for all 4 possible 2-bit values */
    int all_match = 1;
    for (int bits = 0; bits < 4; bits++) {
        int8_t neon_val = NEON_DECODE[bits];
        int8_t canon_val = canonical_decode((uint8_t)bits);
        int8_t ternary_h_val = trit_unpack((uint8_t)bits, 0);
        if (neon_val != canon_val || canon_val != ternary_h_val) {
            printf("  Mismatch at bits=%d: NEON=%d canonical=%d ternary.h=%d\n",
                   bits, neon_val, canon_val, ternary_h_val);
            all_match = 0;
        }
    }
    TEST(all_match, "NEON decode table matches canonical and ternary.h for all 4 values");
}

void test_all_256_packed_bytes(void) {
    printf("\n=== Exhaustive 256-Byte Verification ===\n");

    int all_correct = 1;
    int failures = 0;

    for (int byte_val = 0; byte_val < 256; byte_val++) {
        uint8_t packed = (uint8_t)byte_val;

        for (int pos = 0; pos < 4; pos++) {
            uint8_t bits = (packed >> (pos * 2)) & 0x3;

            /* ternary.h decode */
            int8_t th_val = trit_unpack(packed, pos);

            /* canonical reference */
            int8_t canon_val = canonical_decode(bits);

            /* NEON decode table */
            int8_t neon_val = NEON_DECODE[bits];

            if (th_val != canon_val || canon_val != neon_val) {
                if (failures < 5) {
                    printf("  Mismatch: byte=0x%02X pos=%d bits=%d "
                           "ternary.h=%d canonical=%d NEON=%d\n",
                           byte_val, pos, bits, th_val, canon_val, neon_val);
                }
                all_correct = 0;
                failures++;
            }
        }
    }
    if (failures > 5) {
        printf("  ... and %d more mismatches\n", failures - 5);
    }
    TEST(all_correct, "All 256 packed bytes decode consistently (1024 trit checks)");
}

void test_pack4_bit_patterns(void) {
    printf("\n=== Pack4 Bit Pattern Verification ===\n");

    /* Pack known trits and verify exact bit patterns */

    /* All zeros: 00 00 00 00 = 0x00 */
    TEST(trit_pack4(0, 0, 0, 0) == 0x00, "pack4(0,0,0,0) == 0x00");

    /* All +1: 01 01 01 01 = 0x55 */
    TEST(trit_pack4(1, 1, 1, 1) == 0x55, "pack4(1,1,1,1) == 0x55");

    /* All -1: 10 10 10 10 = 0xAA */
    TEST(trit_pack4(-1, -1, -1, -1) == 0xAA, "pack4(-1,-1,-1,-1) == 0xAA");

    /* Mixed: +1 at pos0, -1 at pos1, 0 at pos2, +1 at pos3 */
    /* bits: 01 | 10 | 00 | 01 = 0x01 | 0x08 | 0x00 | 0x40 = 0x49 */
    uint8_t mixed = trit_pack4(1, -1, 0, 1);
    TEST(trit_unpack(mixed, 0) ==  1, "mixed pos0 == +1");
    TEST(trit_unpack(mixed, 1) == -1, "mixed pos1 == -1");
    TEST(trit_unpack(mixed, 2) ==  0, "mixed pos2 == 0");
    TEST(trit_unpack(mixed, 3) ==  1, "mixed pos3 == +1");

    /* Verify the packed byte matches expected bit pattern */
    /* pos0: +1 = 01, pos1: -1 = 10, pos2: 0 = 00, pos3: +1 = 01 */
    /* byte = 01 | (10 << 2) | (00 << 4) | (01 << 6) */
    /*      = 0x01 | 0x08 | 0x00 | 0x40 = 0x49 */
    TEST(mixed == 0x49, "pack4(1,-1,0,1) == 0x49");
}

void test_dot_product_with_canonical_encoding(void) {
    printf("\n=== Dot Product with Canonical Encoding ===\n");

    /* Weights: [+1, -1, +1, 0] */
    uint8_t packed = trit_pack4(1, -1, 1, 0);
    float x[4] = {2.0f, 3.0f, 5.0f, 7.0f};

    /* Expected: 1*2 + (-1)*3 + 1*5 + 0*7 = 2 - 3 + 5 = 4.0 */
    float result = ternary_dot(&packed, x, 4);
    TEST(fabsf(result - 4.0f) < 1e-6f, "dot([+1,-1,+1,0], [2,3,5,7]) == 4.0");

    /* All -1 weights */
    packed = trit_pack4(-1, -1, -1, -1);
    result = ternary_dot(&packed, x, 4);
    /* Expected: -2 - 3 - 5 - 7 = -17.0 */
    TEST(fabsf(result - (-17.0f)) < 1e-6f, "dot([-1,-1,-1,-1], [2,3,5,7]) == -17.0");
}

void test_reserved_encoding_is_zero(void) {
    printf("\n=== Reserved Encoding (0x3) Treated as Zero ===\n");

    /* Manually construct a byte with reserved encoding at pos 0 */
    uint8_t packed = 0x03;  /* bits 11 at pos 0 */
    int8_t val = trit_unpack(packed, 0);
    TEST(val == 0, "Reserved encoding 0b11 decodes to 0");

    /* Dot product should skip reserved-encoded positions */
    float x[4] = {100.0f, 0.0f, 0.0f, 0.0f};
    float result = ternary_dot(&packed, x, 4);
    TEST(fabsf(result) < 1e-6f, "Reserved encoding contributes 0 to dot product");
}

int main(void) {
    printf("===================================================\n");
    printf("  YINSEN CANONICAL ENCODING VERIFICATION\n");
    printf("===================================================\n");

    test_encoding_constants();
    test_encode_roundtrip();
    test_neon_decode_table_matches();
    test_all_256_packed_bytes();
    test_pack4_bit_patterns();
    test_dot_product_with_canonical_encoding();
    test_reserved_encoding_is_zero();

    printf("\n===================================================\n");
    printf("  RESULTS: %d/%d passed\n", tests_passed, tests_run);
    printf("===================================================\n");

    if (tests_passed == tests_run) {
        printf("  ALL TESTS PASSED\n");
        return 0;
    } else {
        printf("  SOME TESTS FAILED\n");
        return 1;
    }
}
