/*
 * YINSEN Example: Hello Ternary
 *
 * Demonstrates ternary weights: {-1, 0, +1}
 * No multiplication - just add/subtract.
 *
 * Compile: gcc -o hello_ternary hello_ternary.c -I../include -lm
 * Run:     ./hello_ternary
 */

#include <stdio.h>
#include "../include/ternary.h"

int main() {
    printf("===================================================\n");
    printf("  YINSEN: Hello Ternary\n");
    printf("===================================================\n\n");

    /* --------------------------------------------------------
     * Ternary weights: {-1, 0, +1}
     *
     * +1 = add the input
     * -1 = subtract the input
     *  0 = skip (sparse)
     * -------------------------------------------------------- */

    printf("Ternary weights use only three values:\n");
    printf("  +1 = add\n");
    printf("  -1 = subtract\n");
    printf("   0 = skip\n\n");

    /* Example: compute dot product with ternary weights */
    float x[4] = {1.0f, 2.0f, 3.0f, 4.0f};

    /* Weights: [+1, -1, 0, +1] */
    /* Packed into 1 byte (2 bits per weight) */
    uint8_t w = trit_pack4(1, -1, 0, 1);

    printf("Input vector:  [%.0f, %.0f, %.0f, %.0f]\n", x[0], x[1], x[2], x[3]);
    printf("Ternary weights: [+1, -1, 0, +1]\n\n");

    /* Compute: w . x = (+1)*1 + (-1)*2 + (0)*3 + (+1)*4 = 1 - 2 + 0 + 4 = 3 */
    float result = ternary_dot(&w, x, 4);

    printf("Dot product = (+1)*1 + (-1)*2 + (0)*3 + (+1)*4\n");
    printf("            = 1 - 2 + 0 + 4\n");
    printf("            = %.0f\n\n", result);

    /* --------------------------------------------------------
     * Memory comparison: ternary vs float
     * -------------------------------------------------------- */

    size_t tern_bytes, float_bytes;
    float ratio;
    ternary_memory_stats(1024, &tern_bytes, &float_bytes, &ratio);

    printf("Memory for 1024 weights:\n");
    printf("  Float32: %zu bytes\n", float_bytes);
    printf("  Ternary: %zu bytes\n", tern_bytes);
    printf("  Compression: %.0fx\n\n", ratio);

    /* --------------------------------------------------------
     * Why ternary?
     * -------------------------------------------------------- */

    printf("Why ternary?\n");
    printf("  - No multiplication (just add/subtract)\n");
    printf("  - 16x smaller than float32\n");
    printf("  - Deterministic (integer-like behavior)\n");
    printf("  - Hardware-friendly (works on any ALU)\n");

    return 0;
}
