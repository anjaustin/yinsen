/*
 * YINSEN Example: Hello XOR
 *
 * The simplest frozen shape.
 *
 * Compile: gcc -o hello_xor hello_xor.c -I../include -lm
 * Run:     ./hello_xor
 */

#include <stdio.h>
#include "../include/apu.h"

int main() {
    printf("===================================================\n");
    printf("  YINSEN: Hello XOR\n");
    printf("===================================================\n\n");

    printf("The frozen XOR shape: a + b - 2ab\n\n");

    printf("Truth Table:\n");
    printf("+---+---+----------+\n");
    printf("| a | b | XOR(a,b) |\n");
    printf("+---+---+----------+\n");

    for (int a = 0; a <= 1; a++) {
        for (int b = 0; b <= 1; b++) {
            float result = yinsen_xor((float)a, (float)b);
            printf("| %d | %d |    %.0f     |\n", a, b, result);
        }
    }

    printf("+---+---+----------+\n\n");

    printf("XOR is just a polynomial. No training. No runtime. Just math.\n");

    return 0;
}
