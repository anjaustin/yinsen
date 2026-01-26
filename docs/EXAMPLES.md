# Examples Guide

Practical examples of using Yinsen.

## 1. Hello XOR

The simplest example: using logic shapes.

```c
#include <stdio.h>
#include "yinsen/apu.h"

int main() {
    // XOR truth table using frozen polynomial
    printf("XOR Truth Table:\n");
    for (int a = 0; a <= 1; a++) {
        for (int b = 0; b <= 1; b++) {
            float result = yinsen_xor((float)a, (float)b);
            printf("  XOR(%d, %d) = %.0f\n", a, b, result);
        }
    }
    return 0;
}
```

Output:
```
XOR Truth Table:
  XOR(0, 0) = 0
  XOR(0, 1) = 1
  XOR(1, 0) = 1
  XOR(1, 1) = 0
```

## 2. 8-bit Addition

Adding two bytes using ripple carry adder.

```c
#include <stdio.h>
#include "yinsen/apu.h"

// Convert byte to bit array (LSB first)
void byte_to_bits(uint8_t val, float* bits) {
    for (int i = 0; i < 8; i++) {
        bits[i] = (val >> i) & 1 ? 1.0f : 0.0f;
    }
}

// Convert bit array to byte
uint8_t bits_to_byte(const float* bits) {
    uint8_t val = 0;
    for (int i = 0; i < 8; i++) {
        if (bits[i] > 0.5f) val |= (1 << i);
    }
    return val;
}

int main() {
    uint8_t a = 137;  // 10001001
    uint8_t b = 42;   // 00101010

    float a_bits[8], b_bits[8], result_bits[8];
    float carry;

    byte_to_bits(a, a_bits);
    byte_to_bits(b, b_bits);

    yinsen_ripple_add_8bit(a_bits, b_bits, 0.0f, result_bits, &carry);

    uint8_t result = bits_to_byte(result_bits);
    printf("%d + %d = %d (carry: %.0f)\n", a, b, result, carry);
    // Output: 137 + 42 = 179 (carry: 0)

    return 0;
}
```

## 3. Simple Neural Network

Using ONNX shapes for a forward pass.

```c
#include <stdio.h>
#include "yinsen/onnx_shapes.h"

int main() {
    // Input: 3 features
    float input[3] = {0.5f, -0.3f, 0.8f};

    // Layer 1: 3 -> 4 (weights + bias)
    float W1[12] = {
        0.1f, 0.2f, -0.1f,
        0.3f, -0.2f, 0.4f,
        -0.1f, 0.1f, 0.2f,
        0.2f, 0.3f, -0.3f
    };
    float b1[4] = {0.0f, 0.1f, -0.1f, 0.0f};
    float hidden[4];

    // Compute: hidden = ReLU(input @ W1.T + b1)
    yinsen_gemm(input, W1, b1, hidden, 1, 4, 3, 1.0f, 1.0f);
    for (int i = 0; i < 4; i++) {
        hidden[i] = yinsen_relu(hidden[i]);
    }

    // Layer 2: 4 -> 2 (output)
    float W2[8] = {
        0.5f, -0.3f, 0.2f, 0.1f,
        -0.2f, 0.4f, -0.1f, 0.3f
    };
    float b2[2] = {0.0f, 0.0f};
    float logits[2];

    yinsen_gemm(hidden, W2, b2, logits, 1, 2, 4, 1.0f, 1.0f);

    // Softmax for probabilities
    float probs[2];
    yinsen_softmax(logits, probs, 2);

    printf("Output probabilities: [%.4f, %.4f]\n", probs[0], probs[1]);

    return 0;
}
```

## 4. CfC Network

Using a CfC cell for sequence processing.

```c
#include <stdio.h>
#include <string.h>
#include "yinsen/cfc.h"

// Tiny CfC: 2 inputs, 4 hidden, 2 outputs
#define INPUT_DIM 2
#define HIDDEN_DIM 4
#define OUTPUT_DIM 2

// Frozen weights (would normally come from training/evolution)
static const float W_gate[24] = { /* ... initialized weights ... */ };
static const float b_gate[4] = {0, 0, 0, 0};
static const float W_cand[24] = { /* ... initialized weights ... */ };
static const float b_cand[4] = {0, 0, 0, 0};
static const float tau[1] = {1.0f};
static const float W_out[8] = { /* ... initialized weights ... */ };
static const float b_out[2] = {0, 0};

int main() {
    // Setup parameters
    CfCParams cell = {
        .input_dim = INPUT_DIM,
        .hidden_dim = HIDDEN_DIM,
        .W_gate = W_gate,
        .b_gate = b_gate,
        .W_cand = W_cand,
        .b_cand = b_cand,
        .tau = tau,
        .tau_shared = 1
    };

    CfCOutputParams output = {
        .hidden_dim = HIDDEN_DIM,
        .output_dim = OUTPUT_DIM,
        .W_out = W_out,
        .b_out = b_out
    };

    // Process a sequence
    float h[HIDDEN_DIM] = {0};  // Hidden state
    float h_new[HIDDEN_DIM];
    float dt = 0.1f;

    // Simulate 10 time steps
    for (int t = 0; t < 10; t++) {
        float x[INPUT_DIM] = {sinf(t * 0.5f), cosf(t * 0.5f)};

        yinsen_cfc_cell(x, h, dt, &cell, h_new);
        memcpy(h, h_new, sizeof(h));

        printf("t=%d: h=[%.3f, %.3f, %.3f, %.3f]\n",
               t, h[0], h[1], h[2], h[3]);
    }

    // Get output probabilities
    float probs[OUTPUT_DIM];
    yinsen_cfc_output_softmax(h, &output, probs);
    printf("Final output: [%.4f, %.4f]\n", probs[0], probs[1]);

    return 0;
}
```

## 5. Evolving a CfC Network

Using EntroMorph to evolve weights.

```c
#include <stdio.h>
#include "yinsen/entromorph.h"

// Simple fitness function: classify XOR
float evaluate(LiquidGenome* genome) {
    CfCParams cell;
    CfCOutputParams output;
    entro_genome_to_params(genome, &cell, &output);

    int correct = 0;
    float patterns[4][2] = {{0,0}, {0,1}, {1,0}, {1,1}};
    int labels[4] = {0, 1, 1, 0};  // XOR outputs

    for (int i = 0; i < 4; i++) {
        float h[4] = {0};
        float h_new[4];

        // Single step
        yinsen_cfc_cell(patterns[i], h, 0.1f, &cell, h_new);

        // Get prediction
        float probs[2];
        yinsen_cfc_output_softmax(h_new, &output, probs);

        int pred = probs[1] > probs[0] ? 1 : 0;
        if (pred == labels[i]) correct++;
    }

    return (float)correct / 4.0f;  // Accuracy
}

int main() {
    EntroRNG rng;
    entro_rng_seed(&rng, 12345);

    // Create initial genome
    LiquidGenome genome;
    entro_genesis(&genome, 2, 4, 2, &rng, 0);

    // Evolution loop
    LiquidGenome best = genome;
    best.fitness = evaluate(&best);

    for (int gen = 0; gen < 1000; gen++) {
        // Create mutant
        LiquidGenome mutant = best;
        entro_mutate(&mutant, &MUTATION_DEFAULT, &rng);
        mutant.fitness = evaluate(&mutant);

        // Keep if better
        if (mutant.fitness > best.fitness) {
            best = mutant;
            printf("Gen %d: fitness = %.2f\n", gen, best.fitness);
        }

        if (best.fitness >= 1.0f) {
            printf("Solved at generation %d!\n", gen);
            break;
        }
    }

    // Export winner
    FILE* f = fopen("evolved_xor.h", "w");
    entro_export_header(&best, "XOR_CHIP", f);
    fclose(f);

    printf("Exported to evolved_xor.h\n");
    return 0;
}
```

## 6. Building a Full Adder from Gates

Demonstrating composition of shapes.

```c
#include <stdio.h>
#include "yinsen/apu.h"

// Build full adder manually from gates
void my_full_adder(float a, float b, float cin, float* sum, float* cout) {
    // sum = a XOR b XOR cin
    float ab_xor = yinsen_xor(a, b);
    *sum = yinsen_xor(ab_xor, cin);

    // cout = (a AND b) OR ((a XOR b) AND cin)
    float ab_and = yinsen_and(a, b);
    float xor_cin = yinsen_and(ab_xor, cin);
    *cout = yinsen_or(ab_and, xor_cin);
}

int main() {
    printf("Full Adder Truth Table:\n");
    printf("A B Cin | Sum Cout\n");
    printf("-----------------\n");

    for (int a = 0; a <= 1; a++) {
        for (int b = 0; b <= 1; b++) {
            for (int cin = 0; cin <= 1; cin++) {
                float sum, cout;
                my_full_adder(a, b, cin, &sum, &cout);
                printf("%d %d  %d  |  %.0f   %.0f\n",
                       a, b, cin, sum, cout);
            }
        }
    }

    return 0;
}
```

## Compiling Examples

All examples compile with:

```bash
gcc -I./include -O2 -std=c11 example.c -lm -o example
```

Or use the Makefile:

```bash
make examples
./build/hello_xor
```
