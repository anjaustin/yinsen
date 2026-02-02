# Examples Guide

Practical examples of using Yinsen. Starts with the real working demos, then chip forge usage, then base primitives.

**Last Updated:** 2026-02-01

---

## Enrollment Demos (Working, Live-Validated)

### ISS Telemetry Anomaly Detector

8-channel CfC anomaly detector for International Space Station sensors. Full source: `examples/iss_telemetry.c`.

```bash
# Build
cc -O2 -I include -I include/chips examples/iss_telemetry.c -lm -o examples/iss_telemetry

# Run offline simulation (built-in test data)
./examples/iss_telemetry

# Run on live ISS data
source .venv/bin/activate
python scripts/iss_websocket.py | ./examples/iss_telemetry --stdin
```

4-phase pipeline: Calibrate (learn per-channel scales) -> Enroll (build baseline) -> Normal Test -> Anomaly Test. CfC detects oscillation shifts that 3-sigma misses (20/20 vs 0/20).

### Seismic Detector

3-channel CfC processor for 100 Hz seismic waveforms. Full source: `examples/seismic_detector.c`.

```bash
# Build
cc -O2 -I include -I include/chips examples/seismic_detector.c -lm -o examples/seismic_detector

# Run offline with tau ablation
./examples/seismic_detector

# Run on live seismic data from GFZ Stuttgart
source .venv/bin/activate
python scripts/seismic_seedlink.py | ./examples/seismic_detector --stdin
```

Includes built-in tau ablation (matched vs ISS-tau vs constant-tau) and STA/LTA baseline comparison.

### Keystroke Biometric

v3 hybrid linear discriminant. Full source: `examples/keystroke_biometric.c`.

```bash
cc -O2 -I include -I include/chips examples/keystroke_biometric.c -lm -o examples/keystroke_biometric
./examples/keystroke_biometric
```

268-byte discriminant, 110 ns/keystroke. Uses power iteration PCA on 8-dim CfC hidden state.

---

## Chip Forge Examples

### Using CFC_CELL_GENERIC

The simplest chip forge usage: a CfC cell with variable time step.

```c
#include <string.h>
#include "onnx_shapes.h"
#include "chips/activation_chip.h"
#include "chips/cfc_cell_chip.h"

#define INPUT_DIM  2
#define HIDDEN_DIM 8

// Frozen weights (from training or enrollment)
static const float W_gate[HIDDEN_DIM * (INPUT_DIM + HIDDEN_DIM)] = { /* ... */ };
static const float b_gate[HIDDEN_DIM] = {0};
static const float W_cand[HIDDEN_DIM * (INPUT_DIM + HIDDEN_DIM)] = { /* ... */ };
static const float b_cand[HIDDEN_DIM] = {0};
static const float tau[1] = {1.0f};

int main() {
    float h[HIDDEN_DIM] = {0};
    float h_new[HIDDEN_DIM];

    for (int t = 0; t < 100; t++) {
        float x[INPUT_DIM] = { /* sensor reading */ };
        float dt = 0.01f;  // 100 Hz

        CFC_CELL_GENERIC(x, h, dt, W_gate, b_gate, W_cand, b_cand,
                         tau, 1, INPUT_DIM, HIDDEN_DIM, h_new);
        memcpy(h, h_new, sizeof(h));
    }
    return 0;
}
```

### Using CFC_CELL_LUT (1.54x faster)

Precomputed decay + LUT activations. Best balance of speed and accuracy.

```c
#include "onnx_shapes.h"
#include "chips/activation_chip.h"
#include "chips/cfc_cell_chip.h"

int main() {
    // One-time initialization
    ACTIVATION_LUT_INIT();

    // Precompute decay for fixed sample rate
    float decay[HIDDEN_DIM];
    cfc_precompute_decay(tau, 1, 0.01f, HIDDEN_DIM, decay);

    float h[HIDDEN_DIM] = {0};
    for (int t = 0; t < 100; t++) {
        float x[INPUT_DIM] = { /* sensor reading */ };
        CFC_CELL_LUT(x, h, W_gate, b_gate, W_cand, b_cand,
                      decay, INPUT_DIM, HIDDEN_DIM, h);
    }
    return 0;
}
```

### Using CFC_CELL_SPARSE (2.73x faster, zero multiplies)

The moneyball. Ternary weights + sparse index lists. Zero multiplies in the GEMM.

```c
#include "onnx_shapes.h"
#include "chips/activation_chip.h"
#include "chips/cfc_cell_chip.h"

int main() {
    ACTIVATION_LUT_INIT();

    // Ternary weights (from quantization probe: threshold 0.10)
    static const float W_gate_ternary[H * C] = { /* {-1, 0, +1} values */ };
    static const float W_cand_ternary[H * C] = { /* {-1, 0, +1} values */ };

    // Build sparse index lists (once at init)
    CfcSparseWeights sw;
    cfc_build_sparse(W_gate_ternary, W_cand_ternary,
                     0.5f,  // threshold: anything > 0.5 is +1, < -0.5 is -1
                     HIDDEN_DIM, INPUT_DIM + HIDDEN_DIM,
                     0,     // transposed=0 for GEMM-native layout
                     &sw);

    // Precompute decay
    float decay[HIDDEN_DIM];
    cfc_precompute_decay(tau, 1, 0.01f, HIDDEN_DIM, decay);

    // Hot path: zero multiplies, 31 adds instead of 160 MACs
    float h[HIDDEN_DIM] = {0};
    for (int t = 0; t < 100; t++) {
        float x[INPUT_DIM] = { /* sensor reading */ };
        CFC_CELL_SPARSE(x, h, &sw, b_gate, b_cand,
                         decay, INPUT_DIM, HIDDEN_DIM, h);
    }
    return 0;
}
```

### Using LUT+lerp Activations Directly

```c
#include "chips/activation_chip.h"

int main() {
    ACTIVATION_LUT_INIT();  // Fill 2KB tables, call once

    // Drop-in replacements for libm
    float s = SIGMOID_CHIP_LUT(1.5f);    // 4.7e-5 max error
    float t = TANH_CHIP_LUT(-0.8f);      // 3.8e-4 max error

    // Vectorized
    float x[8] = {-2, -1, 0, 0.5, 1, 1.5, 2, 3};
    float y[8];
    SIGMOID_VEC_CHIP_LUT(x, y, 8);

    return 0;
}
```

### Using FFT for Spectral Features

Pipeline: sensor -> FFT -> magnitude -> CfC.

```c
#include "chips/fft_chip.h"

int main() {
    float real[256], imag[256];

    // Fill real[] with sensor samples, imag[] with zeros
    for (int i = 0; i < 256; i++) {
        real[i] = sensor_read();
        imag[i] = 0.0f;
    }

    // In-place FFT
    FFT_CHIP(real, imag, 256);

    // Power spectrum (first 128 bins = positive frequencies)
    float power[256];
    FFT_POWER(real, imag, power, 256);

    // Feed spectral features to CfC...
    return 0;
}
```

---

## Base Primitive Examples

### Hello XOR

```c
#include <stdio.h>
#include "apu.h"

int main() {
    for (int a = 0; a <= 1; a++)
        for (int b = 0; b <= 1; b++)
            printf("XOR(%d, %d) = %.0f\n", a, b, yinsen_xor((float)a, (float)b));
    return 0;
}
```

### Ternary Dot Product

```c
#include <stdio.h>
#include "ternary.h"

int main() {
    // Weights: [+1, -1, 0, +1]
    uint8_t w = trit_pack4(1, -1, 0, 1);
    float x[4] = {1.0f, 2.0f, 3.0f, 4.0f};

    // 1*1 + (-1)*2 + 0*3 + 1*4 = 3. No multiplication.
    float result = ternary_dot(&w, x, 4);
    printf("Ternary dot: %.1f\n", result);  // 3.0
    return 0;
}
```

### Quantizing Float Weights to Ternary

```c
#include <stdio.h>
#include "ternary.h"

int main() {
    float weights[8] = {0.8f, -0.9f, 0.1f, -0.2f, 0.7f, -0.6f, 0.0f, 0.5f};
    uint8_t packed[2];
    ternary_quantize_absmean(weights, packed, 8);

    TernaryStats stats;
    ternary_stats(packed, 8, &stats);
    printf("Sparsity: %.1f%% (%d zeros of %d)\n",
           stats.sparsity * 100.0f, stats.zeros, stats.total);
    return 0;
}
```

---

## Compiling

All examples compile with:

```bash
# Chip forge examples (include both paths)
cc -O2 -I include -I include/chips example.c -lm -o example

# Base primitive examples (include path only)
cc -O2 -I include example.c -lm -o example

# Or use the Makefile for built-in examples
make examples
```

---

## See Also

- [API.md](API.md) - Complete function reference
- [THEORY.md](THEORY.md) - Mathematical foundations
- [EDGE_CASES.md](EDGE_CASES.md) - Known behaviors
