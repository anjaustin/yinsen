# Yinsen

**Verified 2-bit computation engine.**

Frozen math shapes. Deterministic ternary primitives. Zero dependencies. Pure C, header-only.

Every weight is {-1, 0, +1}. Every operation is multiplication-free. Every primitive is exhaustively verified where feasible. This is not a quantization trick -- it is a 2-bit architecture that makes neural computation enumerable.

## What Yinsen Does

Yinsen is a chip forge for temporal anomaly detection. Eight frozen primitives compose into a CfC (Closed-form Continuous-time) cell that runs at **20 ns/step with zero multiplies**. No training required -- enroll on normal data, detect anomalies immediately.

The pipeline: **Sensor -> CfC Cell -> PCA Discriminant -> Anomaly Score**

Validated on live data from the International Space Station, real seismic feeds, and keystroke biometrics.

### Performance Ladder

| Variant | Speed | Multiplies | Activations | When to use |
|---|---|---|---|---|
| `CFC_CELL_GENERIC` | 54 ns | 160 MACs + 24 expf | libm | Variable dt, arbitrary float weights |
| `CFC_CELL_FIXED` | 51 ns | 160 MACs | libm | Fixed dt, precomputed decay |
| `CFC_CELL_LUT` | 35 ns | 160 MACs | LUT+lerp | Fixed dt, LUT activations (200x more accurate than poly) |
| `CFC_CELL_SPARSE` | **20 ns** | **0** (31 adds) | LUT+lerp | Fixed dt, ternary weights, LUT + sparsity |

### Memory

| Application | Channels | Total Memory | L1 Usage |
|---|---|---|---|
| ISS Telemetry | 8 | 3,552 bytes | 5.4% of 64KB |
| Seismic Detector | 3 | 1,768 bytes | 2.7% of 64KB |
| Keystroke Biometric | 1 | 268 bytes | 0.4% of 64KB |

## Chip Forge

Eight frozen primitives in `include/chips/`. All 105 chip tests pass.

| Chip | File | Tests | Purpose |
|------|------|-------|---------|
| CfC Cell | `cfc_cell_chip.h` | 14 | Liquid neural cell (GENERIC, FIXED, LUT, SPARSE) |
| GEMM | `gemm_chip.h` | 8 | General matrix multiply (BARE, BIASED, full) |
| Activations | `activation_chip.h` | 29 | Sigmoid/tanh/exp (PRECISE, FAST3, LUT+lerp) |
| Decay | `decay_chip.h` | 14 | Temporal decay exp(-dt/tau) |
| Ternary Dot | `ternary_dot_chip.h` | 7 | Multiplication-free dot product |
| FFT | `fft_chip.h` | 16 | Radix-2 spectral features |
| Softmax | `softmax_chip.h` | 10 | Classification output |
| Normalization | `norm_chip.h` | 9 | LayerNorm, RMSNorm, BatchNorm |

### Activation Tiers

| Tier | Sigmoid Error | Tanh Error | Speed | Notes |
|---|---|---|---|---|
| PRECISE (libm) | exact | exact | ~4.2 ns/call | `SIGMOID_CHIP`, `TANH_CHIP` |
| FAST3 (degree-3) | 8.7e-2 | 2.4e-2 | ~3.6 ns/call | `SIGMOID_CHIP_FAST3`, `TANH_CHIP_FAST3` |
| LUT+lerp (256) | 4.7e-5 | 3.8e-4 | ~3.4 ns/call | `SIGMOID_CHIP_LUT`, `TANH_CHIP_LUT` |

LUT+lerp is 200x more accurate than FAST3 for only 4ns more per CfC step. 2KB shared read-only tables, hot in L1 across all channels.

## Ternary Quantization

Float CfC weights quantized to {-1, 0, +1} preserve 99% of detection quality at threshold=0.10. 81% of weights become zero. Weight memory drops from 640 bytes to 40 bytes (16x compression). The sparse variant then skips all zeros: 31 adds instead of 160 MACs.

Validated via falsification probes:
- Probe 1 (Bridge): Side-by-side float vs ternary on ISS simulation -- all anomalies detected at threshold 0.10-0.20
- Probe 2 (FPU): Discovered that sparse ternary is the moneyball -- zero multiplies, 2.73x faster

## Enrollment Demos

Three working demos validated on live data:

### ISS Telemetry (`examples/iss_telemetry.c`)
8-channel CfC anomaly detector for International Space Station sensors. CMG wheel speeds, thermal, cabin pressure, CO2. Live Lightstreamer connection validated with 384 detection samples.

```bash
cc -O2 -I include -I include/chips examples/iss_telemetry.c -lm -o examples/iss_telemetry
./examples/iss_telemetry          # offline simulation
source .venv/bin/activate && python scripts/iss_websocket.py | ./examples/iss_telemetry --stdin  # live
```

### Seismic Detector (`examples/seismic_detector.c`)
3-channel CfC processor for 100 Hz seismic waveforms with built-in tau ablation and STA/LTA baseline. Live GFZ SeedLink validated: 7,351 samples from GE.STU Stuttgart.

```bash
cc -O2 -I include -I include/chips examples/seismic_detector.c -lm -o examples/seismic_detector
./examples/seismic_detector       # offline with tau ablation
source .venv/bin/activate && python scripts/seismic_seedlink.py | ./examples/seismic_detector --stdin  # live
```

### Keystroke Biometric (`examples/keystroke_biometric.c`)
v3 hybrid linear discriminant. 268-byte discriminant, 110 ns/keystroke. Full falsification cycle through 4 probes.

```bash
cc -O2 -I include -I include/chips examples/keystroke_biometric.c -lm -o examples/keystroke_biometric
./examples/keystroke_biometric
```

## Why Ternary?

| Property | Float32 | Int8 | Ternary (2-bit) |
|----------|---------|------|---------|
| Multiply op | `a * b` | `a * b` | `switch(w)` |
| Memory/weight | 32 bits | 8 bits | 2 bits |
| Determinism | Platform-dependent | Exact | Exact |
| Auditability | "This weight is 0.7324..." | "This weight is 83..." | "+1, -1, or skip" |

A dot product becomes conditional accumulation: add if +1, subtract if -1, skip if 0. No multiplication required. The state space is finite and enumerable, which means exhaustive verification is possible at small scales.

## Verification Status

### Proven (exhaustive, 100% input coverage)
- Logic gates as exact polynomials (XOR, AND, OR, NOT, NAND, NOR, XNOR)
- Full adder (8/8 input combinations)
- 8-bit ripple adder (65,536/65,536 combinations)
- 2x2 ternary matvec (81/81 weight configurations)
- **4x4 ternary matvec (43,046,721/43,046,721 configurations)**

### Tested (property tests, single platform)
- Chip forge: 8 primitives, 105 tests (GEMM, activations, decay, ternary dot, FFT, softmax, normalization, CfC sparse)
- CfC cell: 4 variants (GENERIC, FIXED, LUT, SPARSE), determinism, stability
- CfC SPARSE: bit-identical to CfC LUT over 100 steps (14 tests)
- LUT+lerp: max error 4.7e-5 sigmoid, 3.8e-4 tanh (13 LUT tests)
- Ternary quantization: 99% quality at threshold 0.10 (Probe 1)
- Enrollment demos: validated on live ISS, seismic, and keystroke data

### Hardware Backends (research kernels)
- **NEON** -- SDOT/TBL/I8MM/Ghost-Stream kernels for Apple Silicon
- **Metal** -- GPU threadgroup-cooperative kernel (tiled matvec with shared memory + simd_sum), exhaustive verification
- **SME** -- ARM Scalable Matrix Extension kernels (M4, with scalar fallbacks)

### Falsified
- EntroMorph evolution -- converges numerically but produces meaningless solutions (see `docs/FALSIFICATION_ENTROMORPH.md`)
- Depth for ternary CfC -- 2-layer 155x worse than 1-layer
- Trajectory distillation -- hurts instead of helping

## Quick Start

```bash
make test          # Run core tests (~125)
make falsify       # Run 38 adversarial tests
make prove4x4      # Run 43M exhaustive 4x4 ternary matvec proof (~1 sec)

# Chip tests (105 tests for the chip forge)
cc -O2 -I include -I include/chips test/test_chips.c -lm -o test/test_chips && ./test/test_chips

# Build enrollment demos
cc -O2 -I include -I include/chips examples/iss_telemetry.c -lm -o examples/iss_telemetry
cc -O2 -I include -I include/chips examples/seismic_detector.c -lm -o examples/seismic_detector
cc -O2 -I include -I include/chips examples/keystroke_biometric.c -lm -o examples/keystroke_biometric
```

## Usage

```c
// Chip forge (the product)
#include "chips/activation_chip.h"   // 3-tier activations (PRECISE, FAST3, LUT)
#include "chips/cfc_cell_chip.h"     // CfC cell (GENERIC, FIXED, LUT, SPARSE)
#include "chips/gemm_chip.h"         // GEMM (BARE, BIASED, full)
#include "chips/decay_chip.h"        // Temporal decay
#include "chips/ternary_dot_chip.h"  // Multiplication-free dot product
#include "chips/fft_chip.h"          // Radix-2 FFT
#include "chips/softmax_chip.h"      // Softmax + argmax
#include "chips/norm_chip.h"         // LayerNorm, RMSNorm, BatchNorm

// Base primitives
#include "trit_encoding.h"   // Canonical 2-bit encoding
#include "ternary.h"         // Ternary weight system
#include "onnx_shapes.h"     // Neural network ops
#include "apu.h"             // Logic and arithmetic
```

```bash
cc -O2 -I include -I include/chips your_code.c -lm
```

## Project Structure

```
yinsen/
├── include/               # Header-only C library (zero dependencies)
│   ├── chips/             # Chip forge: 8 frozen primitives
│   │   ├── cfc_cell_chip.h       # CfC cell (GENERIC, FIXED, LUT, SPARSE)
│   │   ├── activation_chip.h     # Sigmoid/tanh/exp (PRECISE, FAST3, LUT+lerp)
│   │   ├── gemm_chip.h           # General matrix multiply
│   │   ├── decay_chip.h          # Temporal decay exp(-dt/tau)
│   │   ├── ternary_dot_chip.h    # Multiplication-free dot product
│   │   ├── fft_chip.h            # Radix-2 FFT
│   │   ├── softmax_chip.h        # Numerically stable softmax
│   │   └── norm_chip.h           # LayerNorm, RMSNorm, BatchNorm
│   ├── apu.h              # Logic + arithmetic (PROVEN)
│   ├── onnx_shapes.h      # Activations, matmul, softmax (TESTED)
│   ├── trit_encoding.h    # Canonical 2-bit encoding (single source of truth)
│   ├── ternary.h          # Ternary weights {-1,0,+1} (PROVEN at 4x4)
│   ├── cfc.h              # CfC cell, float weights (TESTED)
│   ├── cfc_ternary.h      # CfC cell, ternary weights (TESTED)
│   └── entromorph.h       # Evolution (FALSIFIED)
├── metal/                 # Metal GPU kernels (Swift Package)
├── neon/                  # NEON SIMD kernels (C + intrinsics)
├── sme/                   # ARM SME kernels (C + assembly)
├── test/                  # 230 tests incl. exhaustive proofs
│   ├── test_chips.c       # 105 chip forge tests
│   ├── test_shapes.c      # 44 shape tests
│   ├── test_cfc.c         # 13 CfC tests
│   ├── test_ternary.c     # 55 ternary tests
│   └── test_cfc_ternary.c # 13 CfC ternary tests
├── examples/
│   ├── iss_telemetry.c    # 8-channel ISS anomaly detector (live validated)
│   ├── seismic_detector.c # 3-channel seismic processor (live validated)
│   ├── keystroke_biometric.c # v3 hybrid linear discriminant
│   ├── hello_xor.c        # Logic gate demo
│   ├── hello_ternary.c    # Ternary weight demo
│   ├── train_sine.c       # v1 ternary CfC training proof
│   ├── train_sine_v2.c    # v2 factorial experiment, MSE 0.000362
│   └── diagnostic_v3.c    # v3 multi-task, Lorenz wall
├── experiments/
│   ├── keystroke_probes/   # Falsification probes 1-4
│   ├── iss_probes/         # ISS falsification probes 1-3
│   └── ternary_quantization/  # Bridge experiment + FPU optimization
├── scripts/
│   ├── iss_websocket.py    # ISS Lightstreamer v2 bridge
│   └── seismic_seedlink.py # GFZ SeedLink bridge (100 Hz)
├── docs/                  # Verification reports, claims register, theory
├── journal/               # LMM research methodology and scratchpad
└── CHANGELOG.md
```

## Architecture

```
┌─────────────────────────────────────────────┐
│  Hardware Backends                           │
│  - NEON SIMD (SDOT, I8MM, Ghost-Stream)     │
│  - Metal GPU (tiled matvec, verified)        │
│  - SME (streaming matrix, scalar fb)         │
├─────────────────────────────────────────────┤
│  Chip Forge (8 frozen primitives)   [TESTED] │
│  - CfC Cell (4 variants: 54→20 ns)          │
│  - GEMM, Activations (3 tiers), Decay       │
│  - Ternary Dot, FFT, Softmax, Norm          │
├─────────────────────────────────────────────┤
│  Enrollment Demos                   [LIVE]   │
│  - ISS Telemetry (8 ch, v2 pre-scaling)     │
│  - Seismic Detector (tau ablation)           │
│  - Keystroke Biometric (PCA discriminant)    │
├─────────────────────────────────────────────┤
│  Ternary Primitives                [PROVEN]  │
│  - 2-bit trit encoding                      │
│  - Multiplication-free dot product           │
│  - 43M+ configurations exhaustively verified │
├─────────────────────────────────────────────┤
│  APU (Logic/Arithmetic)            [PROVEN]  │
│  - Gates as exact polynomials                │
│  - Exhaustive adder verification             │
└─────────────────────────────────────────────┘
```

## Key Insights

1. **Enrollment IS the product.** No training needed for temporal anomaly detection. CfC = temporal feature extractor. PCA discriminant = decision layer. 268 bytes, human-readable.

2. **The FPU doesn't care what it multiplies by.** `1.0 * x` takes the same cycles as `0.3 * x`. Ternary constraint is on VALUES, not INSTRUCTIONS.

3. **Sparsity is the moneyball.** At 81% zero weights, sparse index lists beat dense GEMM by 2.73x. Zero multiplies.

4. **LUT+lerp is 200x more accurate than FAST3 for 4ns.** Cubic splines are SLOWER than precise libm on Apple Silicon (hardware transcendentals).

5. **Tau matters when decay dynamic range matches signal temporal structure.** Seismic R=2700x matches T=3000x -- tau differentiation gives 2.2-2.4x faster detection.

## Open Questions

1. **Cross-platform determinism?** (Untested -- only darwin/arm64 so far)
2. **ARM Cortex-M4 deployment?** (Next experiment: the deployment proof)
3. **Where is the quantization wall?** (12.69x degradation on Lorenz; partially explained by width, not geometry)

## Known Limitations

- **Quantization wall on complex tasks**: Lorenz shows 12.69x degradation from float (sine and anomaly detection are fine)
- **Determinism untested across platforms**: `expf()` in activations may vary
- **Header-only tradeoffs**: Good for embedding, bad for build times at scale
- **VLA stack usage**: Documented in chip headers; constrained platforms need care

## Documentation

- [VERIFICATION.md](docs/VERIFICATION.md) - Complete verification report (230 tests)
- [CLAIMS.md](docs/CLAIMS.md) - Verification claims register
- [TEST_MATRIX.md](docs/TEST_MATRIX.md) - Every test mapped to code
- [API.md](docs/API.md) - Function reference (base + chip forge)
- [EXAMPLES.md](docs/EXAMPLES.md) - Usage examples
- [THEORY.md](docs/THEORY.md) - Mathematical foundations
- [EDGE_CASES.md](docs/EDGE_CASES.md) - Known behaviors and limitations
- [FALSIFICATION_ENTROMORPH.md](docs/FALSIFICATION_ENTROMORPH.md) - Why evolution is broken

## License

MIT
