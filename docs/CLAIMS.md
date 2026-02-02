# Claims Register

This document tracks every claim made in the yinsen repository and its verification status.

**Last Updated:** 2026-02-01  
**Platform:** darwin/arm64  
**Total Tests:** 230 (125 core + 105 chips + 24 encoding + 38 falsification + 11 entromorph + 1 exhaustive 4x4 + seismic/ISS/keystroke demos)

## Verification Levels

| Level | Meaning | Confidence |
|-------|---------|------------|
| **PROVEN** | Exhaustively tested, all cases verified | 100% |
| **TESTED** | Property tests pass on single platform | High |
| **VALIDATED** | Tested on live real-world data | High |
| **FALSIFIED** | Attempted to break, found critical failure | DO NOT USE |
| **KNOWN ISSUE** | Edge case with documented behavior | Use with caution |
| **UNTESTED** | Code exists but no tests | Unknown |

## Quick Summary

| Category | PROVEN | TESTED | VALIDATED | FALSIFIED | KNOWN |
|----------|--------|--------|-----------|-----------|-------|
| Logic gates | 7 | - | - | - | - |
| Arithmetic | 2 | - | - | - | - |
| Activations | - | 6 | - | - | - |
| Matrix ops | - | 2 | - | - | - |
| CfC (base) | - | 4 | - | - | 2 |
| Ternary | 2 | 11 | - | - | 1 |
| Ternary CfC | - | 5 | - | - | 1 |
| **Chip Forge** | - | **17** | - | - | - |
| **CfC Variants** | - | **8** | - | - | - |
| **LUT+lerp** | - | **4** | - | - | - |
| **CfC SPARSE** | - | **6** | - | - | - |
| **Ternary Quant** | - | **4** | **2** | - | - |
| **Enrollment** | - | - | **3** | - | - |
| **Tau Principle** | - | - | **1** | - | - |
| EntroMorph | - | 8 | - | **1** | - |
| Training | - | 3 | - | **2** | - |
| **Total** | **11** | **78** | **6** | **3** | **4** |

---

## Chip Forge (include/chips/)

### GEMM Chip (gemm_chip.h)

| Claim | Level | Evidence |
|-------|-------|----------|
| GEMM_CHIP_BARE computes correct C = A @ B | **TESTED** | 8 tests in test_chips.c |
| GEMM_CHIP_BIASED fuses bias correctly | **TESTED** | test_chips.c |
| GEMM_CHIP matches yinsen_gemm | **TESTED** | Comparison test |

### Activation Chip (activation_chip.h)

| Claim | Level | Evidence |
|-------|-------|----------|
| SIGMOID_CHIP matches libm | **TESTED** | Exact comparison |
| SIGMOID_CHIP_FAST max error ~0.07 | **TESTED** | Sweep test |
| SIGMOID_CHIP_FAST3 max error ~2e-3 | **TESTED** | Sweep test |
| SIGMOID_CHIP_LUT max error 4.7e-5 | **TESTED** | 256-point sweep |
| TANH_CHIP matches libm | **TESTED** | Exact comparison |
| TANH_CHIP_FAST max error ~0.14 | **TESTED** | Sweep test |
| TANH_CHIP_FAST3 max error ~5e-3 | **TESTED** | Sweep test |
| TANH_CHIP_LUT max error 3.8e-4 | **TESTED** | 256-point sweep |
| EXP_CHIP matches libm | **TESTED** | Exact comparison |
| EXP_CHIP_FAST ~4% relative error | **TESTED** | Range test |
| LUT+lerp 200x more accurate than FAST3 | **TESTED** | Probe 2 comparison |
| LUT tables are 2KB total | **TESTED** | sizeof check |
| ACTIVATION_LUT_INIT is idempotent | **TESTED** | Double-init test |
| Vectorized variants match scalar | **TESTED** | 29 total activation tests |

### Decay Chip (decay_chip.h)

| Claim | Level | Evidence |
|-------|-------|----------|
| DECAY_CHIP_SCALAR = exp(-dt/tau) | **TESTED** | 14 tests in test_chips.c |
| DECAY_CHIP_SHARED broadcasts correctly | **TESTED** | Vector comparison |
| DECAY_CHIP_VECTOR per-neuron correct | **TESTED** | Vector comparison |
| DECAY_CHIP_FAST ~4% error | **TESTED** | Tolerance test |

### Ternary Dot Chip (ternary_dot_chip.h)

| Claim | Level | Evidence |
|-------|-------|----------|
| TERNARY_DOT_CHIP matches ternary_dot | **TESTED** | 7 tests in test_chips.c |
| Handles non-multiple-of-4 lengths | **TESTED** | Lengths 1,2,3,5,7 |
| Zero weights return 0 | **TESTED** | All-zero test |

### FFT Chip (fft_chip.h)

| Claim | Level | Evidence |
|-------|-------|----------|
| FFT_CHIP computes correct DFT | **TESTED** | 16 tests in test_chips.c |
| IFFT_CHIP inverts FFT (roundtrip) | **TESTED** | Roundtrip test |
| Parseval's theorem holds | **TESTED** | Energy conservation |
| Power-of-2 requirement | **TESTED** | N=8,16,32,64,128,256 |

### Softmax Chip (softmax_chip.h)

| Claim | Level | Evidence |
|-------|-------|----------|
| SOFTMAX_CHIP sums to 1.0 | **TESTED** | 10 tests in test_chips.c |
| Numerically stable with large values | **TESTED** | max-subtraction verified |
| ARGMAX_CHIP returns correct index | **TESTED** | Multiple cases |

### Normalization Chip (norm_chip.h)

| Claim | Level | Evidence |
|-------|-------|----------|
| LAYERNORM_CHIP matches reference | **TESTED** | 9 tests in test_chips.c |
| RMSNORM_CHIP matches reference | **TESTED** | Comparison test |
| BATCHNORM_CHIP matches reference | **TESTED** | Comparison test |

---

## CfC Cell Variants (cfc_cell_chip.h)

| Claim | Level | Evidence |
|-------|-------|----------|
| CFC_CELL_GENERIC matches yinsen_cfc_cell | **TESTED** | Comparison test |
| CFC_CELL_FIXED matches GENERIC with precomputed decay | **TESTED** | 100-step comparison |
| CFC_CELL_LUT matches FIXED within LUT tolerance | **TESTED** | 100-step comparison |
| CFC_CELL_SPARSE is bit-identical to LUT with same ternary weights | **TESTED** | 100-step bit comparison (14 tests) |
| CFC_CELL_SPARSE uses zero multiplies in GEMM | **TESTED** | By construction (add/subtract only) |
| CFC_CELL_SPARSE is 2.73x faster than GENERIC | **TESTED** | Benchmarked on Apple M-series |
| CFC_CELL_LUT is 1.54x faster than GENERIC | **TESTED** | Benchmarked |
| cfc_precompute_decay matches runtime decay | **TESTED** | Exact comparison |

---

## LUT+lerp Activations

| Claim | Level | Evidence |
|-------|-------|----------|
| LUT+lerp sigmoid max error 4.7e-5 | **TESTED** | Full sweep over [-8,8] |
| LUT+lerp tanh max error 3.8e-4 | **TESTED** | Full sweep over [-8,8] |
| 200x more accurate than FAST3 | **TESTED** | Probe 2 direct comparison |
| L2 divergence after 1000 CfC steps: 6.8e-4 (vs FAST3: 0.137) | **TESTED** | Probe 2 |

---

## Ternary Quantization

| Claim | Level | Evidence |
|-------|-------|----------|
| Ternary CfC preserves 99% detection quality at threshold 0.10 | **VALIDATED** | Probe 1: side-by-side ISS simulation |
| All anomalies detected at threshold 0.10-0.20 | **VALIDATED** | Probe 1: 3/3 anomalies |
| 81% of weights are zero at threshold 0.10 | **TESTED** | Probe 1 weight analysis |
| Weight compression: 640 bytes -> 40 bytes (16x) | **TESTED** | Probe 1 measurement |
| FPU multiplies 1.0 as fast as 0.3 | **TESTED** | Probe 2 benchmarks |
| Sparse ternary: 31 adds replaces 160 MACs | **TESTED** | Probe 2 analysis |

---

## Enrollment Demos

| Claim | Level | Evidence |
|-------|-------|----------|
| ISS telemetry: CfC detects oscillation shifts (20/20 vs 3-sigma 0/20) | **VALIDATED** | iss_telemetry.c + live Lightstreamer (384 samples) |
| ISS: 8 channels, 3,552 bytes, 79 ns/channel | **VALIDATED** | Measured |
| Seismic: CfC beats STA/LTA on 4/5 tests | **VALIDATED** | seismic_detector.c + live SeedLink (7,351 samples) |
| Seismic: 58 ns/channel, 1,768 bytes, 148,810x real-time headroom | **VALIDATED** | Measured |
| Keystroke: 268-byte discriminant, 110 ns/keystroke | **VALIDATED** | keystroke_biometric.c, 4-probe falsification cycle |
| Keystroke: Easy 20/20, Medium 20/20, Hard 16/20, Control 10/20 | **VALIDATED** | v3 hybrid scoring |

---

## Tau Principle

| Claim | Level | Evidence |
|-------|-------|----------|
| Tau differentiation emerges when R ~ T | **VALIDATED** | Seismic: R=2700x, T=3000x, 2.2-2.4x faster detection |
| ISS tau doesn't differentiate (T~1) | **VALIDATED** | ISS tau ablation shows no improvement |

---

## Discriminant Convergence

| Claim | Level | Evidence |
|-------|-------|----------|
| PCA discriminant converges to 0.84-0.89 across 3 domains | **TESTED** | Keystroke, ISS, seismic all in range |
| PCA(5) is sweet spot for 8-dim hidden state | **TESTED** | N_PCS ~ ceil(HIDDEN_DIM * 0.6) |

---

## Logic Gates (apu.h)

| Claim | Level | Evidence |
|-------|-------|----------|
| XOR(a,b) = a + b - 2ab is exact for {0,1} | **PROVEN** | 4/4 truth table |
| AND(a,b) = ab is exact for {0,1} | **PROVEN** | 4/4 truth table |
| OR(a,b) = a + b - ab is exact for {0,1} | **PROVEN** | 4/4 truth table |
| NOT(a) = 1 - a is exact for {0,1} | **PROVEN** | 2/2 cases |
| NAND(a,b) = 1 - ab is exact for {0,1} | **PROVEN** | 4/4 truth table |
| NOR(a,b) = 1 - a - b + ab is exact for {0,1} | **PROVEN** | 4/4 truth table |
| XNOR(a,b) = 1 - a - b + 2ab is exact for {0,1} | **PROVEN** | 4/4 truth table |

## Arithmetic (apu.h)

| Claim | Level | Evidence |
|-------|-------|----------|
| Full adder produces correct sum and carry | **PROVEN** | 8/8 combinations |
| 8-bit ripple adder produces correct results | **PROVEN** | 65,536/65,536 combinations |

## Ternary Weights (ternary.h)

| Claim | Level | Evidence |
|-------|-------|----------|
| 2x2 ternary matvec is correct | **PROVEN** | 81/81 configurations |
| 4x4 ternary matvec is correct | **PROVEN** | 43,046,721/43,046,721 configurations |
| Trit encoding/decoding is lossless | **TESTED** | Pack/unpack roundtrip |
| Ternary dot product matches integer sum | **TESTED** | Property tests |
| Quantization finds nearest trit | **TESTED** | Boundary tests |
| Absmean quantization adapts to distribution | **TESTED** | test_absmean_quantize() |
| 4x memory compression vs int8 | **TESTED** | 2 bits vs 8 bits |
| Non-multiple-of-4 lengths work | **TESTED** | Lengths 1,2,3,5,6,7 |
| Zero-length inputs return 0 | **TESTED** | test_empty_inputs() |
| Reserved encoding (0b11) treated as 0 | **TESTED** | dot, matvec, int8 paths |
| Int8 dot matches float within quant error | **TESTED** | test_int8_dot() |

## CfC (cfc.h)

| Claim | Level | Evidence |
|-------|-------|----------|
| CfC cell is deterministic | **TESTED** | Same input -> same output |
| CfC cell outputs are bounded | **TESTED** | Zero input doesn't explode |
| CfC cell decays over time | **TESTED** | State -> 0 with zero input |
| CfC cell is numerically stable | **TESTED** | 10,000 iterations |

## Ternary CfC (cfc_ternary.h)

| Claim | Level | Evidence |
|-------|-------|----------|
| Ternary CfC is deterministic | **TESTED** | Same input -> same output |
| Ternary CfC outputs are bounded | **TESTED** | 100 random inputs bounded |
| Ternary CfC is numerically stable | **TESTED** | 1,000 iterations |
| 4.4x memory compression vs float CfC | **TESTED** | 52 vs 228 bytes |
| Ternary CfC output is valid distribution | **TESTED** | Softmax sums to 1.0 |

## Encoding (trit_encoding.h)

| Claim | Level | Evidence |
|-------|-------|----------|
| Canonical encoding: 00=0, 01=+1, 10=-1, 11=reserved | **TESTED** | 24 tests |
| All backends use canonical encoding | **TESTED** | Cross-backend consistency |
| Reserved encoding (0b11) decoded as 0 | **TESTED** | Verified in dot, matvec, int8 paths |

## Training (Experimental)

| Claim | Level | Evidence |
|-------|-------|----------|
| Ternary CfC can learn sine approximation | **TESTED** | MSE 0.000362 (v2) |
| Ternary CfC can predict Lorenz | **TESTED** | MSE 0.001490 (12.69x from float) |
| Width dominates over depth | **TESTED** | v2 factorial + v3 depth experiment |
| Trajectory distillation helps | **FALSIFIED** | v2 factorial: distillation hurts |
| Depth improves Lorenz prediction | **FALSIFIED** | v3: 2-layer 155x worse |

## EntroMorph (entromorph.h)

| Claim | Level | Evidence |
|-------|-------|----------|
| Evolution converges on XOR | **MISLEADING** | 100/100 converge but 0/100 confident |
| Solutions are useful | **FALSIFIED** | Fragile to 1% noise (88% accuracy) |

## Cross-Cutting

| Claim | Level | Evidence |
|-------|-------|----------|
| Deterministic across platforms | **UNTESTED** | Only darwin/arm64 tested |
| No dependencies beyond libc | **TESTED** | Compiles with just -lm |
| Enrollment IS the product for anomaly detection | **VALIDATED** | 3 domains, no training needed |

---

## Changelog

- 2026-02-01: Added chip forge (105 tests), enrollment (3 demos, live data), ternary quantization (2 probes), CfC variants (4 tiers), LUT+lerp, SPARSE, tau principle. Total: 230 tests.
- 2026-01-31: Added encoding claims, training claims. Total: 199 tests.
- 2026-01-26: Initial claims register.
