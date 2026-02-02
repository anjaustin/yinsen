# Test Matrix

Complete mapping of every test to its target code and verification level.

**Generated:** 2026-02-01  
**Platform:** darwin/arm64  
**Total C tests:** 230

---

## test_chips.c (105 tests)

### GEMM (8 tests)

| Test | Function | Property | Status |
|------|----------|----------|--------|
| GEMM_CHIP_BARE identity | `GEMM_CHIP_BARE` | I @ x = x | TESTED |
| GEMM_CHIP_BARE known values | `GEMM_CHIP_BARE` | Correct product | TESTED |
| GEMM_CHIP_BIASED | `GEMM_CHIP_BIASED` | C = A@B + bias | TESTED |
| GEMM_CHIP full | `GEMM_CHIP` | C = alpha*A@B + beta*bias | TESTED |
| GEMM vs yinsen_gemm | `GEMM_CHIP` | Matches base primitive | TESTED |
| GEMM M=1 (matvec) | `GEMM_CHIP_BARE` | Dot product case | TESTED |
| GEMM rectangular | `GEMM_CHIP_BARE` | Non-square M!=N!=K | TESTED |
| GEMM zero matrix | `GEMM_CHIP_BARE` | Returns zeros | TESTED |

### Activations (29 tests)

| Test | Function | Property | Status |
|------|----------|----------|--------|
| Sigmoid precise | `SIGMOID_CHIP` | Matches libm | TESTED |
| Sigmoid σ(0)=0.5 | `SIGMOID_CHIP` | Exact at zero | TESTED |
| Sigmoid fast | `SIGMOID_CHIP_FAST` | Error < 0.08 | TESTED |
| Sigmoid fast3 | `SIGMOID_CHIP_FAST3` | Error < 0.01 | TESTED |
| Sigmoid LUT | `SIGMOID_CHIP_LUT` | Error < 5e-5 | TESTED |
| Sigmoid LUT saturation | `SIGMOID_CHIP_LUT` | Correct at +/-8 | TESTED |
| Tanh precise | `TANH_CHIP` | Matches libm | TESTED |
| Tanh(0)=0 | `TANH_CHIP` | Exact at zero | TESTED |
| Tanh fast | `TANH_CHIP_FAST` | Error < 0.15 | TESTED |
| Tanh fast3 | `TANH_CHIP_FAST3` | Error < 0.01 | TESTED |
| Tanh LUT | `TANH_CHIP_LUT` | Error < 4e-4 | TESTED |
| Exp precise | `EXP_CHIP` | Matches libm | TESTED |
| Exp fast | `EXP_CHIP_FAST` | ~4% relative error | TESTED |
| ReLU positive | `RELU_CHIP` | relu(2)=2 | TESTED |
| ReLU negative | `RELU_CHIP` | relu(-2)=0 | TESTED |
| ReLU zero | `RELU_CHIP` | relu(0)=0 | TESTED |
| GELU | `GELU_CHIP` | Matches reference | TESTED |
| GELU fast | `GELU_CHIP_FAST` | Reasonable error | TESTED |
| SiLU | `SILU_CHIP` | Matches reference | TESTED |
| SiLU fast | `SILU_CHIP_FAST` | Reasonable error | TESTED |
| LUT init idempotent | `ACTIVATION_LUT_INIT` | Safe to call twice | TESTED |
| Sigmoid vec | `SIGMOID_VEC_CHIP` | Matches scalar | TESTED |
| Sigmoid vec LUT | `SIGMOID_VEC_CHIP_LUT` | Matches scalar LUT | TESTED |
| Tanh vec | `TANH_VEC_CHIP` | Matches scalar | TESTED |
| Tanh vec LUT | `TANH_VEC_CHIP_LUT` | Matches scalar LUT | TESTED |
| ReLU vec | `RELU_VEC_CHIP` | Matches scalar | TESTED |
| Exp vec | `EXP_VEC_CHIP` | Matches scalar | TESTED |
| LUT accuracy sweep | Various LUT | Max error across range | TESTED |
| LUT vs FAST3 comparison | LUT vs FAST3 | LUT >> FAST3 accuracy | TESTED |

### Decay (14 tests)

| Test | Function | Property | Status |
|------|----------|----------|--------|
| Scalar basic | `DECAY_CHIP_SCALAR` | exp(-dt/tau) | TESTED |
| Scalar dt=0 | `DECAY_CHIP_SCALAR` | Returns 1.0 | TESTED |
| Scalar large dt | `DECAY_CHIP_SCALAR` | Returns ~0 | TESTED |
| Scalar fast | `DECAY_CHIP_SCALAR_FAST` | ~4% error | TESTED |
| Shared broadcast | `DECAY_CHIP_SHARED` | All elements equal | TESTED |
| Vector per-neuron | `DECAY_CHIP_VECTOR` | Different taus | TESTED |
| Matches cfc_precompute_decay | `cfc_precompute_decay` | Exact match | TESTED |
| tau_shared mode | `cfc_precompute_decay` | Broadcasts correctly | TESTED |
| Per-neuron mode | `cfc_precompute_decay` | Different decays | TESTED |
| Extreme dt (tiny) | `DECAY_CHIP_SCALAR` | Finite | TESTED |
| Extreme dt (huge) | `DECAY_CHIP_SCALAR` | ~0 | TESTED |
| Extreme tau (tiny) | `DECAY_CHIP_SCALAR` | ~0 | TESTED |
| Extreme tau (huge) | `DECAY_CHIP_SCALAR` | ~1 | TESTED |
| Fast vs precise | `DECAY_CHIP_SCALAR_FAST` | Error bounded | TESTED |

### Ternary Dot (7 tests)

| Test | Function | Property | Status |
|------|----------|----------|--------|
| Basic dot | `TERNARY_DOT_CHIP` | Correct sum | TESTED |
| All positive | `TERNARY_DOT_CHIP` | Sum of x | TESTED |
| All negative | `TERNARY_DOT_CHIP` | -Sum of x | TESTED |
| All zero | `TERNARY_DOT_CHIP` | Returns 0 | TESTED |
| Mixed | `TERNARY_DOT_CHIP` | Correct sum | TESTED |
| Matches ternary_dot | `TERNARY_DOT_CHIP` | Matches base | TESTED |
| Non-multiple-of-4 | `TERNARY_DOT_CHIP` | Handles remainders | TESTED |

### FFT (16 tests)

| Test | Function | Property | Status |
|------|----------|----------|--------|
| DC signal | `FFT_CHIP` | Single bin | TESTED |
| Pure sine | `FFT_CHIP` | Two bins | TESTED |
| Roundtrip | `FFT_CHIP`+`IFFT_CHIP` | Lossless | TESTED |
| Parseval energy | `FFT_POWER` | Conserved | TESTED |
| N=8 | `FFT_CHIP` | Correct | TESTED |
| N=16 | `FFT_CHIP` | Correct | TESTED |
| N=32 | `FFT_CHIP` | Correct | TESTED |
| N=64 | `FFT_CHIP` | Correct | TESTED |
| N=128 | `FFT_CHIP` | Correct | TESTED |
| N=256 | `FFT_CHIP` | Correct | TESTED |
| All zeros | `FFT_CHIP` | All zeros out | TESTED |
| Impulse | `FFT_CHIP` | Flat spectrum | TESTED |
| Magnitude | `FFT_MAGNITUDE` | sqrt(re^2+im^2) | TESTED |
| Power spectrum | `FFT_POWER` | re^2 + im^2 | TESTED |
| IFFT normalization | `IFFT_CHIP` | 1/N scaling | TESTED |
| Complex input | `FFT_CHIP` | Non-zero imag | TESTED |

### Softmax (10 tests)

| Test | Function | Property | Status |
|------|----------|----------|--------|
| Sum = 1 | `SOFTMAX_CHIP` | Probabilities sum | TESTED |
| All positive | `SOFTMAX_CHIP` | Correct | TESTED |
| Large values | `SOFTMAX_CHIP` | Numerically stable | TESTED |
| Uniform input | `SOFTMAX_CHIP` | Equal probabilities | TESTED |
| Single element | `SOFTMAX_CHIP` | Returns 1.0 | TESTED |
| Fast sum = 1 | `SOFTMAX_CHIP_FAST` | Probabilities sum | TESTED |
| Fast vs precise | `SOFTMAX_CHIP_FAST` | Same argmax | TESTED |
| Argmax basic | `ARGMAX_CHIP` | Correct index | TESTED |
| Argmax tie | `ARGMAX_CHIP` | Returns first | TESTED |
| In-place alias | `SOFTMAX_CHIP` | out can alias x | TESTED |

### Normalization (9 tests)

| Test | Function | Property | Status |
|------|----------|----------|--------|
| LayerNorm basic | `LAYERNORM_CHIP` | Mean=0, var=1 | TESTED |
| LayerNorm identity | `LAYERNORM_CHIP` | gamma=1,beta=0 | TESTED |
| LayerNorm affine | `LAYERNORM_CHIP` | gamma/beta applied | TESTED |
| RMSNorm basic | `RMSNORM_CHIP` | Correct scaling | TESTED |
| RMSNorm vs LayerNorm | `RMSNORM_CHIP` | Different (no mean sub) | TESTED |
| BatchNorm basic | `BATCHNORM_CHIP` | Correct normalization | TESTED |
| BatchNorm identity | `BATCHNORM_CHIP` | gamma=1,beta=0 | TESTED |
| All ones input | Various | No division by zero | TESTED |
| Epsilon behavior | Various | eps prevents NaN | TESTED |

### CfC Sparse (14 tests)

| Test | Function | Property | Status |
|------|----------|----------|--------|
| SPARSE vs LUT 100 steps | `CFC_CELL_SPARSE` | Bit-identical | TESTED |
| SPARSE stability 1000 steps | `CFC_CELL_SPARSE` | Bounded, finite | TESTED |
| SPARSE vs FIXED (precise) | `CFC_CELL_SPARSE` | Within LUT tolerance | TESTED |
| SPARSE zero input | `CFC_CELL_SPARSE` | Finite output | TESTED |
| SPARSE all-zero weights | `CFC_CELL_SPARSE` | Bias-only path works | TESTED |
| cfc_build_sparse transposed=0 | `cfc_build_sparse` | GEMM-native layout | TESTED |
| cfc_build_sparse transposed=1 | `cfc_build_sparse` | Demo layout | TESTED |
| cfc_build_sparse threshold | `cfc_build_sparse` | Respects threshold | TESTED |
| Sparse row sentinel | `CfcSparseRow` | -1 terminated | TESTED |
| Sparse max indices | `cfc_build_sparse` | Handles full row | TESTED |
| LUT vs GENERIC | `CFC_CELL_LUT` | Within tolerance | TESTED |
| FIXED vs GENERIC | `CFC_CELL_FIXED` | Exact match | TESTED |
| Precompute decay | `cfc_precompute_decay` | Exact match | TESTED |
| GENERIC correctness | `CFC_CELL_GENERIC` | Matches yinsen_cfc_cell | TESTED |

---

## test_shapes.c (44 tests)

### Logic Gates — Truth Tables (26 tests, PROVEN)

| Test | Function | All combinations verified |
|------|----------|--------------------------|
| XOR | `yinsen_xor` | 4/4 |
| AND | `yinsen_and` | 4/4 |
| OR | `yinsen_or` | 4/4 |
| NOT | `yinsen_not` | 2/2 |
| NAND | `yinsen_nand` | 4/4 |
| NOR | `yinsen_nor` | 4/4 |
| XNOR | `yinsen_xnor` | 4/4 |

### Full Adder (8 tests, PROVEN)

All 8 input combinations (a, b, carry_in) verified.

### 8-Bit Adder (1 test, PROVEN)

65,536/65,536 input pairs verified.

### Activations (7 tests, TESTED)

ReLU (3), Sigmoid (2), Tanh (2) property tests.

### Softmax (2 tests, TESTED)

Sum=1, numerical stability.

### MatMul (1 test, TESTED)

2x3 @ 3x2 spot check.

---

## test_cfc.c (13 tests)

| Test | Function | Property | Status |
|------|----------|----------|--------|
| Determinism | `yinsen_cfc_cell` | Same in -> same out | TESTED |
| Bounded | `yinsen_cfc_cell` | Zero in doesn't explode | TESTED |
| Decay | `yinsen_cfc_cell` | State -> 0 | TESTED |
| Stability | `yinsen_cfc_cell` | 10K iterations | TESTED |
| Output | `yinsen_cfc_output` | Finite values | TESTED |
| Softmax | `yinsen_cfc_output_softmax` | Sum = 1.0 | TESTED |
| tau=0 | `yinsen_cfc_cell` | Returns NaN | TESTED |
| tau<0 | `yinsen_cfc_cell` | Returns NaN | TESTED |
| tau=1e-10 | `yinsen_cfc_cell` | Finite (extreme decay) | TESTED |
| tau=1e10 | `yinsen_cfc_cell` | Finite (no decay) | TESTED |
| dt=0 | `yinsen_cfc_cell` | Finite (no decay) | TESTED |
| dt<0 | `yinsen_cfc_cell` | Finite (amplification) | TESTED |
| tau per-neuron | `yinsen_cfc_cell` | Per-neuron tau works | TESTED |

---

## test_ternary.c (55 tests)

Encoding (6), dot product (3), 2x2 exhaustive (81 configs, PROVEN), quantization (4), pack/unpack (2), absmean (2), int8 (3), sparsity (2), plus additional property tests.

---

## test_cfc_ternary.c (13 tests)

Same structure as test_cfc.c but for ternary CfC cell.

---

## test_ternary_4x4.c (1 test, 43M verifications)

43,046,721 weight configurations (3^16), all verified against float reference.

---

## test_encoding_canonical.c (24 tests)

Canonical encoding verification: encode/decode roundtrip, constants, cross-backend consistency, reserved encoding behavior.

---

## test_falsify.c (38 tests)

Adversarial tests: zeros/empty, extremes, invalid inputs (NaN/Inf, negative dt, zero tau), overflow, alignment, CfC extremes, stress.

---

## test_entromorph.c (11 tests)

Component tests (8 pass correctly) + convergence tests (2 pass but are MISLEADING — FALSIFIED).

---

## Summary

| File | Tests | Status |
|------|-------|--------|
| test_chips.c | 105 | 105 pass |
| test_shapes.c | 44 | 44 pass |
| test_cfc.c | 13 | 13 pass |
| test_ternary.c | 55 | 55 pass |
| test_cfc_ternary.c | 13 | 13 pass |
| test_encoding_canonical.c | 24 | 24 pass |
| test_falsify.c | 38 | 37 pass, 1 documented |
| test_entromorph.c | 11 | 11 pass (2 misleading) |
| test_ternary_4x4.c | 1 | 43M verified |
| **Total** | **230** | **All pass** |

Additionally: exhaustive 4x4 proof (43,046,721 configurations), Metal GPU tests (Swift package, separate runner).

---

## Metal GPU Tests (not counted in C total)

Run via `swift run yinsen-metal-tests` in `metal/`.

- 4x4 CPU exhaustive (81) + GPU matvec
- 8x8 boundary (7) + random (100K) + linearity (1K)
- Tiled: identity, negation, mixed, 1K fuzz at 16x16
- All using canonical encoding
