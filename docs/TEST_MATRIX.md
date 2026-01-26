# Test Matrix

Complete mapping of every test to its target code and verification level.

**Generated:** 2026-01-26  
**Platform:** darwin/arm64

---

## test_shapes.c (44 tests)

### Logic Gates - Truth Tables

| Test | Function | Input | Expected | Status |
|------|----------|-------|----------|--------|
| XOR(0,0) | `yinsen_xor` | 0,0 | 0 | **PROVEN** |
| XOR(0,1) | `yinsen_xor` | 0,1 | 1 | **PROVEN** |
| XOR(1,0) | `yinsen_xor` | 1,0 | 1 | **PROVEN** |
| XOR(1,1) | `yinsen_xor` | 1,1 | 0 | **PROVEN** |
| AND(0,0) | `yinsen_and` | 0,0 | 0 | **PROVEN** |
| AND(0,1) | `yinsen_and` | 0,1 | 0 | **PROVEN** |
| AND(1,0) | `yinsen_and` | 1,0 | 0 | **PROVEN** |
| AND(1,1) | `yinsen_and` | 1,1 | 1 | **PROVEN** |
| OR(0,0) | `yinsen_or` | 0,0 | 0 | **PROVEN** |
| OR(0,1) | `yinsen_or` | 0,1 | 1 | **PROVEN** |
| OR(1,0) | `yinsen_or` | 1,0 | 1 | **PROVEN** |
| OR(1,1) | `yinsen_or` | 1,1 | 1 | **PROVEN** |
| NOT(0) | `yinsen_not` | 0 | 1 | **PROVEN** |
| NOT(1) | `yinsen_not` | 1 | 0 | **PROVEN** |
| NAND(0,0) | `yinsen_nand` | 0,0 | 1 | **PROVEN** |
| NAND(0,1) | `yinsen_nand` | 0,1 | 1 | **PROVEN** |
| NAND(1,0) | `yinsen_nand` | 1,0 | 1 | **PROVEN** |
| NAND(1,1) | `yinsen_nand` | 1,1 | 0 | **PROVEN** |
| NOR(0,0) | `yinsen_nor` | 0,0 | 1 | **PROVEN** |
| NOR(0,1) | `yinsen_nor` | 0,1 | 0 | **PROVEN** |
| NOR(1,0) | `yinsen_nor` | 1,0 | 0 | **PROVEN** |
| NOR(1,1) | `yinsen_nor` | 1,1 | 0 | **PROVEN** |
| XNOR(0,0) | `yinsen_xnor` | 0,0 | 1 | **PROVEN** |
| XNOR(0,1) | `yinsen_xnor` | 0,1 | 0 | **PROVEN** |
| XNOR(1,0) | `yinsen_xnor` | 1,0 | 0 | **PROVEN** |
| XNOR(1,1) | `yinsen_xnor` | 1,1 | 1 | **PROVEN** |

### Full Adder - All 8 Combinations

| Test | Function | Input (a,b,c) | Expected (sum,carry) | Status |
|------|----------|---------------|----------------------|--------|
| FA(0,0,0) | `yinsen_full_adder` | 0,0,0 | 0,0 | **PROVEN** |
| FA(0,0,1) | `yinsen_full_adder` | 0,0,1 | 1,0 | **PROVEN** |
| FA(0,1,0) | `yinsen_full_adder` | 0,1,0 | 1,0 | **PROVEN** |
| FA(0,1,1) | `yinsen_full_adder` | 0,1,1 | 0,1 | **PROVEN** |
| FA(1,0,0) | `yinsen_full_adder` | 1,0,0 | 1,0 | **PROVEN** |
| FA(1,0,1) | `yinsen_full_adder` | 1,0,1 | 0,1 | **PROVEN** |
| FA(1,1,0) | `yinsen_full_adder` | 1,1,0 | 0,1 | **PROVEN** |
| FA(1,1,1) | `yinsen_full_adder` | 1,1,1 | 1,1 | **PROVEN** |

### 8-Bit Adder - Exhaustive

| Test | Function | Coverage | Status |
|------|----------|----------|--------|
| 256×256 | `yinsen_ripple_add_8bit` | 65,536/65,536 | **PROVEN** |

### Activations

| Test | Function | Property | Status |
|------|----------|----------|--------|
| ReLU positive | `yinsen_relu` | relu(2) = 2 | TESTED |
| ReLU negative | `yinsen_relu` | relu(-2) = 0 | TESTED |
| ReLU zero | `yinsen_relu` | relu(0) = 0 | TESTED |
| Sigmoid zero | `yinsen_sigmoid` | σ(0) = 0.5 | TESTED |
| Sigmoid range | `yinsen_sigmoid` | 0 < σ(x) < 1 | TESTED |
| Tanh zero | `yinsen_tanh` | tanh(0) = 0 | TESTED |
| Tanh range | `yinsen_tanh` | -1 < tanh(x) < 1 | TESTED |

### Softmax

| Test | Function | Property | Status |
|------|----------|----------|--------|
| Sum = 1 | `yinsen_softmax` | Σ = 1.0 | TESTED |
| Stability | `yinsen_softmax` | Large values OK | TESTED |

### MatMul

| Test | Function | Property | Status |
|------|----------|----------|--------|
| 2×3 @ 3×2 | `yinsen_gemm` | Correct result | TESTED |

---

## test_cfc.c (6 tests)

| Test | Function | Property | Status |
|------|----------|----------|--------|
| Determinism | `yinsen_cfc_cell` | Same in → same out | TESTED |
| Bounded | `yinsen_cfc_cell` | Zero in doesn't explode | TESTED |
| Decay | `yinsen_cfc_cell` | State → 0 | TESTED |
| Stability | `yinsen_cfc_cell` | 10K iterations | TESTED |
| Output | `yinsen_cfc_output` | Finite values | TESTED |
| Softmax | `yinsen_cfc_output_softmax` | Σ = 1.0 | TESTED |

---

## test_ternary.c (55 tests)

### Encoding

| Test | Function | Property | Status |
|------|----------|----------|--------|
| +1 encode | `ternary_encode` | +1 → 0b01 | TESTED |
| 0 encode | `ternary_encode` | 0 → 0b00 | TESTED |
| -1 encode | `ternary_encode` | -1 → 0b11 | TESTED |
| +1 decode | `ternary_decode` | 0b01 → +1 | TESTED |
| 0 decode | `ternary_decode` | 0b00 → 0 | TESTED |
| -1 decode | `ternary_decode` | 0b11 → -1 | TESTED |

### Dot Product

| Test | Function | Property | Status |
|------|----------|----------|--------|
| Basic dot | `ternary_dot` | Correct sum | TESTED |
| Zero weights | `ternary_dot` | Returns 0 | TESTED |
| Mixed weights | `ternary_dot` | Correct sum | TESTED |

### 2×2 Matvec - Exhaustive

| Test | Function | Coverage | Status |
|------|----------|----------|--------|
| All 81 configs | `ternary_matvec` | 81/81 | **PROVEN** |

### Quantization

| Test | Function | Property | Status |
|------|----------|----------|--------|
| Positive → +1 | `ternary_quantize` | 0.6 → +1 | TESTED |
| Negative → -1 | `ternary_quantize` | -0.6 → -1 | TESTED |
| Small → 0 | `ternary_quantize` | 0.1 → 0 | TESTED |
| Boundary | `ternary_quantize` | -0.4 → 0 | TESTED |

### Pack/Unpack

| Test | Function | Property | Status |
|------|----------|----------|--------|
| Roundtrip | `ternary_pack/unpack` | Lossless | TESTED |
| Compression | - | 8 → 2 bytes | TESTED |

### Absmean (BitNet)

| Test | Function | Property | Status |
|------|----------|----------|--------|
| Uniform dist | `ternary_quantize_absmean` | Adapts scale | TESTED |
| Scale calc | `ternary_absmean_scale` | Correct mean | TESTED |

### Int8 Operations

| Test | Function | Property | Status |
|------|----------|----------|--------|
| Quantize | `ternary_quantize_activations` | Range [-127,127] | TESTED |
| Dequantize | `ternary_dequantize_activations` | Roundtrip OK | TESTED |
| Int8 dot | `ternary_dot_int8` | Matches float | TESTED |

### Sparsity

| Test | Function | Property | Status |
|------|----------|----------|--------|
| Count zeros | `ternary_count_zeros` | Correct count | TESTED |
| Sparsity % | `ternary_sparsity` | Correct ratio | TESTED |

---

## test_cfc_ternary.c (6 tests)

| Test | Function | Property | Status |
|------|----------|----------|--------|
| Determinism | `yinsen_cfc_ternary_cell` | Same in → same out | TESTED |
| Bounded | `yinsen_cfc_ternary_cell` | Random inputs OK | TESTED |
| Stability | `yinsen_cfc_ternary_cell` | 1K iterations | TESTED |
| Output | `yinsen_cfc_ternary_output` | Finite values | TESTED |
| Memory | - | 52 vs 228 bytes | TESTED |
| Softmax | `yinsen_cfc_ternary_output_softmax` | Σ = 1.0 | TESTED |

---

## test_falsify.c (38 tests)

### Zeros/Empty

| Test | Target | Result | Status |
|------|--------|--------|--------|
| Absmean zeros | `ternary_quantize_absmean` | All 0 | PASS |
| Zero-length dot | `ternary_dot` | Returns 0 | PASS |
| Zero-length quantize | `ternary_quantize` | No crash | PASS |

### Extremes

| Test | Target | Result | Status |
|------|--------|--------|--------|
| Large (1e30) | `ternary_quantize` | ±1 | PASS |
| Denormal | `ternary_quantize` | No crash | PASS |
| Single element | All | Works | PASS |

### Invalid Inputs

| Test | Target | Result | Status |
|------|--------|--------|--------|
| NaN | Various | Propagates | KNOWN |
| Inf | Various | Propagates | KNOWN |
| Negative dt | CfC | Amplifies | KNOWN |
| Zero tau | CfC | decay=0 | KNOWN |

### Overflow

| Test | Target | Result | Status |
|------|--------|--------|--------|
| Int32 @ 1000 | `ternary_dot_int8` | No overflow | PASS |

### Alignment

| Test | Target | Result | Status |
|------|--------|--------|--------|
| Length 1 | `ternary_dot` | Works | PASS |
| Length 2 | `ternary_dot` | Works | PASS |
| Length 3 | `ternary_dot` | Works | PASS |
| Length 5 | `ternary_dot` | Works | PASS |

### CfC Extremes

| Test | Target | Result | Status |
|------|--------|--------|--------|
| dt=0.0001 | CfC | Finite | PASS |
| dt=1000 | CfC | Finite | PASS |
| dt=0 | CfC | Finite | PASS |
| tau=1e-10 | CfC | Finite | PASS |

### Stress

| Test | Target | Result | Status |
|------|--------|--------|--------|
| 10K iterations | CfC ternary | Bounded | PASS |

### Reserved Encoding

| Test | Target | Result | Status |
|------|--------|--------|--------|
| 0b10 encoding | `ternary_decode` | Treated as 0 | PASS |

---

## test_entromorph.c (11 tests)

### Component Tests

| Test | Function | Property | Status |
|------|----------|----------|--------|
| RNG determinism | `entro_rng_seed` | Same seed → same | TESTED |
| RNG different | `entro_rng_seed` | Diff seed → diff | TESTED |
| RNG float range | `entro_rng_float` | [0, 1) | TESTED |
| RNG gaussian | `entro_rng_gaussian` | mean/std OK | TESTED |
| Genesis dims | `entro_genesis` | Correct dims | TESTED |
| Genesis tau | `entro_genesis` | Range [0.01,100] | TESTED |
| Mutation | `entro_mutate` | Changes weights | TESTED |
| Genome→params | `entro_genome_to_params` | Valid conversion | TESTED |

### Convergence Tests

| Test | Function | Property | Status |
|------|----------|----------|--------|
| XOR convergence | Evolution loop | 100/100 converge | **MISLEADING** |
| Multi-run | Evolution loop | 5/5 converge | **MISLEADING** |
| Export | `entro_export_header` | Valid C | TESTED |

---

## Summary

| File | Tests | PROVEN | TESTED | FALSIFIED | KNOWN |
|------|-------|--------|--------|-----------|-------|
| test_shapes.c | 44 | 36 | 8 | - | - |
| test_cfc.c | 6 | - | 6 | - | - |
| test_ternary.c | 55 | 81* | 55 | - | - |
| test_cfc_ternary.c | 6 | - | 6 | - | - |
| test_falsify.c | 38 | - | 34 | - | 4 |
| test_entromorph.c | 11 | - | 8 | 2** | - |
| **Total** | **160** | **117** | **117** | **2** | **4** |

\* 81 2×2 matvec configurations in exhaustive test  
\** Convergence tests pass but results are meaningless (FALSIFIED)
