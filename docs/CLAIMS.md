# Claims Register

This document tracks every claim made in the yinsen repository and its verification status.

**Last Updated:** 2026-01-26  
**Platform:** darwin/arm64  
**Total Tests:** 149 (111 core + 38 falsification)

## Verification Levels

| Level | Meaning | Confidence |
|-------|---------|------------|
| **PROVEN** | Exhaustively tested, all cases verified | 100% |
| **TESTED** | Property tests pass on single platform | High |
| **FALSIFIED** | Attempted to break, found critical failure | DO NOT USE |
| **KNOWN ISSUE** | Edge case with documented behavior | Use with caution |
| **UNTESTED** | Code exists but no tests | Unknown |
| **HYPOTHESIS** | Conceptual claim, not testable | N/A |

## Quick Summary

| Category | PROVEN | TESTED | FALSIFIED | KNOWN |
|----------|--------|--------|-----------|-------|
| Logic gates | 7 | - | - | - |
| Arithmetic | 2 | - | - | - |
| Activations | - | 6 | - | - |
| Matrix ops | - | 2 | - | - |
| CfC | - | 4 | - | 2 |
| Ternary | 2 | 11 | - | 1 |
| Ternary CfC | - | 5 | - | 1 |
| EntroMorph | - | 8 | **1** | - |
| **Total** | **11** | **36** | **1** | **4** |

---

## Logic Gates (apu.h)

| Claim | Level | Evidence |
|-------|-------|----------|
| XOR(a,b) = a + b - 2ab is exact for {0,1} | **PROVEN** | 4/4 truth table in test_shapes.c |
| AND(a,b) = ab is exact for {0,1} | **PROVEN** | 4/4 truth table |
| OR(a,b) = a + b - ab is exact for {0,1} | **PROVEN** | 4/4 truth table |
| NOT(a) = 1 - a is exact for {0,1} | **PROVEN** | 2/2 cases |
| NAND(a,b) = 1 - ab is exact for {0,1} | **PROVEN** | 4/4 truth table |
| NOR(a,b) = 1 - a - b + ab is exact for {0,1} | **PROVEN** | 4/4 truth table |
| XNOR(a,b) = 1 - a - b + 2ab is exact for {0,1} | **PROVEN** | 4/4 truth table |

## Arithmetic (apu.h)

| Claim | Level | Evidence |
|-------|-------|----------|
| Full adder produces correct sum and carry | **PROVEN** | 8/8 combinations in test_shapes.c |
| 8-bit ripple adder produces correct results | **PROVEN** | 65,536/65,536 combinations |

## Activations (onnx_shapes.h)

| Claim | Level | Evidence |
|-------|-------|----------|
| ReLU(x) = max(0, x) | **TESTED** | Property tests |
| Sigmoid(0) = 0.5 | **TESTED** | Spot check |
| Sigmoid output in (0, 1) | **TESTED** | Property test |
| Tanh(0) = 0 | **TESTED** | Spot check |
| Softmax outputs sum to 1 | **TESTED** | Property test |
| Softmax numerically stable | **TESTED** | Large value test |

## Matrix Operations (onnx_shapes.h)

| Claim | Level | Evidence |
|-------|-------|----------|
| MatMul computes correct result | **TESTED** | Spot check vs known values |
| GEMM computes correct result | **TESTED** | Implicit via CfC tests |

## CfC (cfc.h)

| Claim | Level | Evidence |
|-------|-------|----------|
| CfC cell is deterministic | **TESTED** | Same input → same output (single platform) |
| CfC cell outputs are bounded | **TESTED** | Zero input doesn't explode |
| CfC cell decays over time | **TESTED** | State → 0 with zero input |
| CfC cell is numerically stable | **TESTED** | 10,000 iterations without NaN/Inf |
| CfC is equivalent to ODE solution | **UNTESTED** | No comparison performed |
| CfC is deterministic across platforms | **UNTESTED** | Only tested on darwin/arm64 |
| CfC handles irregular dt correctly | **UNTESTED** | No variable-dt tests |

## Ternary Weights (ternary.h)

| Claim | Level | Evidence |
|-------|-------|----------|
| Trit encoding/decoding is lossless | **TESTED** | Pack/unpack roundtrip test |
| Ternary dot product matches integer sum | **TESTED** | Property tests |
| 2x2 ternary matvec is correct | **PROVEN** | 81/81 configurations (all 3^4 weight combos) |
| 4x4 ternary matvec is correct | **PROVEN** | 43,046,721/43,046,721 configurations (all 3^16 weight combos) |
| Ternary quantization finds nearest trit | **TESTED** | Boundary tests (-0.6, -0.4, 0.1, 0.6) |
| Absmean quantization adapts to distribution | **TESTED** | test_absmean_quantize() |
| 4x memory compression vs int8 | **TESTED** | 8 weights → 2 bytes (4 bits each vs 8 bits) |
| Non-multiple-of-4 lengths work correctly | **TESTED** | Lengths 1,2,3,5,6,7 tested |
| Zero-length inputs return 0 | **TESTED** | test_empty_inputs() |
| Large values (1e30) quantize correctly | **TESTED** | test_large_values() |
| Denormal values don't crash | **TESTED** | test_small_values() |
| Reserved encoding (0b10) treated as 0 | **TESTED** | test_reserved_encoding() |
| Int8 dot matches float within quant error | **TESTED** | test_int8_dot() |
| Int32 accumulator safe to 16.9M elements | **TESTED** | Calculated, spot-checked at 1000 |

## Ternary Weights - Known Behaviors

| Behavior | Status | Notes |
|----------|--------|-------|
| NaN/Inf inputs propagate | **DOCUMENTED** | Not validated, see EDGE_CASES.md |
| All-zeros input returns all-zeros output | **TESTED** | Handles gracefully |

## Ternary CfC (cfc_ternary.h)

| Claim | Level | Evidence |
|-------|-------|----------|
| Ternary CfC is deterministic | **TESTED** | Same input → same output |
| Ternary CfC outputs are bounded | **TESTED** | 100 random inputs bounded |
| Ternary CfC is numerically stable | **TESTED** | 10,000 iterations without NaN/Inf |
| Ternary CfC output is valid distribution | **TESTED** | Softmax sums to 1.0 |
| 4.4x memory compression vs float CfC | **TESTED** | 52 bytes vs 228 bytes (measured) |
| Ternary CfC eliminates multiply in forward pass | **HYPOTHESIS** | Conceptually true, not benchmarked |
| Zero dt produces finite output | **TESTED** | decay=1, full retention |
| Large dt (1000) produces finite output | **TESTED** | decay≈0, full update |
| Tiny tau (1e-10) produces finite output | **TESTED** | decay≈0 |
| Zero tau produces decay=0 | **TESTED** | exp(-inf)=0, mathematically valid |

## Ternary CfC - Known Behaviors

| Behavior | Status | Notes |
|----------|--------|-------|
| Negative dt produces amplification | **DOCUMENTED** | Invalid input, see EDGE_CASES.md |
| Zero tau produces full update | **DOCUMENTED** | exp(-dt/0)=exp(-inf)=0 |

## EntroMorph (entromorph.h)

### Component Tests (PASS)

| Claim | Level | Evidence |
|-------|-------|----------|
| RNG is deterministic with same seed | **TESTED** | test_rng_deterministic() |
| RNG produces [0,1) floats | **TESTED** | 1000 samples tested |
| RNG gaussian has correct mean/std | **TESTED** | 10K samples, mean±0.1, std±0.2 |
| Genesis creates valid dimensions | **TESTED** | test_genesis_dimensions() |
| Genesis tau in [0.01, 100] | **TESTED** | test_genesis_tau_range() |
| Mutation changes weights | **TESTED** | test_mutation_changes_weights() |
| Genome to CfC params works | **TESTED** | test_genome_to_params() |
| Genome export produces valid C | **TESTED** | test_export_header() |

### Evolution Convergence (FALSIFIED)

| Claim | Level | Evidence |
|-------|-------|----------|
| Evolution converges on XOR | **MISLEADING** | 100/100 converge but 0/100 have >10% confidence |
| Solutions are useful | **FALSE** | Solutions fragile to 1% noise (88% accuracy) |
| Evolution actually learns | **FALSE** | Solutions cluster at 0.5, don't escape |
| Mutation rates are appropriate | **UNKNOWN** | Can't evaluate without working fitness |

### Root Cause (see docs/FALSIFICATION_ENTROMORPH.md)

| Finding | Implication |
|---------|-------------|
| Genesis produces neutral networks | All predictions ≈ 0.5 |
| Cross-entropy rewards staying near 0.5 | No gradient toward confident solutions |
| 1M random genomes: 0 confident | Initialization fundamentally broken |
| Random solutions have BETTER fitness than confident ones | Fitness function is wrong |

## Cross-Cutting Claims

| Claim | Level | Evidence |
|-------|-------|----------|
| "Deterministic across platforms" | **UNTESTED** | Only one platform tested |
| "Header-only C is auditable" | **HYPOTHESIS** | Organizational claim |
| "No dependencies beyond libc" | **TESTED** | Compiles with just -lm |
| "Primitives are complete" | **HYPOTHESIS** | No formal proof |
| "Primitives are minimal" | **HYPOTHESIS** | No formal proof |

---

## What Would Upgrade a Claim

| Current | Upgrade To | Required Action |
|---------|------------|-----------------|
| UNTESTED → TESTED | Add property tests, pass on one platform |
| TESTED → PROVEN | Exhaustive test or formal proof |
| Single-platform → Cross-platform | CI on Linux, macOS, Windows, ARM, x86 |
| HYPOTHESIS → TESTED | Define falsifiable test, run it |

---

## Changelog

- 2026-01-26: **EntroMorph FALSIFIED** - convergence is misleading, solutions have 0% confidence
- 2026-01-26: EntroMorph TESTED - evolution converges on XOR (5/5 runs)
- 2026-01-26: Added falsification test results and edge case documentation
- 2026-01-26: Added BitNet b1.58 features (absmean, int8, energy estimation)
- 2026-01-26: Added ternary weights and ternary CfC claims
- 2026-01-26: Initial claims register created during skeptic review
