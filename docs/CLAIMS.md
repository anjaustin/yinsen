# Claims Register

This document tracks every claim made in the yinsen repository and its verification status.

## Verification Levels

| Level | Meaning |
|-------|---------|
| **PROVEN** | Exhaustively tested, all cases verified |
| **TESTED** | Property tests pass on single platform |
| **UNTESTED** | Code exists but no tests |
| **HYPOTHESIS** | Conceptual claim, not a testable property |

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

## EntroMorph (entromorph.h)

| Claim | Level | Evidence |
|-------|-------|----------|
| Evolution converges | **UNTESTED** | No convergence tests |
| Evolution produces useful networks | **UNTESTED** | No benchmark tasks |
| Genome export produces valid C | **UNTESTED** | No export tests |
| Mutation rates are appropriate | **UNTESTED** | No tuning experiments |

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

- 2026-01-26: Initial claims register created during skeptic review
