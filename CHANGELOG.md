# Changelog

All notable changes to Yinsen are documented in this file.

Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

---

## [Unreleased]

Nothing yet.

---

## [0.1.0] - 2026-01-26

### Added

#### Core Implementation
- **Logic gates as polynomials** (`apu.h`)
  - XOR, AND, OR, NOT, NAND, NOR, XNOR
  - Exhaustively verified for all binary inputs
- **Arithmetic** (`apu.h`)
  - Full adder (8/8 combinations verified)
  - 8-bit ripple carry adder (65,536/65,536 combinations verified)
- **ONNX-compatible operations** (`onnx_shapes.h`)
  - Activations: ReLU, Sigmoid, Tanh, GELU, SiLU
  - Softmax with numerical stability
  - MatMul
- **CfC cell** (`cfc.h`)
  - Gated recurrence with explicit time constant
  - Determinism and stability tested (10K iterations)
- **Ternary weight system** (`ternary.h`)
  - 2-bit trit encoding for {-1, 0, +1}
  - Multiplication-free dot product and matvec
  - Pack/unpack for 4x memory compression
  - Exhaustive 2x2 matvec test (81/81 configurations)
- **Ternary CfC cell** (`cfc_ternary.h`)
  - CfC with ternary weights
  - 4.4x memory compression vs float CfC (measured)
  - Determinism and stability tested (1K iterations)
- **EntroMorph evolution engine** (`entromorph.h`)
  - Present but UNTESTED - no convergence verification

#### Tests
- `test_shapes.c` - 44 tests for logic, arithmetic, activations
- `test_cfc.c` - 6 tests for CfC cell
- `test_ternary.c` - 32 tests for ternary primitives
- `test_cfc_ternary.c` - 6 tests for ternary CfC

#### Examples
- `hello_xor.c` - Logic gate demonstration
- `hello_ternary.c` - Ternary weight demonstration

#### Documentation
- `README.md` - Project overview with ternary thesis
- `docs/THEORY.md` - Mathematical foundations
- `docs/API.md` - Function reference
- `docs/EXAMPLES.md` - Usage guide
- `docs/CLAIMS.md` - Verification claims register

#### Research Process
- `journal/LMM.md` - Lincoln Manifold Method for exploration
- `journal/scratchpad/` - Active exploration workspace
- `journal/archive/` - Completed explorations
- Retention policy: never delete anything

### Verified (PROVEN)
- All 7 logic gates (exhaustive truth tables)
- Full adder (8/8 input combinations)
- 8-bit adder (65,536/65,536 combinations)
- 2x2 ternary matvec (81/81 weight configurations)

### Tested (Single Platform)
- Activations (property tests)
- Softmax (sum=1, numerical stability)
- MatMul (spot checks)
- CfC cell (determinism, stability)
- Ternary CfC cell (determinism, stability, compression)
- Ternary pack/unpack (roundtrip lossless)

### Not Yet Tested
- EntroMorph evolution convergence
- Cross-platform determinism
- CfC equivalence to ODE solution
- End-to-end network solving a task

---

## Commit History

| Hash | Date | Description |
|------|------|-------------|
| `504de5a` | 2026-01-26 | Document ternary as core architectural thesis |
| `73d3e95` | 2026-01-26 | Add ternary CfC cell |
| `d9cc04d` | 2026-01-26 | Add ternary weight system - the missing core of TriX |
| `327b1b9` | 2026-01-26 | Honest reckoning: remove unearned claims, add claims register |
| `1937acf` | 2026-01-26 | Rewrite README with certifiability-first throughline |
| `5121236` | 2026-01-26 | LMM exploration: identify throughline |
| `8638130` | 2026-01-26 | Establish retention policy |
| `4a7d097` | 2026-01-26 | Update LMM to use journal/scratchpad/ |
| `0e14687` | 2026-01-26 | Add Lincoln Manifold Method to journal |
| `c002841` | 2026-01-26 | Add comprehensive documentation |
| `e75d733` | 2026-01-26 | Initial commit: verified frozen computation primitives |

---

## Versioning

This project uses [Semantic Versioning](https://semver.org/):
- **MAJOR**: Breaking changes to API or verified properties
- **MINOR**: New features, new verified primitives
- **PATCH**: Bug fixes, documentation, tests

Until 1.0.0, the API is unstable and may change without notice.
