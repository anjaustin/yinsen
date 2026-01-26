# Changelog

All notable changes to Yinsen are documented in this file.

Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

---

## [Unreleased]

### Added
- **EntroMorph convergence tests** (`test/test_entromorph.c`)
  - 11 tests covering RNG, genesis, mutation, and XOR convergence
  - **Critical result:** Evolution converges on XOR in 5/5 runs (typical: 10-30 generations)
  - Tournament selection with elitism
  - Genome export to C header
- **Absmean quantization** (`ternary_quantize_absmean`) - BitNet b1.58 method
  - Adapts to weight distribution automatically
  - `ternary_absmean_scale()` to get scale factor
- **Int8 activation quantization** - for fully integer forward pass
  - `ternary_quantize_activations()` - symmetric per-tensor quantization
  - `ternary_dequantize_activations()` - convert back to float
  - `TernaryQuantParams` struct for scale/zero_point
- **Integer ternary operations**
  - `ternary_dot_int8()` - integer-only dot product
  - `ternary_matvec_int8()` - integer-only matrix-vector multiply
- **Energy estimation** - based on Horowitz 2014 (7nm estimates)
  - `ternary_matvec_energy_pj()` - ternary path energy
  - `float_matvec_energy_pj()` - float path energy
  - `ternary_energy_savings_ratio()` - ~43x savings for matmul
- **Extended sparsity statistics**
  - `TernaryStats` struct with full distribution
  - `ternary_count_zeros()`, `ternary_count_positive()`, `ternary_count_negative()`
  - `ternary_sparsity()` - fraction of zero weights
  - `ternary_stats()` - compute all statistics

### Changed
- Updated terminology to "1.58-bit" (log2(3) = 1.58) per BitNet convention
- Documentation now explains zero as "explicit feature filtering"

### Tests
- **test_entromorph.c** - 11 tests for evolution engine
  - RNG determinism and distribution
  - Genesis and mutation
  - XOR convergence (THE critical test)
  - Genome export
- Added 23 new tests to test_ternary.c (55 total)
  - Absmean quantization tests
  - Int8 quantization roundtrip tests
  - Integer dot product vs float comparison
  - Energy estimation validation
- **Total: 160 tests** (159 pass, 1 documented behavior)

### Research
- Added BitNet b1.58 comparison analysis (`journal/scratchpad/bitnet_comparison.md`)
- Added actionable learnings document (`journal/scratchpad/bitnet_learnings.md`)
- LMM exploration of Yinsen's potential (`journal/scratchpad/potential_*.md`)

### Falsification Testing
- **New test suite:** `test/test_falsify.c` with 38 adversarial tests
- **Edge cases documented:** `docs/EDGE_CASES.md`
- Verified robust against: zeros, denormals, large values, misaligned lengths
- Verified 10K iteration stability
- Documented known behaviors: NaN propagation, negative dt, zero tau

### Documentation
- **API.md:** Complete ternary.h and cfc_ternary.h reference
- **EXAMPLES.md:** 5 new ternary examples (7-11)
- **CLAIMS.md:** Updated with falsification results

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
  - Genome representation, mutation, export
  - Now TESTED: converges on XOR

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
- Cross-platform determinism
- CfC equivalence to ODE solution
- Evolution on tasks beyond XOR
- Ternary evolution (currently float-only)

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
