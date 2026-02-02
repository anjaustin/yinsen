# Changelog

All notable changes to Yinsen are documented in this file.

Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

---

## [Unreleased]

### `a320dfe` — CFC_CELL_SPARSE: Zero-Multiply CfC (February 2026)

**The moneyball.** Ternary weights at threshold=0.10 are 81% zero. Sparse index lists skip all zeros: 31 adds instead of 160 MACs. Zero multiplies in the GEMM. Bit-identical to CFC_CELL_LUT.

#### Added
- **`CFC_CELL_SPARSE`** in `cfc_cell_chip.h` — zero-multiply CfC step using sparse index lists
- **`CfcSparseRow`** / **`CfcSparseWeights`** types — int8 index lists for +1/-1 weights, -1 sentinel terminated
- **`cfc_build_sparse()`** — builds sparse from float weights at init (handles GEMM layout via `transposed` flag)
- **14 new tests** in `test_chips.c` for CfC sparse variant:
  - Bit-identity with CFC_CELL_LUT over 100 steps
  - Stability, correctness against CFC_CELL_FIXED (precise, with LUT accuracy tolerance)

#### Performance
- CFC_CELL_SPARSE: **20 ns/step** (2.73x faster than GENERIC baseline)
- CFC_CELL_LUT: 35 ns/step (1.54x)
- CFC_CELL_GENERIC: 54 ns/step (baseline)

---

### `50f80bd` — Chip Forge, Enrollment Demos, Falsification Probes (February 2026)

The big commit. Eight frozen primitives, three enrollment demos validated on live data, four keystroke falsification probes, and ternary quantization validation.

#### Added — Chip Forge (8 primitives)
- **`include/chips/cfc_cell_chip.h`** — CfC cell (GENERIC, FIXED, LUT variants) + `cfc_precompute_decay()`
- **`include/chips/activation_chip.h`** — Sigmoid/tanh/exp in 3 tiers:
  - PRECISE (libm): exact
  - FAST3 (degree-3 rational): ~2e-3 error
  - LUT+lerp (256 entries): 4.7e-5 sigmoid error, 200x better than FAST3
  - `ACTIVATION_LUT_INIT()` for one-time table fill (2KB shared read-only)
  - Vectorized variants: `SIGMOID_VEC_CHIP`, `TANH_VEC_CHIP`, etc.
- **`include/chips/gemm_chip.h`** — GEMM_CHIP (full), GEMM_CHIP_BIASED, GEMM_CHIP_BARE
- **`include/chips/decay_chip.h`** — DECAY_CHIP_SCALAR, DECAY_CHIP_SHARED, DECAY_CHIP_VECTOR, DECAY_CHIP_FAST variants
- **`include/chips/ternary_dot_chip.h`** — TERNARY_DOT_CHIP (float), INT8, INT16 variants
- **`include/chips/fft_chip.h`** — Radix-2 Cooley-Tukey FFT, IFFT, magnitude, power spectrum
- **`include/chips/softmax_chip.h`** — SOFTMAX_CHIP, SOFTMAX_CHIP_FAST, ARGMAX_CHIP
- **`include/chips/norm_chip.h`** — LAYERNORM_CHIP, RMSNORM_CHIP, BATCHNORM_CHIP
- **`test/test_chips.c`** — 91 chip tests (later expanded to 105 with sparse)

#### Added — Enrollment Demos
- **`examples/iss_telemetry.c`** — 8-channel CfC anomaly detector for ISS
  - v2 pre-scaling fix (root cause: CMG at ~0.001g was invisible to CfC gates)
  - 4-phase pipeline: Calibrate → Enroll → Normal Test → Anomaly Test
  - CfC beats 3-sigma on oscillation shifts (20/20 vs 0/20)
  - Memory: 3,552 bytes total (5.4% of 64KB L1)
  - Live ISS Lightstreamer connection validated with 384 detection samples
- **`examples/keystroke_biometric.c`** — v3 hybrid linear discriminant
  - Full falsification cycle through probes 1-4
  - v1 falsified (seed artifact), v2 honest, linear discriminant via power iteration PCA
  - v3: 0.3*mean + 0.7*PCA scoring. Easy 20/20, Medium 20/20, Hard 16/20, Control 10/20
  - 268-byte discriminant, 110 ns/keystroke
- **`scripts/iss_websocket.py`** — ISS Lightstreamer v2 bridge
- **`scripts/seismic_seedlink.py`** — GFZ SeedLink bridge (100 Hz)

#### Added — Falsification Probes
- **`experiments/keystroke_probes/keystroke_probe{1,2,3,4}.c`** — Keystroke falsification cycle
- **`experiments/iss_probes/iss_probe{1,2,3}.c`** — ISS falsification probes
- **`experiments/ternary_quantization/quant_probe1.c`** — Bridge experiment: float vs ternary CfC side-by-side
  - Sweet spot: threshold 0.10-0.20. 99% quality, all anomalies detected.
  - At t=0.10: 160 weights → 31 nonzero (19.4%), 129 zero (80.6%)
  - Weight memory: 640 bytes → 40 bytes (16x compression)
- **`experiments/ternary_quantization/quant_probe2_fpu.c`** — FPU optimization probe
  - Ternary-as-float GEMM: FPU multiplies 1.0 as fast as 0.3
  - LUT+lerp: 200x more accurate than FAST3, only 4ns more per CfC step
  - Sparse ternary: 31 adds instead of 160 MACs, 2.73x faster

#### Added — LMM Journals
- `journal/scratchpad/real_gap_{raw,nodes,reflect,synth}.md` — Real Gap LMM (all 4 phases)
- `journal/scratchpad/iss_improvement_{raw,nodes,reflect,synth}.md` — v2 pre-scaling
- `journal/scratchpad/keystroke_lda_{raw,nodes,reflect,synth}.md` — Linear discriminant
- `journal/scratchpad/chip_applications_{raw,nodes,reflect,synth}.md` — Application selection

---

### `abb186a` — Seismic Detector with Tau Ablation (February 2026)

#### Added
- **`examples/seismic_detector.c`** — 3-channel CfC processor for 100 Hz seismic waveforms
  - Built-in tau ablation (matched vs ISS-tau vs constant-tau)
  - STA/LTA baseline comparison
  - Live GFZ SeedLink validated: 7,351 samples from GE.STU Stuttgart
  - Performance: 58 ns/channel/step (GENERIC), 1,768 bytes total
  - Real-time headroom: 148,810x at 100 Hz

#### Discovered — The Tau Principle
- Tau differentiation emerges when decay dynamic range R = max(decay)/min(decay) is commensurate with signal temporal structure T = max(timescale)/min(timescale)
- Seismic R=2700x matches T=3000x → tau matters (2.2-2.4x faster detection)
- ISS R=2700x but T~1 (slow sensors) → tau doesn't help

#### Added — LMM Journals
- `journal/scratchpad/seismic_tau_{raw,nodes,reflect,synth}.md` — Tau ablation LMM

---

### CRITICAL: EntroMorph Evolution FALSIFIED

**The evolution engine does NOT work.** Falsification testing revealed:

- 100/100 runs "converge" but 0/100 have >10% confidence margin
- Solutions predict ~0.5 for all inputs (random chance determines correctness)
- Cross-entropy fitness REWARDS staying near 0.5
- 1,000,000 random genomes searched: ZERO had meaningful confidence
- Solutions fragile to 1% noise (88% accuracy vs expected 100%)

**Root cause:** Genesis initialization + cross-entropy fitness create a trap where evolution finds numerical coincidences, not learned functions.

**See:** `docs/FALSIFICATION_ENTROMORPH.md` for full analysis.

### Added

#### Phase 2: Encoding Unification
- **`include/trit_encoding.h`** — single source of truth for 2-bit encoding
  - Canonical: `00`=0, `01`=+1, `10`=-1, `11`=reserved
  - All backends migrated: `ternary.h`, Metal shaders, NEON, SME
  - `TRIT_NEG` changed from `0x3` to `0x2` in `ternary.h`
  - `test/test_encoding_canonical.c` — 24 tests verifying cross-backend consistency

#### Phase 3: Metal Kernel
- **`metal/kernels/ternary_matvec_tiled.metal`** — threadgroup-cooperative kernel
  - Shared memory + `simd_sum` two-level reduction
  - One threadgroup per output row, up to 256 threads
  - Both `ternary_matvec_tiled` and `ternary_matvec_tiled_bias` variants
  - Include-guarded trit primitives (`#ifndef YINSEN_TRIT_PRIMITIVES`)
- **Swift API rewrite:** `YinsenMetal.swift` now loads .metal from bundle/filesystem (no more hardcoded embedded string)
- **Encoding bug fixed in Swift layer:** All Swift CPU references and hardcoded bit patterns updated from old encoding (`11`=-1) to canonical (`10`=-1)
- **Renamed** `ternary_dot8_simd` → `ternary_dot8_vectorized` (honesty: no Metal simdgroup intrinsics)
- **Retired** `ternary_matvec_8x8_single` (marked as retired in-file)
- **Benchmarks (debug):** 1024×1024 cooperative 0.273ms (1.71x faster than per-row), 3.84 Gop/s

#### Phase 4: Bug Fixes
- **VLA undefined behavior fixed:** `decay[]` array in `cfc.h` and `cfc_ternary.h` now initialized to NAN (was uninitialized VLA)
- **W_out comment fixed:** Both CfC headers now correctly say `[output_dim, hidden_dim]`
- **VLA stack budget documented** in both headers
- **7 new tau edge-case tests per CfC file** (13 tests each, up from 6)
  - tau=0 → NaN, tau<0 → NaN, tau=1e-10, tau=1e10, dt=0, dt<0, per-neuron tau

#### Training Experiments
- **v1 (`train_sine.c`):** First proof ternary CfC can learn. Float MSE 0.032, Ternary MSE 0.228
- **v2 (`train_sine_v2.c`):** 2x2x2 factorial experiment with Adam optimizer
  - Best: Config 5 (STE, no distill, h=32) MSE **0.000362** — 633x improvement over v1
  - Width dominates; smoothstep neutral; trajectory distillation hurts
- **v3 (`diagnostic_v3.c`):** Multi-task diagnostics
  - Multi-Freq: 1.47x degradation (no wall)
  - Copy-8-20: Float teacher itself failed (MSE 0.81)
  - **Lorenz: 12.69x degradation ("THE WALL")** — float 0.000117, ternary 0.001490
  - Depth falsified: 2-layer 155x worse than 1-layer
  - Ring voxel CfC: ternary MSE 0.000894 (beats flat via width+sparsity, not geometry; autocorrelation 0.096)

#### Other Additions
- **EntroMorph convergence tests** (`test/test_entromorph.c`)
  - 11 tests covering RNG, genesis, mutation, and XOR convergence
  - Tests PASS but results are MISLEADING (see falsification)
- **EntroMorph falsification tests**
  - `test/test_entromorph_falsify.c` - 11 tests identifying the problem
  - `test/test_entromorph_deep.c` - confidence analysis
  - `test/test_entromorph_diagnosis.c` - root cause analysis
  - `docs/FALSIFICATION_ENTROMORPH.md` - full report
- **Absmean quantization** (`ternary_quantize_absmean`) - BitNet b1.58 method
- **Int8 activation quantization** - for fully integer forward pass
- **Integer ternary operations** - `ternary_dot_int8()`, `ternary_matvec_int8()`
- **Energy estimation** - based on Horowitz 2014 (7nm estimates)
- **Extended sparsity statistics** - `TernaryStats`, `ternary_sparsity()`, etc.
- **`make test-all`** target in Makefile (runs all test suites)

### Changed
- Retired old LLM-focused `PRD.md` to `journal/archive/PRD_llm_retired.md`
- Rewrote `README.md` with unified identity: "Verified 2-bit computation engine"
- Active roadmap in `PRD_consolidation.md` (all 4 phases marked COMPLETE)
- Updated terminology to "1.58-bit" (log2(3) = 1.58) per BitNet convention
- Fixed `forged_bridge.h` encoding in parent TriX repo
- Fixed misleading SME comment in `sme/ternary_sme.h`

### Tests
- **test_encoding_canonical.c** — 24 tests for canonical encoding verification
- **test_entromorph.c** — 11 tests (components work, convergence misleading)
- **test_cfc.c** — now 13 tests (was 6, added 7 tau edge-case tests)
- **test_cfc_ternary.c** — now 13 tests (was 6, added 7 tau edge-case tests)
- **test_falsify.c** — 38 adversarial tests (4 stale tests fixed for canonical encoding)
- **test_ternary.c** — 55 tests (added absmean, int8, energy estimation)
- **Metal tests** (separate Swift test runner):
  - 4x4 CPU exhaustive (81) + GPU matvec
  - 8x8 boundary (7) + random (100K) + linearity (1K)
  - Tiled: identity, negation, mixed, 1K fuzz at 16x16
- **Chip tests** (`test_chips.c`): 105 tests (GEMM 8, activation 29, decay 14, ternary_dot 7, FFT 16, softmax 10, norm 9, CfC sparse 14) — added in `50f80bd` and `a320dfe`
- **Total C tests: 230** (all pass)

### Documentation
- **API.md:** Complete ternary.h and cfc_ternary.h reference, W_out comment fixed
- **EXAMPLES.md:** 5 new ternary examples (7-11)
- **CLAIMS.md:** Updated with encoding, training, falsification claims
- **VERIFICATION.md:** Updated counts, encoding section, training results, tau tests
- **TEST_MATRIX.md:** Updated function names, encoding, counts
- **THEORY.md:** Updated encoding table, training status
- **EDGE_CASES.md:** Documented all falsification edge cases

### Falsification Testing
- **New test suite:** `test/test_falsify.c` with 38 adversarial tests
- Verified robust against: zeros, denormals, large values, misaligned lengths
- Verified 10K iteration stability (float CfC), 1K iterations (ternary CfC)
- Documented known behaviors: NaN propagation, negative dt, zero tau (→ NaN)

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
- `test_cfc.c` - 6 tests for CfC cell (later expanded to 13)
- `test_ternary.c` - 32 tests for ternary primitives (later expanded to 55)
- `test_cfc_ternary.c` - 6 tests for ternary CfC (later expanded to 13)

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
| `a320dfe` | 2026-02 | Add CFC_CELL_SPARSE: zero-multiply CfC at 20 ns/step (2.73x faster) |
| `50f80bd` | 2026-02 | Add chip forge, enrollment demos, falsification probes, ternary quantization |
| `abb186a` | 2026-02 | Add seismic detector with tau ablation: matched-timescale tau validated |
| `a213e24` | 2026-01 | Add Ghost-Stream and Block-16 kernels: 186 GOP/s peak |
| `1c4e2b4` | 2026-01 | Add I8MM SMMLA kernels for batch>1 workloads |
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
