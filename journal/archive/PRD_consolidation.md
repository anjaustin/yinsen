# Yinsen Consolidation PRD: Identity, Encoding, Metal, and Bugs

**Version:** 1.0
**Date:** 2026-01-31
**Status:** ACTIVE
**Supersedes:** PRD.md (LLM-focused roadmap -- retired)

---

## Executive Summary

This PRD addresses four structural problems identified during a cold-eye review of the Yinsen codebase. These are not feature requests. They are load-bearing issues that must be resolved before any forward work makes sense.

| # | Problem | Severity |
|---|---------|----------|
| 1 | The project has two incompatible identities | Architectural |
| 2 | The 2-bit trit encoding is inconsistent across backends | Correctness |
| 3 | The Metal kernels are correctness references, not compute kernels | Performance |
| 4 | `cfc_ternary.h` and `cfc.h` contain undefined behavior | Bug |

---

## Problem 1: Two Incompatible Identities

### Diagnosis

The codebase serves two masters:

**Identity A -- "Auditable CfC for Regulated Industries"**
Lives in: `journal/scratchpad/throughline_synth.md`, `potential_synth.md`, `potential_reflect.md`, `bitnet_comparison.md`
Pitch: "The neural network a regulator can read." Tiny CfC networks (50-10K parameters), nurse/pilot/judge can trace decisions, FDA/IEC certification pathway.

**Identity B -- "Ternary LLM on Apple Silicon"**
Lives in: `PRD.md`
Pitch: Build and deploy a ternary large language model. 25M parameters, 100 tok/s, CLI/API/embedded.

These serve different users, require different infrastructure, have different success criteria, and compete for the same development time. Neither has been executed. Meanwhile, the project's *actual* proven strength -- frozen math shapes and deterministic ternary primitives -- is not adequately represented by either identity.

### Resolution: A Single Identity

Yinsen is a **verified 2-bit computation engine**.

It builds frozen, exhaustively-proven mathematical primitives and deterministic neural building blocks where every weight is {-1, 0, +1} and every operation is multiplication-free. It is not an LLM runtime. It is not competing with BitNet. It is a research library that makes neural computation auditable by making it enumerable.

**One-liner:** Verified ternary primitives. Frozen shapes. Deterministic computation.

**What this identity keeps:**
- Exhaustive verification as a first-class discipline
- Ternary as a 2-bit architecture (not a quantization trick)
- Header-only C, zero dependencies, compiles anywhere
- CfC as the recurrent cell (honest about what is proven and what is not)
- Hardware acceleration (Metal, NEON, SME) as performance backends for proven operations

**What this identity drops:**
- The LLM ambition (Phases 2-6 of the old PRD)
- The "compete with BitNet/LFNs at scale" framing
- The 5-week-to-full-stack timeline
- The API server, CLI, embedded deployment targets (premature)
- The regulatory/certification positioning (no evidence, no market validation)

**What this identity defers:**
- Training. The fundamental question "can ternary networks learn?" is real and important. It should be answered through experimentation, not roadmapped as a product phase. When someone trains a useful ternary network (by any means -- STE, evolution v2, manual construction), the inference stack follows. Not before.

### Deliverables

1. **Retire `PRD.md`.** Move it to `journal/archive/PRD_llm_retired.md` with a header note explaining why.
2. **Update `README.md`.** Remove the "Roadmap" section that references fixing EntroMorph and benchmarking ternary vs float. Replace with the consolidated identity statement and the real roadmap (this PRD's Phases 1-3).
3. **Update the architecture diagram** in `README.md` to reflect the actual stack: APU (PROVEN) -> Ternary Primitives (PROVEN) -> CfC Cells (TESTED) -> Hardware Backends (Metal/NEON/SME). Remove the EntroMorph layer or label it RETIRED.

### Acceptance Criteria

- [x] One identity statement exists in `README.md`, not two in different documents
- [x] No document in the repo promises an LLM, API server, or CLI
- [x] The word "certifiable" does not appear unless accompanied by evidence
- [x] The README accurately describes what exists today

> **Status: COMPLETE** (2026-01-31). PRD.md retired to `journal/archive/PRD_llm_retired.md`. README rewritten with unified identity.

---

## Problem 2: Encoding Inconsistency Across Backends

### Diagnosis

Two mutually incompatible 2-bit trit encodings exist in the codebase:

| 2-bit Pattern | Encoding A (`ternary.h`, Metal) | Encoding B (NEON, SME) |
|:---:|:---:|:---:|
| `00` | 0 | 0 |
| `01` | +1 | +1 |
| `10` | reserved (=0) | **-1** |
| `11` | -1 | **0** |

**Files using Encoding A** (3 files):
- `include/ternary.h` (lines 14-18, 44-46)
- `metal/kernels/ternary_core.metal` (lines 7-12, 26-47)
- `metal/kernels/ternary_8x8.metal` (lines 22-26, 41-45)

**Files using Encoding B** (6 files):
- `neon/neon_ternary.c` (lines 13-14, 49-54, 62-73)
- `sme/ternary_sme.h` (lines 12-16)
- `sme/ternary_sme.c` (lines 90-98)
- `sme/sme_kernels.s` (lines 23-24, 50-51)
- `sme/sme_matmul_kernel.s` (lines 19, 105-108)
- `sme/sme_fused_linear.c` (lines 22-28)

Additionally, `sme/ternary_sme.h` line 12 claims `"Weight encoding (same as Metal)"` but then specifies Encoding B, which is the opposite of Metal. This comment is actively misleading.

**Consequence:** Weights packed by any Encoding A function and consumed by any Encoding B function (or vice versa) silently drop all negative weights. Every `-1` becomes `0`. The computation is silently, catastrophically wrong with no error or warning.

### Resolution: Adopt Encoding B as Canonical

**Rationale:** Encoding B (`10` = -1) is the natural binary representation. The value `2` in unsigned binary maps to `-1` in the ternary interpretation. This is the encoding used by the hardware backends (NEON, SME) where performance matters, and it maps cleanly to hardware operations:

- NEON TBL: index 2 -> -1 (natural LUT position)
- SME: `cmpeq ... #2` (compare to literal 2)
- SDOT/SMMLA: weights are already int8 `{0, +1, -1}` which pack as `{0, 1, 2}`

Encoding A's choice of `11` for -1 (both bits set = negative) has mnemonic appeal but wastes the `10` codepoint and requires an extra bit-manipulation step to decode on hardware.

### Deliverables

#### Phase 2a: Define the Canonical Encoding

Create `include/trit_encoding.h` -- a single-source-of-truth header defining the encoding:

```c
/*
 * YINSEN CANONICAL TRIT ENCODING (2 bits per trit)
 *
 *   00 (0) = 0   (zero / skip)
 *   01 (1) = +1  (add)
 *   10 (2) = -1  (subtract)
 *   11 (3) = reserved (decoded as 0, must not be produced by encoders)
 *
 * This encoding is CANONICAL. All backends (CPU, Metal, NEON, SME)
 * MUST use this encoding. Packed weight buffers are portable across
 * all backends without conversion.
 */

#define TRIT_ZERO     0x0  /* 00 */
#define TRIT_POS      0x1  /* 01 */
#define TRIT_NEG      0x2  /* 10 */
#define TRIT_RESERVED 0x3  /* 11 - must not be produced */
```

#### Phase 2b: Migrate Encoding A Files

Update the three Encoding A files to use the new canonical encoding:

1. **`include/ternary.h`** -- Change `TRIT_NEG` from `0x3` to `0x2`. Update `trit_unpack` and `trit_encode`. Update the encoding comment block.

2. **`metal/kernels/ternary_core.metal`** -- Change the `trit_sign` function. The branchless formula `lsb * (1 - 2 * msb)` must be replaced. New formula: `sign = (encoding == 1) ? 1 : (encoding == 2) ? -1 : 0` or a branchless equivalent using bit operations. Update the encoding comment block.

3. **`metal/kernels/ternary_8x8.metal`** -- Same changes as `ternary_core.metal`.

#### Phase 2c: Fix the Misleading SME Comment

Update `sme/ternary_sme.h` line 12 from:
```c
 * Weight encoding (same as Metal):
```
to:
```c
 * Weight encoding (canonical - see include/trit_encoding.h):
```

#### Phase 2d: Cross-Backend Verification Test

Create `test/test_encoding_canonical.c` that:

1. Packs a known weight vector using `ternary.h`'s `trit_encode`/`trit_pack4`
2. Unpacks using `ternary.h`'s `trit_unpack`
3. Verifies roundtrip correctness for all trit values {-1, 0, +1}
4. Verifies that the packed bytes match the expected bit patterns under the canonical encoding
5. Verifies that the NEON decode table `{0, 1, -1, 0}` correctly decodes every 2-bit index under the canonical encoding
6. Tests all 256 possible packed bytes (exhaustive for 4-trit packing)

Add a Makefile target: `make test-encoding`.

#### Phase 2e: Update All Existing Tests

Re-run `make test`, `make falsify`, `make prove4x4`, NEON tests (`neon/make test`), SME tests (`sme/make test`), and Metal tests (`swift run yinsen-metal-tests` in `metal/`). Every test must pass with the unified encoding.

### Acceptance Criteria

- [x] `include/trit_encoding.h` exists and defines the canonical encoding
- [x] `grep -r "0x3" include/ternary.h` does not match `TRIT_NEG`
- [x] All 9 files across all backends reference or match the same encoding
- [x] `test_encoding_canonical.c` passes with all 256 byte patterns
- [x] All existing test suites pass (`make test`, `make falsify`, `make prove4x4`, NEON, SME, Metal)
- [x] The misleading "same as Metal" comment in `sme/ternary_sme.h` is corrected
- [x] A single packed weight buffer produces correct results on CPU, NEON, SME, and Metal

> **Status: COMPLETE** (2026-01-31). Canonical encoding `trit_encoding.h` created. ternary.h, both Metal shaders, and SME comment migrated. 24 encoding tests pass.

---

## Problem 3: Metal Kernels Are Not Compute Kernels

### Diagnosis

Every ternary Metal kernel in the project uses one thread per output row with no threadgroup cooperation, no shared memory, no simdgroup operations, and no K-dimension parallelism. For a 4096x4096 matvec, each thread serially walks 4096 elements. This is orders of magnitude slower than the NEON kernels or a properly tiled GPU implementation.

Current state across all `.metal` files:

| Kernel | Threadgroup Shared Memory | Simdgroup Ops | K-Reduction Parallelism |
|--------|:---:|:---:|:---:|
| `ternary_matvec` | No | No | No |
| `ternary_matvec_bias` | No | No | No |
| `ternary_matvec_8x8` | No | No | No |
| `ternary_matvec_8x8_single` | No | No | No |
| `ternary_matvec_tiled_8x8` | No | No | No |
| `ternary_matvec_8x8_batched` | No | No | No |
| `ternary_matvec_8x8_f16` | No | No | No |

Meanwhile, the *non-ternary* kernels (`layernorm`, `softmax_rows`, `rmsnorm`) already use threadgroup shared memory and parallel reductions correctly. The pattern exists in the codebase; it just hasn't been applied to the ternary kernels.

Also: `YinsenMetal.swift` embeds a hardcoded subset of kernel source as a string literal (line 39), which is out of sync with the actual `.metal` files. The Package.swift copies `.metal` files as resources, but the Swift API doesn't load them.

Also: the function `ternary_dot8_simd` in `ternary_8x8.metal` is misleadingly named. It uses no Metal simdgroup intrinsics; it manually constructs `float4` vectors.

### Resolution: Build One Real Kernel, Retire the Rest

The goal is not to build an entire GPU inference stack. It is to have one ternary matvec kernel that is both **proven correct** and **actually fast**, demonstrating that verified ternary computation works on GPU at real dimensions.

### Deliverables

#### Phase 3a: Implement `ternary_matvec_tiled.metal`

A single new kernel file implementing a threadgroup-cooperative ternary matvec:

**Architecture:**
- One threadgroup per output row (or small tile of rows)
- 256 threads per threadgroup, each handling `ceil(K / 256)` elements
- Wider packing: `uint32_t` (16 trits per word) for fewer memory transactions
- Two-level reduction: `simd_sum` within each 32-thread simdgroup, then shared-memory reduction across simdgroups
- Input vector loaded once into threadgroup shared memory, reused by all threads

**Variants (in order of priority):**
1. `ternary_matvec_tiled` -- float32 activations, threadgroup K-reduction
2. `ternary_matvec_tiled_f16` -- float16 activations (2x bandwidth, 2x ALU on Apple Silicon)

**Encoding:** Must use the canonical encoding from Phase 2.

**Verification:**
- The existing exhaustive 4x4 test must pass against the new kernel
- A new random-input fuzz test (10,000 iterations, random M/N/weights/inputs) must match the CPU reference to within float epsilon
- The new kernel must be tested at real dimensions: 512x512, 1024x1024, 4096x4096

#### Phase 3b: Benchmark and Record Results

Create `metal/test/BenchmarkTiled.swift`:
- Measure throughput in GOP/s at 512x512, 1024x1024, 4096x4096
- Measure latency in microseconds per operation
- Compare against the old one-thread-per-row kernel
- **Persist results** to `metal/benchmark_results.md` with date, hardware, and kernel version

This is the first time benchmark results will be recorded in this project.

#### Phase 3c: Fix the Swift API

Update `YinsenMetal.swift`:
- Load `.metal` files from the bundle resources (which Package.swift already copies) instead of embedding a hardcoded string
- Expose the new tiled kernel through the API
- Remove the stale embedded kernel source

#### Phase 3d: Rename `ternary_dot8_simd`

Rename to `ternary_dot8_vectorized` (or similar) to not claim simdgroup behavior it doesn't have. This is a small change but matters for trust in the codebase.

#### Phase 3e: Retire Toy Kernels

Move `ternary_matvec_8x8_single` (single-thread computes all 8 rows) to a `metal/kernels/archive/` directory or delete it. It exists only as a curiosity and adds confusion. Keep:
- `ternary_core.metal` -- as the proven-correct reference
- `ternary_8x8.metal` -- only the verification kernel and the f16 variant (as forward-looking references)
- `ternary_matvec_tiled.metal` -- the new real kernel
- `activations.metal`, `layernorm.metal` -- already reasonable

### Acceptance Criteria

- [x] `ternary_matvec_tiled.metal` exists and uses threadgroup shared memory + simd_sum
- [x] Exhaustive 4x4 test passes against the new kernel
- [x] 10,000-iteration fuzz test passes at multiple dimensions
- [x] Benchmark results are recorded in `metal/benchmark_results.md`
- [x] The new kernel achieves measurable speedup over the one-thread-per-row baseline at 1024x1024 (1.71x)
- [x] `YinsenMetal.swift` loads shaders from bundle resources, not embedded strings
- [x] `ternary_dot8_simd` is renamed to `ternary_dot8_vectorized`
- [x] No function in the codebase claims "simd" without using Metal simdgroup intrinsics

> **Status: COMPLETE** (2026-01-31). New threadgroup-cooperative kernel with shared memory + simd_sum. 1.71x speedup at 1024x1024. Encoding bug in Swift layer fixed (was using old 11=-1 encoding). YinsenMetal.swift rewritten to load .metal from filesystem/bundle. All tests pass (4x4 exhaustive, 8x8 boundary+random+linearity, tiled fuzz).

---

## Problem 4: Undefined Behavior in CfC Cell

### Diagnosis

Both `cfc_ternary.h` and `cfc.h` contain the same bug in the per-element tau validation path.

**Location:** `cfc_ternary.h` lines 109-140, `cfc.h` lines 111-141.

**The bug:**

When `tau_shared == false` (per-element tau) and `params->tau[i] <= 0.0f` for some index `i`:

1. Line 109/111: `float decay[hid_dim];` is declared as an uninitialized VLA on the stack.
2. Lines 126-128 / 128-130: The code sets `h_new[i] = NAN` and executes `continue`, skipping `decay[i] = expf(...)`. The `decay[i]` slot remains uninitialized.
3. Lines 137 / 138: The Step 5 loop unconditionally reads `decay[i]` for all `i`:
   ```c
   float retention = (1.0f - gate[i]) * h_prev[i] * decay[i];
   ```
   This reads uninitialized memory (undefined behavior). The result overwrites the NAN that was written to `h_new[i]`, destroying the intended error signal.

**Contrast with the shared-tau path** (lines 112-118 / 114-120): When `tau_shared == true` and `tau[0] <= 0`, the function writes NAN to all `h_new[i]` and **returns early**. This path is correct. The per-element path lacks the early return.

**Additional issues in the same files:**

1. **`W_out` dimension comment** (`cfc_ternary.h` line 54): Comment says `[hidden_dim, output_dim]` but the indexing at lines 152-159 iterates `output_dim` rows of `hidden_dim` columns. The correct shape is `[output_dim, hidden_dim]`.

2. **VLA stack overflow risk:** All temporary arrays are VLAs on the stack. For `hidden_dim=4096`, `yinsen_cfc_ternary_cell` allocates ~96KB on the stack per call. No bounds checking exists.

### Resolution

#### Phase 4a: Fix the Uninitialized Read

In both `cfc_ternary.h` and `cfc.h`, change the per-element tau loop to skip the corresponding element in Step 5, mirroring the NAN-and-continue pattern but preventing the overwrite.

**Fix approach:** In the Step 5 loop, check if `decay[i]` was set before reading it. The cleanest way is to initialize `decay` to NAN at declaration, then let Step 5's math propagate the NAN naturally:

```c
float decay[hid_dim];
/* Initialize to NAN so invalid tau propagates cleanly */
for (int i = 0; i < hid_dim; i++) decay[i] = NAN;
```

This eliminates the UB (no uninitialized read) and preserves the NAN error signal (NAN * anything = NAN, so `h_new[i]` will be NAN for invalid tau elements). It costs one memset-equivalent loop, which is negligible relative to the expf() calls.

Alternatively, add a `continue` guard in Step 5:

```c
for (int i = 0; i < hid_dim; i++) {
    if (isnan(h_new[i])) continue;  /* tau was invalid, preserve NAN */
    float retention = (1.0f - gate[i]) * h_prev[i] * decay[i];
    float update = gate[i] * candidate[i];
    h_new[i] = retention + update;
}
```

**Recommendation:** Use the initialization approach. It is simpler, eliminates UB by construction, and does not require a branch in the hot path.

#### Phase 4b: Fix the `W_out` Comment

In `cfc_ternary.h` line 54, change:
```c
const uint8_t* W_out;   /* Packed ternary [hidden_dim, output_dim] */
```
to:
```c
const uint8_t* W_out;   /* Packed ternary [output_dim, hidden_dim] */
```

Check `cfc.h` for the same issue and fix if present.

#### Phase 4c: Add Tests for the Fixed Paths

Create tests in `test/test_cfc_ternary.c` and `test/test_cfc.c`:

1. **Per-element tau with invalid values:** Set `tau_shared = 0`, provide a tau array where some elements are `<= 0` and others are valid. Verify that `h_new[i]` is NAN for invalid tau indices and finite for valid ones.

2. **All-invalid tau:** Set all tau elements to 0 or negative. Verify all `h_new[i]` are NAN.

3. **Per-element tau happy path:** Set `tau_shared = 0`, all tau values valid. Verify output matches the shared-tau path with the same tau value.

#### Phase 4d: Document VLA Stack Budget

Add a comment block at the top of `yinsen_cfc_ternary_cell` and `yinsen_cfc_cell` documenting the stack requirement:

```c
/*
 * Stack usage: (input_dim + 6 * hidden_dim) * sizeof(float) bytes.
 * For hidden_dim=256:  ~6.5 KB
 * For hidden_dim=1024: ~25 KB
 * For hidden_dim=4096: ~96 KB
 *
 * Callers on stack-constrained platforms (embedded, threads with small
 * stacks) must ensure sufficient stack space.
 */
```

This does not fix the VLA issue (which would require a larger refactor to use caller-provided scratch buffers) but makes it visible.

### Acceptance Criteria

- [x] `decay[]` is initialized before any possible read in both `cfc_ternary.h` and `cfc.h`
- [x] No path through either function reads an uninitialized VLA element
- [x] `W_out` comment says `[output_dim, hidden_dim]`
- [x] Tests exist for per-element tau with invalid values (>= 3 test cases)
- [x] Tests exist for per-element tau with all-valid values
- [x] Stack usage is documented in a comment block
- [x] `make test` passes with all new tests

> **Status: COMPLETE** (2026-01-31). VLA initialized to NAN, W_out comment fixed in both headers, 8 new tau edge-case tests added, stack budget documented.

---

## Phasing and Dependencies

```
Phase 1: Identity Consolidation
  (no code dependencies, can start immediately)
  |
Phase 2: Encoding Unification --------+
  (changes ternary.h, Metal shaders)  |
  |                                    |
Phase 3: Metal Kernel ----------------+
  (depends on canonical encoding)     |
                                       |
Phase 4: Bug Fixes --------------------
  (independent of encoding/Metal, can run in parallel with Phase 2)
```

**Phase 1** and **Phase 4** have no code dependencies and can execute in parallel.
**Phase 2** must complete before **Phase 3** (the new Metal kernel must use the canonical encoding).
**Phase 4** is independent of Phases 2 and 3.

### Estimated Effort

| Phase | Scope | Estimate |
|-------|-------|----------|
| 1: Identity | Doc updates, README rewrite | 1-2 hours |
| 2: Encoding | 9 files changed, 1 new header, 1 new test | 1 day |
| 3: Metal | 1 new kernel, benchmarks, Swift API fix | 2-3 days |
| 4: Bug fixes | 2 headers, 2 test files, comments | 2-3 hours |

**Total: ~4-5 days of focused work.**

---

## What This PRD Does Not Cover

- **Training.** The question "can ternary networks learn?" is critical but orthogonal. It should be explored experimentally (MLX, PyTorch, manual construction) and is not a product deliverable.
- **NEON kernel optimization.** The NEON kernels are already the most mature backend. Further optimization (threading, fusion) is valuable future work but not a structural problem.
- **SME enablement.** macOS 15 does not expose SME to userspace. The SME kernels have scalar fallbacks. This is an Apple platform constraint, not a Yinsen problem.
- **New primitives or operations.** Attention, RoPE, KV-cache, embedding layers are all needed eventually but depend on having a model to run. Build them when there is a model.
- **Test hardening.** The falsification suites have structural softness (unconditional passes, tests that cannot fail). This matters but is a separate effort.

---

## Changelog

- 2026-01-31: Initial PRD created, addressing concerns #2, #3, #5, #6 from codebase review
