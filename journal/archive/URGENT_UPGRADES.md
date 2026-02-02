# URGENT UPGRADES: Metal GPU is 6x Behind NEON CPU

**Date:** 2026-01-31
**Status:** Open -- no work lost, but Metal needs the same tricks NEON already has

---

## The Gap

| Backend | Best Kernel | Matrix | GOP/s | GB/s | Weight Format |
|---------|-------------|--------|------:|-----:|---------------|
| **NEON CPU** | `neon_block16_matvec` | 4096x4096 | **186** | **93** | Int8 direct (1 byte/weight) |
| **NEON CPU** | `neon_int8_matvec_blocked8_k64` | 4096x4096 | **186** | **93** | Int8 direct (B8-K64 layout) |
| **NEON CPU** | `neon_ternary_matvec_blocked8` | 4096x4096 | ~136 | ~34 | 2-bit packed (TBL decode) |
| Metal GPU | `ternary_matvec_tiled` (cooperative) | 4096x4096 | 27 | -- | 2-bit packed |
| Metal GPU | `ternary_matvec_tiled_8x8` | 4096x4096 | **31** | -- | 2-bit packed |
| Metal GPU | `ternary_matvec` (4x4) | 4096x4096 | 16 | -- | 2-bit packed |

**Metal's best (31 Gop/s) is 6x slower than NEON's best (186 Gop/s).**

This is not a regression -- we never had fast Metal kernels. The 93 GB/s and 186 GOP/s numbers were always from NEON CPU (`neon/neon_ternary.c`), recorded in commits `a9f2642` and `a213e24`. The Metal work (Phase 3) focused on correctness and cooperative tiling, not throughput parity.

---

## Why NEON Wins

The NEON kernels discovered two key optimizations that Metal hasn't adopted yet:

### 1. Int8 Direct Weight Format (the big one)

The 2-bit packed format requires a decode step (TBL lookup on NEON, bit extraction on Metal) before SDOT can fire. Eliminating this by storing weights as full int8 bytes (`-1, 0, +1`) costs 4x memory but:

- **Removes the decode bottleneck entirely**
- Allows SDOT/SMMLA to fire on raw loads with zero preprocessing
- On NEON: 136 GOP/s (2-bit TBL) -> 186 GOP/s (Int8 direct) = **1.37x speedup**

Memory cost at 4096x4096: 4 MB (2-bit) -> 16 MB (Int8). Fits in M4 unified memory either way.

### 2. Cache-Friendly Blocked Layouts

Row-major Int8 gets 127 GOP/s. Blocked-8-K64 layout (8 output channels x 64 K-elements per block) gets 186 GOP/s. The blocking:

- Maximizes register utilization (8 accumulators active)
- Unrolls K by 64, reducing loop overhead
- Aligns loads to cache line boundaries
- Enables prefetch-friendly sequential access patterns

### NEON Kernel Progression (for reference)

All at 4096x4096 on Apple M4:

| Kernel | Format | GOP/s | Commit |
|--------|--------|------:|--------|
| Reference (scalar) | 2-bit | ~10 | `ed2e42e` |
| SDOT 1OC | 2-bit TBL | ~92 | `ed2e42e` |
| SDOT 4OC | 2-bit TBL | ~125 | `ed2e42e` |
| SDOT 8OC | 2-bit TBL | ~135 | `ed2e42e` |
| Blocked-8 | 2-bit TBL | ~136 | `41d78df` |
| Int8 Row-Major 8OC | Int8 direct | 127 | `a9f2642` |
| Int8 Blocked-8 K16 | Int8 direct | 166 | `a9f2642` |
| Int8 Blocked-8 K32 | Int8 direct | 174 | `a9f2642` |
| **Int8 Blocked-8 K64** | **Int8 direct** | **186** | `a9f2642` |
| Ghost-12 LDNP (asm) | Int8 direct | slower | `a213e24` |
| Ghost-12 LDR | Int8 direct | 168 | `a213e24` |
| **Block-16** | **Int8 direct** | **186** | `a213e24` |

Key lesson: **LDNP (non-temporal loads) was 26% SLOWER than standard LDR on M4.** Apple's prefetchers are smarter than explicit cache hints. Don't fight the hardware.

---

## What Metal Needs

### Upgrade 1: Int8 Direct Metal Kernel (HIGH PRIORITY)

Write a Metal compute kernel that takes Int8 weights (`-1, 0, +1` as signed bytes) instead of 2-bit packed trits. This eliminates the per-element bit-extraction and decode that currently dominates Metal compute time.

**Files to create:**
- `metal/kernels/ternary_matvec_int8.metal` -- Int8 direct kernel
- Corresponding Swift API in `YinsenMetal.swift`
- Test coverage in `metal/test/`

**Key design:**
- Each thread reads `int8_t` weights directly, no decode
- Use Metal's `simd_sum` for reduction (already proven in cooperative tiled kernel)
- Threadgroup shared memory for input vector (already proven)
- The weight matrix is 4x larger but the ALU is freed from decode overhead

**Expected improvement:** If the same 1.37x factor applies, Metal should go from 31 -> ~42 Gop/s. If the decode was a larger fraction of Metal's bottleneck than NEON's, the improvement could be bigger.

### Upgrade 2: Blocked Weight Layout for Metal (MEDIUM PRIORITY)

The NEON kernels showed that K-unrolling and N-blocking matter enormously (127 -> 186 GOP/s on NEON, 1.46x). Metal should similarly:

- Process multiple output rows per threadgroup (N-blocking)
- Unroll the K-dimension loop to reduce shared memory synchronization points
- Align weight blocks to Metal's preferred access granularity

### Upgrade 3: Metal Bandwidth Measurement (HIGH PRIORITY)

The current Metal benchmarks don't report GB/s. We don't know if Metal is compute-bound or memory-bound. Add bandwidth reporting to `Benchmark4x4vs8x8.swift`:

```swift
let weightBytes = Double(M * N)  // or M * N / 4 for 2-bit
let inputBytes = Double(N * 4)   // float input
let outputBytes = Double(M * 4)  // float output
let totalBytes = weightBytes + inputBytes + outputBytes
let bwGBs = totalBytes / (avgLatencySeconds * 1e9)
```

M4 theoretical memory bandwidth is ~120 GB/s. If Metal is already at 90%+ utilization, the problem isn't the kernel -- it's the format. If Metal is at 20% utilization, there's a dispatch/occupancy problem.

### Upgrade 4: I8MM Exploration on Metal (LOW PRIORITY)

NEON has I8MM (SMMLA) kernels for batch>1 workloads. Metal's GPU has its own matrix multiply capabilities. Worth investigating but not urgent since CfC inference is batch=1.

---

## What We Did NOT Lose

The NEON kernels are intact and untouched:

```bash
cd neon && make test   # Builds and runs all NEON benchmarks
```

Source: `neon/neon_ternary.c` (~1500 lines, 15+ kernel variants)
Test: `neon/test_neon.c` (829 lines, correctness + 10K-iteration benchmarks)

All kernel variants verified correct against scalar reference at N=4096, K=4096.

---

## Falsified Ideas (Don't Repeat These)

| Idea | Result | Source |
|------|--------|--------|
| LDNP non-temporal loads | **26% slower** than LDR on M4 | `a213e24` |
| Ghost-Stream cache bypass | Apple prefetchers already optimal | `a213e24` |
| Deeper K-unrolling past K=64 | Diminishing returns (K32->K64 was only 1.07x) | `a9f2642` |
| I8MM for batch=1 matvec | SMMLA needs 2-row batching, no benefit for batch=1 | `1c4e2b4` |

---

## Priority Order

1. **Measure Metal bandwidth utilization** -- know if we're compute-bound or memory-bound before optimizing
2. **Int8 direct Metal kernel** -- proven 1.37x on NEON, likely more on Metal where decode is costlier
3. **Blocked layout for Metal** -- proven 1.46x on NEON (row-major -> blocked-8-K64)
4. **Re-run NEON benchmarks** to confirm 186 GOP/s still holds (haven't re-verified since the NEON commits)

---

## 7B Model Inference Estimates

Based on ~6.5B ternary ops per token:

| Backend | Best GOP/s | Est. tok/s |
|---------|----------:|----------:|
| NEON CPU (current) | 186 | ~28.5 |
| Metal GPU (current) | 31 | ~4.8 |
| Metal GPU (with Int8 + blocking, estimated) | ~60-80 | ~9-12 |
| NEON + Metal combined (speculative) | ~200+ | ~30+ |

The NEON CPU path is currently the production-viable path for M4 inference. Metal is research/correctness only until these upgrades land.
