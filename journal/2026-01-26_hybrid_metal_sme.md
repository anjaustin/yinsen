# Hybrid Metal + SME Architecture

**Date:** 2026-01-26
**Status:** DEFERRED (Phase 3)

## Context

We have two compute paths for ternary inference on Apple Silicon:

1. **Metal 8×8** — GPU-based, portable, works on all Apple Silicon
2. **SME 16×16** — CPU-based, M4-specific, 4× theoretical speedup

## The Insight

Different workloads favor different paths:

| Workload | Characteristic | Best Path |
|----------|----------------|-----------|
| Single token generation | Latency-bound | Metal GPU |
| Batch embedding | Throughput-bound | SME CPU |
| Prompt prefill | Large matrix, one-shot | SME CPU |
| Autoregressive decode | Small matrix, sequential | Metal GPU |

## Proposed Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Yinsen Inference API                     │
│                                                             │
│  yinsen_forward(model, tokens, batch_size) → logits        │
└─────────────────────────────────────────────────────────────┘
                           │
                           ▼
              ┌────────────────────────┐
              │   Runtime Dispatcher   │
              │                        │
              │  if (batch >= 16 &&    │
              │      has_sme())        │
              │    → SME path          │
              │  else                  │
              │    → Metal path        │
              └────────────────────────┘
                    │           │
         ┌──────────┘           └──────────┐
         ▼                                 ▼
┌─────────────────┐               ┌─────────────────┐
│   SME 16×16     │               │   Metal 8×8     │
│                 │               │                 │
│  Native ARM64   │               │  GPU Compute    │
│  ZA tile ops    │               │  Shader         │
│  Predicate mask │               │  SIMD groups    │
│                 │               │                 │
│  Batch ≥ 16     │               │  Any batch      │
│  M4+ only       │               │  All Apple GPU  │
└─────────────────┘               └─────────────────┘
```

## Weight Storage

Both paths need different memory layouts:

**Metal 8×8:**
- Row-major packed
- 2 bytes per 8 weights (uint16_t per row)
- Standard layout

**SME 16×16:**
- Column-interleaved for `ld1rw` broadcast
- 4 bytes per 16 weights (uint32_t per column slice)
- Requires weight transformation at load time (or dual storage)

### Option A: Dual Storage
Store weights in both formats. 2× memory for weights, but zero runtime conversion.

### Option B: Runtime Transformation  
Store in one format, transform on load. Memory efficient, but startup cost.

### Option C: Unified Layout
Design a layout that works for both with minimal overhead. May not be possible given the different access patterns.

**Recommendation:** Option A for v1. Weights are already tiny (ternary), doubling them is acceptable.

## Detection

```c
#include <sys/sysctl.h>

bool has_sme(void) {
    // Check for SME support via sysctl or feature detection
    int64_t sme = 0;
    size_t size = sizeof(sme);
    
    // Apple doesn't document this, but SME presence can be inferred
    // from CPU model or feature flags
    if (sysctlbyname("hw.optional.arm.FEAT_SME", &sme, &size, NULL, 0) == 0) {
        return sme != 0;
    }
    
    // Fallback: check CPU model
    char model[256];
    size = sizeof(model);
    sysctlbyname("machdep.cpu.brand_string", model, &size, NULL, 0);
    
    // M4 and later have SME
    return strstr(model, "M4") != NULL;
}
```

## Integration Points

### 1. Model Loading
```c
YinsenModel* yinsen_load(const char* path) {
    // Load weights
    // If SME available, also prepare SME-formatted weights
    if (has_sme()) {
        model->weights_sme = transform_to_sme_layout(model->weights);
    }
}
```

### 2. Forward Pass
```c
void yinsen_forward(YinsenModel* model, int* tokens, int n, float* logits) {
    if (n >= 16 && model->weights_sme) {
        sme_forward(model, tokens, n, logits);
    } else {
        metal_forward(model, tokens, n, logits);
    }
}
```

### 3. Attention
Attention is the most complex case. Q/K/V projections benefit from SME batching, but the attention computation itself may be better on GPU.

**Possible split:**
- Prefill: SME for Q/K/V projection, Metal for attention scores
- Decode: Metal for everything (single token)

## Performance Targets

| Workload | Metal 8×8 | SME 16×16 | Hybrid |
|----------|-----------|-----------|--------|
| Prefill 2K context | 50ms | 15ms | 15ms |
| Single token | 5ms | 8ms | 5ms |
| Batch 16 tokens | 20ms | 6ms | 6ms |

(Targets are speculative, need benchmarking)

## Open Questions

1. **Memory pressure:** Does maintaining dual weight formats cause cache thrashing?
2. **Transition overhead:** What's the cost of switching between Metal and SME mid-inference?
3. **Batching in chat:** Can we accumulate tokens and process in batches during multi-turn conversation?
4. **KV cache:** Where does KV cache live? Unified memory helps, but access patterns matter.

## Implementation Order

1. ✅ Metal 8×8 (done)
2. 🔄 SME 16×16 (in progress)
3. ⏳ Hybrid dispatcher (this document)
4. ⏳ Benchmark both paths
5. ⏳ Tune dispatch threshold

## References

- ARM SME Programmer's Guide
- Apple M4 microarchitecture (reverse-engineered)
- Your associate's assembly listing (see commit for SME kernel)

---

*This is a Phase 3 item. Complete SME 16×16 and Metal optimizations first.*
