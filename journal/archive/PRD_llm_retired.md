# Yinsen LMM: Product Requirements Document

**Version:** 1.0
**Date:** 2026-01-26
**Status:** RETIRED (2026-01-31)

> **Retirement note:** This PRD was retired during identity consolidation.
> The LLM ambition outpaced the project's proven foundation (verified ternary
> primitives and deterministic computation). Replaced by PRD_consolidation.md,
> which focuses on resolving structural issues: encoding inconsistency, Metal
> kernel maturity, CfC bugs, and project identity. Archived for provenance.

## Executive Summary

Build and deploy a ternary large language model (LMM) on Apple Silicon using proven Metal compute kernels. Train from scratch, deploy everywhere: CLI, API server, embedded.

## Problem Statement

Current LLM inference is:
- **Memory-bound:** Moving 32-bit weights saturates bandwidth
- **Hardware-locked:** CUDA/NVIDIA dependency
- **Unverifiable:** Complex kernels with no correctness proofs
- **Bloated:** Gigabytes of weights for simple tasks

## Solution

A vertically-integrated stack:
1. **Proven Metal kernels** - Exhaustively verified ternary operations
2. **Training pipeline** - MLX on Apple Silicon, ternary-aware
3. **Frozen model export** - Packed ternary weights, auditable
4. **Multi-target runtime** - CLI, HTTP API, embedded binary

## Goals

| Goal | Metric | Target |
|------|--------|--------|
| Correctness | Kernel test coverage | 100% (exhaustive proof) |
| Size | Model weights | <100MB for base model |
| Speed | Tokens/second on M4 | >100 tok/s |
| Portability | Build targets | macOS, iOS, embedded C |
| Auditability | Lines of kernel code | <1000 |

## Non-Goals

- CUDA/NVIDIA support (future work)
- Training infrastructure (use existing MLX)
- Beating GPT-4 on benchmarks (prove the architecture works first)
- Browser/WASM deployment (future work)

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      Applications                           │
├─────────────┬─────────────────────┬─────────────────────────┤
│  yinsen-cli │   yinsen-server     │    yinsen-embed         │
│  (terminal) │   (HTTP API)        │    (static binary)      │
├─────────────┴─────────────────────┴─────────────────────────┤
│                    Inference Runtime                        │
│  - Token sampling                                           │
│  - KV cache management                                      │
│  - Streaming output                                         │
├─────────────────────────────────────────────────────────────┤
│                    Yinsen Metal Kernels                     │
│  - ternary_matvec (PROVEN)                                  │
│  - ternary_attention                                        │
│  - layernorm, softmax, embeddings                           │
├─────────────────────────────────────────────────────────────┤
│                    Metal Performance Shaders                │
│                    (M4 GPU - Unified Memory)                │
└─────────────────────────────────────────────────────────────┘
```

---

## Phase 1: Metal Kernels (PROVEN)

**Objective:** Port Yinsen ternary operations to Metal, prove correctness exhaustively.

### Deliverables

| File | Description | Verification |
|------|-------------|--------------|
| `metal/kernels/ternary_dot.metal` | 4-element ternary dot product | 81 cases (3^4) |
| `metal/kernels/ternary_matvec.metal` | Matrix-vector multiply | 43,046,721 cases (3^16 for 4x4) |
| `metal/kernels/activations.metal` | ReLU, GELU, SiLU, Softmax | Property tests |
| `metal/kernels/layernorm.metal` | Layer normalization | Property tests |
| `metal/kernels/attention.metal` | Ternary Q/K/V attention | Spot checks + fuzz |
| `metal/test/test_exhaustive.swift` | GPU verification harness | CI integration |
| `metal/YinsenMetal.swift` | Swift API bindings | Unit tests |

### Acceptance Criteria

- [ ] All 43,046,721 4x4 weight configurations pass on GPU
- [ ] Results match CPU reference (`include/ternary.h`) to float epsilon
- [ ] Kernel source is <500 lines total
- [ ] No external dependencies beyond Metal stdlib

### Technical Specification

**Trit Encoding (matches `ternary.h`):**
```
00 = 0  (zero - skip)
01 = +1 (add)
11 = -1 (subtract)
10 = reserved (treated as 0)
```

**Memory Layout:**
- Weights: packed uint8, 4 trits per byte, row-major
- Activations: float32 (int8 quantization in Phase 2)
- Output: float32

**Thread Model:**
- One thread per output row (simple, correct first)
- Optimize for occupancy after correctness proven

---

## Phase 2: Training Pipeline

**Objective:** Train a ternary language model from scratch using MLX on M4.

### Deliverables

| File | Description |
|------|-------------|
| `train/model.py` | Model architecture (Transformer or CfC) |
| `train/ternary_ops.py` | Straight-Through Estimator for ternary |
| `train/data.py` | Data loading and tokenization |
| `train/train.py` | Training loop with checkpointing |
| `train/export.py` | Export to Yinsen packed format |

### Model Architecture

**Option A: Ternary Transformer (Recommended for v1)**
```
Embedding: float32 (small, keep precision)
N x TransformerBlock:
  - LayerNorm (float32)
  - Attention:
    - W_Q, W_K, W_V, W_O: ternary
  - LayerNorm (float32)
  - FFN:
    - W_up, W_down: ternary
LM Head: ternary
```

**Hyperparameters (v1 baseline):**
```
d_model: 512
n_heads: 8
n_layers: 6
d_ff: 2048
vocab_size: 32000 (sentencepiece)
context_length: 2048
```

**Estimated size:**
- Ternary weights: ~25M parameters × 2 bits = ~6MB
- Embeddings: 32K × 512 × 4 bytes = ~64MB
- Total: ~70MB

### Training Strategy

1. **Phase 2a:** Train with float32, standard cross-entropy
2. **Phase 2b:** Fine-tune with STE quantization to ternary
3. **Phase 2c:** Validate quantized model matches training loss

### Acceptance Criteria

- [ ] Model trains without NaN/Inf
- [ ] Validation perplexity < 50 on target corpus
- [ ] Exported weights load correctly in inference runtime
- [ ] Quantization error < 5% accuracy degradation

---

## Phase 3: Inference Runtime

**Objective:** Build inference engine that runs trained model on Metal kernels.

### Deliverables

| File | Description |
|------|-------------|
| `runtime/inference.c` | Core inference loop (C, portable) |
| `runtime/metal_backend.m` | Metal kernel dispatch (Objective-C) |
| `runtime/sampling.c` | Token sampling (greedy, top-k, top-p) |
| `runtime/kv_cache.c` | KV cache for autoregressive generation |
| `runtime/tokenizer.c` | Sentencepiece/BPE tokenizer |

### API Design

```c
// Core inference API
typedef struct YinsenModel YinsenModel;

YinsenModel* yinsen_load(const char* path);
void yinsen_free(YinsenModel* model);

// Single forward pass
void yinsen_forward(
    YinsenModel* model,
    const int* tokens,
    int n_tokens,
    float* logits  // [vocab_size]
);

// Autoregressive generation
int yinsen_generate(
    YinsenModel* model,
    const int* prompt,
    int prompt_len,
    int* output,
    int max_tokens,
    float temperature,
    float top_p
);

// Streaming generation
typedef void (*YinsenCallback)(int token, void* ctx);
void yinsen_generate_stream(
    YinsenModel* model,
    const int* prompt,
    int prompt_len,
    int max_tokens,
    float temperature,
    float top_p,
    YinsenCallback callback,
    void* ctx
);
```

### Acceptance Criteria

- [ ] Generates coherent text from trained model
- [ ] >100 tokens/second on M4
- [ ] Memory usage stable (no leaks over 10K tokens)
- [ ] KV cache correctly handles context window

---

## Phase 4: CLI Application

**Objective:** Terminal interface for interactive generation.

### Deliverables

| File | Description |
|------|-------------|
| `apps/yinsen-cli/main.c` | CLI entry point |
| `apps/yinsen-cli/repl.c` | Interactive REPL |
| `apps/yinsen-cli/args.c` | Argument parsing |

### Usage

```bash
# One-shot generation
$ yinsen "Once upon a time"
Once upon a time, in a land far away, there lived a...

# Interactive mode
$ yinsen -i
yinsen> Hello, how are you?
I'm doing well, thank you for asking...
yinsen> /quit

# With options
$ yinsen --temperature 0.7 --max-tokens 100 "Explain quantum computing"
```

### Acceptance Criteria

- [ ] Builds as single static binary
- [ ] <10MB binary size
- [ ] <100ms startup time
- [ ] Streaming output (tokens appear as generated)

---

## Phase 5: API Server

**Objective:** HTTP server for programmatic access.

### Deliverables

| File | Description |
|------|-------------|
| `apps/yinsen-server/main.c` | Server entry point |
| `apps/yinsen-server/http.c` | Minimal HTTP parser |
| `apps/yinsen-server/routes.c` | API endpoints |
| `apps/yinsen-server/json.c` | JSON serialization |

### API Endpoints

```
POST /v1/completions
{
  "prompt": "Hello",
  "max_tokens": 100,
  "temperature": 0.7,
  "stream": false
}

Response:
{
  "text": "Hello! How can I help you today?",
  "tokens_generated": 8,
  "time_ms": 45
}

POST /v1/completions (streaming)
{
  "prompt": "Hello",
  "max_tokens": 100,
  "stream": true
}

Response: Server-Sent Events
data: {"token": "Hello"}
data: {"token": "!"}
data: {"token": " How"}
...
data: [DONE]
```

### Acceptance Criteria

- [ ] OpenAI-compatible API format
- [ ] Handles concurrent requests
- [ ] Streaming via SSE
- [ ] <5MB binary size

---

## Phase 6: Embedded Target

**Objective:** Minimal runtime for resource-constrained deployment.

### Deliverables

| File | Description |
|------|-------------|
| `apps/yinsen-embed/inference_minimal.c` | CPU-only inference |
| `apps/yinsen-embed/model_static.c` | Weights compiled into binary |

### Constraints

- No dynamic allocation after init
- No floating point (optional fixed-point mode)
- Single-file compilation possible
- Works on bare metal (no OS required)

### Acceptance Criteria

- [ ] Compiles with `-nostdlib` (freestanding)
- [ ] <1MB binary with small model
- [ ] Runs on Raspberry Pi Pico (stretch goal)

---

## Timeline

| Phase | Duration | Dependencies | Milestone |
|-------|----------|--------------|-----------|
| 1 | 1 week | None | Metal kernels proven |
| 2 | 2 weeks | Phase 1 | Model trained |
| 3 | 1 week | Phase 1, 2 | Inference works |
| 4 | 2 days | Phase 3 | CLI ships |
| 5 | 3 days | Phase 3 | API server ships |
| 6 | 1 week | Phase 3 | Embedded demo |

**Total: ~5 weeks to full stack**

---

## Risk Register

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Ternary training doesn't converge | Medium | High | Fall back to post-training quantization |
| Metal kernel slower than expected | Low | Medium | Profile early, optimize hot paths |
| M4 memory bandwidth limits throughput | Medium | Medium | Batch size tuning, kernel fusion |
| Model quality too low to be useful | Medium | High | Start with proven architecture (Transformer) |

---

## Success Criteria

**Phase 1 Complete:**
- Metal kernels pass exhaustive verification
- Performance within 2x of theoretical peak

**Phase 2 Complete:**
- Trained model generates coherent English text
- Perplexity competitive with similar-sized models

**Phase 3-6 Complete:**
- End-to-end demo: type prompt, get response
- All three deployment targets work
- Total codebase <10K lines

---

## Open Questions

1. **Architecture:** Transformer vs CfC vs Hybrid?
   - *Decision needed by Phase 2 start*

2. **Training data:** What corpus?
   - Options: OpenWebText, RedPajama, custom
   - *Decision needed by Phase 2 start*

3. **Tokenizer:** Train custom or use existing?
   - Recommendation: Use Llama tokenizer for compatibility
   - *Decision needed by Phase 2 start*

4. **Int8 activations:** Include in v1?
   - Recommendation: Float32 activations for v1, int8 in v2
   - *Reduces complexity*

---

## Appendix A: File Structure

```
yinsen/
├── include/                    # Existing CPU headers
│   ├── apu.h
│   ├── cfc.h
│   ├── cfc_ternary.h
│   ├── entromorph.h
│   ├── onnx_shapes.h
│   └── ternary.h
├── metal/                      # Phase 1
│   ├── kernels/
│   │   ├── ternary_dot.metal
│   │   ├── ternary_matvec.metal
│   │   ├── attention.metal
│   │   ├── layernorm.metal
│   │   └── activations.metal
│   ├── test/
│   │   ├── test_exhaustive.swift
│   │   └── test_benchmark.swift
│   ├── YinsenMetal.swift
│   └── Package.swift
├── train/                      # Phase 2
│   ├── model.py
│   ├── ternary_ops.py
│   ├── data.py
│   ├── train.py
│   ├── export.py
│   └── requirements.txt
├── runtime/                    # Phase 3
│   ├── inference.c
│   ├── inference.h
│   ├── metal_backend.m
│   ├── metal_backend.h
│   ├── sampling.c
│   ├── kv_cache.c
│   └── tokenizer.c
├── apps/                       # Phase 4-6
│   ├── yinsen-cli/
│   │   └── main.c
│   ├── yinsen-server/
│   │   ├── main.c
│   │   ├── http.c
│   │   └── routes.c
│   └── yinsen-embed/
│       └── inference_minimal.c
├── models/                     # Trained models
│   └── yinsen-1/
│       ├── config.json
│       ├── weights.bin
│       └── tokenizer.model
├── test/                       # Existing tests
├── docs/                       # Existing docs
├── PRD.md                      # This document
├── Makefile
└── README.md
```

---

## Appendix B: Verification Matrix

| Component | Verification Method | Coverage |
|-----------|---------------------|----------|
| ternary_dot4 | Exhaustive (81 cases) | 100% |
| ternary_matvec 4x4 | Exhaustive (43M cases) | 100% |
| ternary_matvec NxM | Property tests + fuzz | High |
| attention | Reference comparison | Medium |
| layernorm | Property tests | High |
| softmax | Property tests | High |
| training loop | Loss convergence | N/A |
| inference | Output coherence | Manual |
| CLI | Integration tests | High |
| API server | Integration tests | High |

---

## Changelog

- 2026-01-26: Initial PRD created
