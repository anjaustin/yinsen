# BitNet b1.58 vs Yinsen: Deltas and Parallels

## BitNet b1.58 Summary

BitNet b1.58 (Microsoft, Feb 2024) is a 1.58-bit LLM architecture where:
- All weights are ternary: **{-1, 0, +1}**
- Activations are 8-bit quantized
- Trained from scratch (not post-training quantization)
- Matches FP16 LLaMA at 3B+ parameters
- Uses LLaMA-compatible components (RMSNorm, SwiGLU, RoPE)

### Key Technical Details

**Quantization Function:**
```
W̃ = RoundClip(W / (γ + ε), -1, 1)
γ = mean(|W|)  # absmean scaling
```

**Results at Scale:**
- 3B BitNet b1.58: matches LLaMA 3B perplexity, 2.71x faster, 3.55x less memory
- 70B BitNet b1.58: 4.1x faster, 8.9x higher throughput
- 71.4x energy savings on matrix multiply (7nm)

---

## The Parallels (What We Share)

| Aspect | BitNet b1.58 | Yinsen |
|--------|--------------|--------|
| **Weight values** | {-1, 0, +1} | {-1, 0, +1} |
| **Bits per weight** | 1.58 (≈2) | 2 |
| **Core insight** | Ternary eliminates multiply | Ternary eliminates multiply |
| **Forward pass** | Addition/subtraction only | Addition/subtraction only |
| **Memory benefit** | ~10x vs FP16 | ~16x vs FP32, ~4x vs int8 |
| **Determinism goal** | Implied (integer ops) | Explicit (auditability) |

### The Fundamental Agreement

Both projects independently arrived at the same core insight:

> **Ternary weights transform matrix multiplication from `Σ(w*x)` to `Σ(x where w=+1) - Σ(x where w=-1)`**

This isn't coincidence. It's the natural endpoint of asking "how simple can neural computation get while remaining useful?"

---

## The Deltas (Where We Diverge)

### Delta 1: Scale

| | BitNet b1.58 | Yinsen |
|-|--------------|--------|
| Target size | 3B - 70B parameters | 50 - 10K parameters |
| Training data | 100B - 2T tokens | Synthetic / small datasets |
| Hardware | GPU clusters, custom chips | MCUs, bare metal |
| Use case | LLM inference | Edge time-series |

**Implication:** BitNet proves ternary works at scale. Yinsen explores ternary at the other extreme - where you can exhaustively verify.

### Delta 2: Architecture

| | BitNet b1.58 | Yinsen |
|-|--------------|--------|
| Base architecture | Transformer (attention) | CfC (recurrent) |
| Sequence handling | Self-attention | Hidden state + dt |
| Irregular time | Not addressed | Native (CfC handles dt) |
| Components | RMSNorm, SwiGLU, RoPE | Sigmoid, Tanh, simple gates |

**Implication:** BitNet is for language (transformers). Yinsen is for time series (recurrent). Different problem domains, same weight constraint.

### Delta 3: Training Method

| | BitNet b1.58 | Yinsen |
|-|--------------|--------|
| Training | Gradient descent (STE) | Evolution (EntroMorph) |
| Gradients | Straight-through estimator | None needed |
| Training cost | Massive (GPU clusters) | Small (CPU feasible) |
| Provenance | Standard ML pipeline | Full evolution trace |

**Implication:** BitNet uses gradients with tricks (STE). Yinsen uses evolution, which naturally handles discrete search spaces. Evolution may be less efficient but produces cleaner audit trails.

### Delta 4: Purpose / Value Proposition

| | BitNet b1.58 | Yinsen |
|-|--------------|--------|
| Goal | Efficient LLM inference | Auditable neural computation |
| Metric | Latency, throughput, energy | Traceability, verifiability |
| Target user | Cloud providers, edge LLMs | Regulated industries |
| Pitch | "Same performance, 10x cheaper" | "You can read the network" |

**Implication:** BitNet optimizes *cost*. Yinsen optimizes *trust*. Different value propositions, same mechanism.

### Delta 5: Verification Approach

| | BitNet b1.58 | Yinsen |
|-|--------------|--------|
| Verification | Benchmarks (perplexity, tasks) | Exhaustive testing + audits |
| Formal methods | None mentioned | Feasible at small scale |
| Certification | Not addressed | Explicit goal |
| Transparency | Model weights available | Decision traces per inference |

**Implication:** BitNet proves utility via benchmarks. Yinsen aims to prove correctness via exhaustive verification. Different standards of "works."

### Delta 6: Activation Quantization

| | BitNet b1.58 | Yinsen |
|-|--------------|--------|
| Activations | 8-bit quantized | Float (for now) |
| Activation range | [-Qb, Qb] per token | Native float range |
| Future direction | Lower bit activations | Fixed-point / LUT activations |

**Implication:** BitNet quantizes activations for speed. Yinsen keeps float activations (simpler, more accurate). Both could move toward fixed-point.

---

## What Yinsen Can Learn from BitNet

### 1. Absmean Quantization
BitNet's quantization function is elegant:
```
W̃ = RoundClip(W / mean(|W|), -1, 1)
```
This could be useful if we ever train with gradients, or for converting pre-trained networks.

### 2. The "1.58 bits" Framing
Mathematically: log₂(3) ≈ 1.58 bits
This is more precise than saying "2 bits" (which implies 4 values). Good for papers.

### 3. Scaling Laws Still Apply
BitNet shows ternary networks follow scaling laws - larger networks perform better. This suggests even small Yinsen networks might benefit from being "as large as possible" within constraints.

### 4. Zero Enables Feature Filtering
BitNet explicitly notes that adding 0 (vs pure {-1, +1}) improves performance because it enables "feature filtering" - some inputs are explicitly ignored. Our ternary already includes 0.

### 5. Hardware Co-Design
BitNet calls for custom hardware optimized for 1-bit operations. Yinsen's edge focus aligns with this - FPGAs and custom silicon for ternary ops.

---

## What BitNet Doesn't Address (Yinsen's Opportunity)

### 1. Auditability
BitNet never mentions explainability, traceability, or certification. The weights are ternary, but there's no framework for understanding *why* a decision was made.

### 2. Irregular Time Series
Transformers don't naturally handle irregular sampling. CfC does. For medical/industrial IoT, this matters.

### 3. Tiny Scale
BitNet's smallest model is 700M parameters. Yinsen targets 100-10K parameters. Different universe.

### 4. Evolution / Gradient-Free Training
BitNet uses standard gradient descent with STE. Evolution offers:
- No gradient estimation tricks
- Natural discrete search
- Complete training provenance

### 5. Formal Verification
At 70B parameters, formal verification is impossible. At 100 parameters, it's feasible. Yinsen can make claims BitNet cannot.

### 6. Regulatory Alignment
BitNet is optimized for cloud inference cost. Yinsen is optimized for regulatory approval. Different markets entirely.

---

## Strategic Positioning

### The Landscape

```
                        SCALE
            Small (100)          Large (70B)
         ┌────────────────┬────────────────┐
         │                │                │
Auditable│   YINSEN       │    [Gap]       │
         │   (here)       │                │
PURPOSE  │                │                │
         ├────────────────┼────────────────┤
         │                │                │
Efficient│   [Gap]        │   BitNet       │
         │                │   b1.58        │
         │                │                │
         └────────────────┴────────────────┘
```

### The Narrative

BitNet proves: **Ternary works at scale.**

Yinsen proves (will prove): **Ternary enables auditability.**

These are complementary, not competitive. BitNet legitimizes ternary for the ML community. Yinsen applies it where auditability matters.

### The Citation

When we publish, we cite BitNet b1.58 as:
> "Recent work has demonstrated that ternary weights can match full-precision performance at scale [Ma et al. 2024]. We explore the complementary question: can ternary weights enable auditable neural computation for regulated industries?"

---

## Concrete Takeaways

1. **We're not alone.** Microsoft validated ternary at massive scale. This is tailwind.

2. **Our differentiation is clear.** BitNet = efficiency. Yinsen = auditability.

3. **Consider absmean quantization** if we add gradient training later.

4. **"1.58 bits" is the proper terminology** for ternary weights.

5. **CfC + ternary is novel.** BitNet is transformers. No one has done ternary CfC.

6. **The regulatory angle is unoccupied.** BitNet doesn't mention certification.

7. **Tiny scale is unoccupied.** No one is doing exhaustive verification of ternary networks.

---

## Updated Yinsen Positioning

**Before:** "Ternary neural computation"
**After:** "Auditable 1.58-bit neural computation for regulated time-series inference"

We ride BitNet's coattails for legitimacy, then differentiate on:
- Auditability (decision traces)
- Time series (CfC architecture)  
- Tiny scale (exhaustive verification)
- Regulated industries (certification pathway)

This is the gap. BitNet can't fill it. We can.
