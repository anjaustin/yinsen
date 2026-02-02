# The Real Gap — NODES

## Date: February 2026

## Node 1: The Two Yinsens
The repo contains two distinct things wearing one name. Thing A: a verified 2-bit arithmetic engine with exhaustive proofs (ternary dot, matvec, quantization, 43M proof). Thing B: a CfC-based temporal anomaly detection system using float32 weights with enrollment-based discriminants (keystroke, ISS, seismic). They share a repo, share the CfC cell chip, and share the verification discipline. But they have never been composed. Thing A has never run inside Thing B.

Tension: are these complementary layers of one system, or two separate projects?

## Node 2: The Enrollment Insight
The demos don't train weights. They initialize by hand, run CfC on normal data, learn a PCA discriminant, and score deviation. This is fundamentally different from the potential_synth roadmap, which assumed evolution-based training. The thing that actually works is: CfC as temporal feature extractor + linear discriminant as decision layer. No weight optimization. No gradient. No evolution. Just enrollment.

This is powerful and nobody planned it.

## Node 3: The Ternary Quantization Test
The fastest bridge between Thing A and Thing B is: take the existing float32 CfC weights, quantize to ternary, run the demos, measure if detection quality holds. This is a single experiment that answers the most important question. If quality holds, ternary CfC is validated end-to-end. If it doesn't, we know exactly where the precision matters.

This has not been done. It should be done before anything else.

## Node 4: The Training Gap Is Maybe Not a Gap
Previous LMMs treated "no training path" as the biggest gap. But the enrollment approach sidesteps training entirely for anomaly detection. You don't need learned weights when the CfC is a feature extractor and the discriminant is the decision layer. The "training gap" is only a gap for tasks that require specific weight configurations — classification, prediction, control. For anomaly detection, enrollment IS the training.

Reframe: the gap isn't "no training." The gap is "we only know how to do anomaly detection." Is that enough to be a product? Or do we need classification too?

## Node 5: The Deployment Proximity
Pure C99, zero dependencies, math.h only. The code cross-compiles to ARM, RISC-V, anything with a C compiler. The seismic detector is 1,768 bytes total state. The entire binary is probably <20KB. An STM32F4 has 1MB flash and 192KB SRAM. There's no technical barrier to deployment — only the fact that nobody has done it and measured the real numbers.

The gap feels large but is probably a weekend of work with a dev board.

## Node 6: The Customer Vacuum
Three LMM cycles have said "regulated industries" as the target customer. Zero LMM cycles have talked to anyone in a regulated industry. This is the most dangerous gap because it's the only one that can't be closed from inside the repo.

What we think we know: medical devices need auditability, aerospace needs determinism, industrial IoT needs edge inference. What we actually know: nothing, empirically.

## Node 7: The Kernel Archipelago
NEON kernels (186 GOP/s), Metal GPU kernels (tiled cooperative), SME on M4 — this is high-performance batch computation for large ternary matrix operations. The demos are single-sample streaming on tiny networks. These serve different use cases. The kernels are for future large ternary networks (language models, vision). The demos are for edge anomaly detection.

Neither invalidates the other. But they compete for attention and create a narrative that's hard to explain. "We have 186 GOP/s ternary kernels AND a 1,768-byte anomaly detector" sounds like two products.

## Node 8: The Falsification Capital
Every other system in this space ships benchmarks that look good. We ship falsification records that document failures honestly. EntroMorph falsified. Tau ablation initially failed. v1 ISS was degenerate. All documented, all archived.

This is an asset. Customers in regulated industries don't trust systems that claim to always work. They trust systems that document when they don't.

## Node 9: The "No Inference" Discipline
We say "execution" not "inference." There is no runtime, no graph executor, no model loader. You call a function. This isn't just language discipline — it's an architectural decision. The CfC cell is a `static inline` function. It compiles to ~50 instructions. There's nothing between you and the math.

This is the deepest differentiator and the hardest to explain. Most ML frameworks add complexity to manage complexity. We removed everything until only the computation remains.

## Node 10: The Convergence Pattern
Across all three demos, the CfC converges to a stable discriminant score on real data:
- Keystroke: 0.89 mean on enrolled user
- ISS CabinP: 0.84 on real ISS data
- Seismic HHZ: 0.87 peak on GFZ data

The discriminant mechanism works across domains. The numbers are consistent. This isn't luck — it's a genuine representation being built by the CfC dynamics + PCA extraction pipeline. The question is whether this generalizes to domains we haven't tested.

## Node 11: The Previous Roadmap Was Wrong
throughline_synth proposed: "Evolve a CfC, export to C, deploy, verify output." We never evolved anything. We built chips, hand-initialized weights, enrolled on data, and connected to live infrastructure. The thing that worked was not the thing that was planned. The emergence went sideways in a useful direction.

This suggests the next roadmap should be less prescriptive and more opportunistic. Plan to test hypotheses, not to build features.

## Node 12: What Actually Shipped
- 8 frozen chip primitives (verified, tested)
- 3 working demos on real live data
- Tau ablation validated (a genuine scientific result)
- 2 live data connections (ISS Lightstreamer, GFZ SeedLink)
- 204 tests passing, 43M exhaustive proof
- Honest falsification record

What didn't ship:
- Ternary CfC (ternary weights in CfC)
- Training/evolution of any kind
- Cross-compiled deployment to embedded target
- Customer validation
- Formal write-up / paper

## Connections Map
```
Node 1 (Two Yinsens) <--tension--> Node 3 (Quant Test bridges them)
Node 2 (Enrollment) <--resolves--> Node 4 (Training gap reframed)
Node 5 (Deployment) <--blocked by--> Node 6 (No customer to deploy for)
Node 7 (Kernels) <--separate from--> Node 9 (No Inference discipline)
Node 8 (Falsification) <--asset for--> Node 6 (Customer trust)
Node 10 (Convergence) <--validates--> Node 2 (Enrollment works)
Node 11 (Wrong roadmap) <--informs--> Node 12 (What shipped)
```

## Delta Items (boundary cases, check carefully)
- Is enrollment a form of training? (It learns a discriminant, but not weights)
- Is the ternary dot product "used" if it's proven but not composed into CfC? (Verified but not deployed)
- Is "anomaly detection" a product category or a technique? (Matters for positioning)
- Are the NEON/Metal kernels technical debt or strategic investment? (Depends on timeline)
