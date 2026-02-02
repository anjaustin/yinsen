# The Real Gap — REFLECT

## Date: February 2026

## Core Insight

**The thing that emerged is not what was planned, and the thing that emerged is the product.**

The planned path was: verified ternary primitives → evolution-trained ternary CfC → deployment to edge → certifiable neural computation. Three LMM cycles and a roadmap all pointed here.

The actual path was: verified ternary primitives → float CfC chip → hand-initialized weights → enrollment-based discriminant → real live data → working anomaly detection across three signal domains.

The planned path required solving training. The actual path sidestepped training entirely. And the actual path produced real results on real infrastructure.

This is the wood telling you where it wants to split.

## Resolved Tensions

### Node 1 vs Node 3: Two Yinsens → One System, Two Timescales

The ternary verification layer and the CfC anomaly detection layer aren't two products. They're two timescales of the same product. The CfC anomaly detection works NOW with float weights. The ternary layer is the FUTURE version that adds auditability and energy efficiency. Node 3 (the quantization test) is the bridge between them.

The ternary layer doesn't need to wait for training. It needs to prove that quantizing the existing float weights to ternary doesn't break detection. If it does break, we know the precision boundary. If it doesn't, we've composed the full stack.

### Node 2 vs Node 4: Enrollment IS the Product

The "training gap" is a gap for general ML. It's not a gap for temporal anomaly detection. Enrollment-based detection is:
- Zero-shot (no labeled anomaly data needed)
- Adaptive (re-enroll for a new environment)
- Auditable (the discriminant is a mean vector + 5 principal components — a human can read it)
- Tiny (268 bytes per channel)

This is not a limitation we're working around. This is the architecture. The CfC cell is a general-purpose temporal feature extractor. The enrollment discriminant is a lightweight, interpretable decision layer. Together they form a system where "training" is just "let it watch normal operation for a while."

The potential_synth headline was: "The neural network a regulator can read." The enrollment discriminant IS readable. 8 mean values, 5 principal components, each with 8 dimensions. That's 268 bytes. A regulator can literally print it on a page.

### Node 5 vs Node 6: Deploy for Yourself Before Deploying for a Customer

The deployment gap and the customer gap are tangled. We can't find a customer without a deployment proof. We can't justify a deployment without a customer. This is a chicken-and-egg that resolves by deploying for yourself.

Buy a $10 dev board. Cross-compile. Run the seismic detector on it. Measure power, latency, memory. Take a photo of the board running. This breaks the loop. Now you have something to show that isn't a MacBook terminal screenshot.

### Node 7 vs Node 9: Kernels Are for Scale, Demos Are for Entry

The NEON/Metal kernels and the CfC demos serve different market entry points. The demos enter at the bottom: single-channel, tiny networks, micro-watt edge devices. The kernels enter at the top: batch ternary inference for larger networks on capable hardware. Both paths are valid. The bottom entry has lower barriers and faster feedback.

Don't confuse the two. The seismic detector doesn't need 186 GOP/s. A vibration sensor on a motor doesn't need Metal compute shaders. But a ternary language model running on an M4 does. Different products, different timelines.

### Node 8 + Node 10: The Moat Is Methodology

The falsification discipline + convergence across three domains is a stronger signal than any single benchmark. It says: "we tested this honestly, and it works in places we didn't expect." The convergence pattern (0.84-0.89 discriminant scores across keystroke/ISS/seismic) is empirical evidence that the CfC + enrollment architecture generalizes.

The competition ships "96% accuracy on benchmark X." We ship "here are all the ways we tried to break it, here's what survived, and here's what didn't." For regulated industries, the second story is more valuable.

## What I Now Understand

The real gap is not four gaps. It's one gap with four symptoms.

**The gap is: the system has never been composed end-to-end from ternary primitives through CfC to live detection on real hardware.**

Every piece works in isolation. The ternary ops are proven. The CfC cell works. The enrollment discriminant works. The live data connections work. But the full stack — ternary weights → ternary CfC → enrollment → live detection → on a target device — has not been built.

The four "gaps" are all consequences of this one unbuilt composition:
1. "Ternary weights aren't in the demos" → compose ternary into CfC
2. "No training path" → not needed; enrollment sidesteps it
3. "No real deployment" → cross-compile the composed stack to a target
4. "No customer" → can't show a customer until you can show the composed stack

## What Surprised Me

The enrollment insight is the biggest thing. Three LMM cycles assumed we needed training. We don't — not for this application class. The CfC cell with hand-initialized weights and enrollment-based discriminants IS the product for temporal anomaly detection. The weights don't need to be optimal; they need to create distinguishable hidden state trajectories for different inputs. The hand-initialized weights do this (proven by convergence at 0.84-0.89 across three domains).

This means the path to product is much shorter than previously estimated. We don't need to solve the training problem. We need to:
1. Prove ternary weights work in this pipeline
2. Cross-compile to a real target
3. Find someone who needs temporal anomaly detection on a constrained device

That's not a year of work. That's weeks.

## Remaining Questions

1. Does ternary quantization of the CfC weights preserve discriminant quality? (Testable immediately)
2. What's the minimum viable target hardware? (Probably STM32F4, ~$5, widely available)
3. Who is currently solving temporal anomaly detection on edge devices, and what are they using? (Market research needed)
4. Is "enrollment-based temporal anomaly detection" a recognized category, or do we need to define it? (Positioning question)
