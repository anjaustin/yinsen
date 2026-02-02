# The Real Gap — RAW

## Date: February 2026

## The question

We built a verified 2-bit computation engine. Eight frozen chip primitives, 204 tests, 43M exhaustive proof. Three working demos on real live data. The tau ablation question is answered. The CfC cell works.

But there are four gaps between "research framework that works" and "thing that matters in the world." What are they really? Not the surface-level "we need X" but the actual structural barriers.

## Stream of consciousness

The ternary weights exist but the demos don't use them. Why? Because the demos use enrollment-based discriminants, not trained weights. The CfC cell takes float weights. The ternary dot product, quantization, and matvec are all proven at the primitive level but they've never been composed into a running CfC. This feels like the gap between having verified transistors and having a working computer. The parts are proven. The system isn't assembled.

But wait — do the demos NEED ternary weights? The keystroke biometric works at 110 ns with float32 weights. The ISS telemetry works at 79 ns. The seismic detector at 67 ns. These are already absurdly fast. The ternary story isn't about speed on Apple Silicon. It's about... what? Energy on Cortex-M? Auditability? The "a regulator can read it" story?

The previous potential_synth.md nailed the headline: "The neural network a regulator can read." Every weight is +1, -1, or 0. A nurse can follow the logic. But we haven't proven that a ternary CfC can actually learn anything useful. EntroMorph was falsified (evolution didn't converge). The training path is the most fundamental gap. Without it, ternary is a storage format, not a computational paradigm.

Actually, is that true? The current demos don't train at all. They initialize weights by hand, then learn a discriminant from enrollment data. The "training" is the PCA-based discriminant learning, not weight optimization. This works surprisingly well. Maybe the right question isn't "how do we train ternary weights" but "how far can enrollment-based discriminants carry us?"

The deployment gap feels real but also maybe premature. Who are we deploying for? The ISS demo is a simulation that also works on real data. The seismic detector connects to GFZ. But these aren't products — they're proof-of-concept demos that happen to touch real infrastructure. The gap between "python shim | C binary" on a MacBook and "C binary on an STM32 reading an accelerometer" is... actually not that big? The C code has zero dependencies. It's pure C99 with math.h. A cross-compilation to ARM would probably just work. But nobody has done it. And "probably works" isn't "proven works."

The customer gap is the scariest one. We have a solution. Who has the problem? The throughline_synth says "regulated industries." The potential_synth says "medical, automotive, aerospace." But these are categories, not companies, not people, not budgets. The honest answer is we don't know who would pay for this. We know who SHOULD want it. We don't know who DOES want it.

Something else is nagging me. The previous LMMs were written before the chip forge, before the demos, before the live data connections. The throughline_synth proposed a roadmap starting with "End-to-end demo: Evolve a CfC, export to C, deploy, verify output." We skipped the evolve step entirely and went straight to hand-initialized CfC with enrollment. And it worked. The roadmap was wrong about what mattered first. What does that mean about the new roadmap?

The thing that actually emerged was: the CfC cell as a universal temporal feature extractor, with PCA discriminant as the decision layer. No training. No evolution. Just: give it data, let it run, learn what "normal" looks like, score deviation. This is closer to a classical anomaly detection system with a neural front-end than it is to a trained neural network. And maybe that's the product.

But then the ternary story becomes: can we replace the float CfC cell with a ternary CfC cell and maintain detection quality? If yes, the energy and auditability benefits apply. If no, we need training to find good ternary weights, which brings us back to the training gap.

The NEON and Metal kernels are another disconnected thread. We have 186 GOP/s peak on Metal, sophisticated NEON kernels with SDOT and I8MM. These are for batch computation — large matrix operations at high throughput. But the demos are single-sample streaming. The kernel work is for a different use case (batch execution of large ternary networks) than the demos (single-cell CfC streaming). Both are valid. Neither connects to the other yet.

What actually scares me: that the ternary verification story and the CfC anomaly detection story are two separate things sharing a repo. The ternary stuff is "provably correct 2-bit arithmetic." The CfC stuff is "temporal anomaly detection that happens to use float weights." The connection is aspirational: "someday we'll run the CfC with ternary weights on verified ternary primitives." But that day hasn't come and we haven't proven it can.

What gives me confidence: the enrollment-based approach actually works on real data from real infrastructure. Three different signal domains (keystroke, ISS telemetry, seismic). The CfC cell is genuinely useful as a temporal feature extractor. The chip forge is clean and well-tested. The falsification discipline is honest. If we can connect the ternary primitives to the CfC demos, we have something real.

## Questions arising

1. Can a ternary CfC cell (weights in {-1,0,+1}) produce useful hidden state dynamics for enrollment-based anomaly detection? Or does the float precision in the weights matter?
2. Is the right training path actually "no training" — just enrollment? And if so, does ternary quantization of the hand-initialized weights preserve detection quality?
3. What's the minimum viable deployment target? Cortex-M4? ESP32? RISC-V? What would it take?
4. Who has actually complained about neural network auditability in the last 12 months? Not in theory — in practice, with budget?
5. Are the NEON/Metal kernel investments stranded, or do they serve a different market than the CfC demos?
6. What happened to EntroMorph? Was it truly falsified, or was the task wrong?

## First instincts

- The fastest path to bridging the gap is: quantize the existing float weights to ternary, run the demos, measure degradation. If quality holds, the bridge is trivial. If it doesn't, we know the real problem.
- The deployment story is closer than it feels. Pure C99 + math.h cross-compiles to anything. The binary size will be tiny. Just do it for one target.
- The customer question can't be answered from inside the repo. It requires going outside.
- The NEON/Metal kernel work serves a future where ternary networks are larger than single CfC cells. That future is real but not now.
