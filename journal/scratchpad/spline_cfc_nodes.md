# Nodes of Interest: Smoothstep ProxQuant + Trajectory Distillation

## Node 1: Hard Forward vs Soft Forward
The PRD assumes a smooth forward pass (weights are continuously quantized). But the RAW phase revealed that a "Shaped STE" (hard forward, polynomial gradient) may be simpler and equally effective. The forward pass always uses real ternary weights. Only the backward gradient is shaped by the smoothstep derivative, focusing learning on weights near decision boundaries.
**Why it matters:** This changes the entire implementation. Hard forward + shaped gradient is ~50 LOC. Smooth forward + analytical backward through the smoothstep is ~200 LOC and has subtle chain rule issues.

## Node 2: The Smoothstep is a Gradient Window, Not a Quantizer
Reframing: the smoothstep's purpose is not to "softly quantize" -- it's to MASK the gradient. Weights that have committed to a trit value (far from boundaries) get zero gradient. Weights near boundaries get full gradient. This is a learnable attention mechanism over the weight space. The polynomial is just the window shape.
**Why it matters:** This reframing simplifies the mental model and the code. The "temperature" beta controls the window width, not the quantizer softness.

## Node 3: The Temperature Schedule is the Training Algorithm
The choice of beta(epoch) IS the training strategy. Low beta = wide window = all weights learn = soft quantization. High beta = narrow window = only boundary weights learn = near-hard quantization. The entire ProxQuant vs STE debate reduces to: what shape is your gradient window, and how does it evolve?
**Tension with Node 1:** If the forward is always hard, then beta only affects gradient flow. If the forward is soft, beta affects both the forward computation AND gradient flow.

## Node 4: Trajectory Distillation is Regularization
Matching the teacher's hidden states is a form of trajectory regularization. It constrains the student's dynamics to a manifold defined by the teacher. This is distinct from output distillation (which only constrains the final mapping). The teacher's trajectory encodes the SEQUENCE of intermediate computations, which is exactly what quantization noise disrupts.
**Why it matters:** This means trajectory distillation could work even if the teacher uses a different architecture. The constraint is on the dynamics, not the weights.

## Node 5: Same-Width Teacher Eliminates the Projection Problem
If teacher and student have the same hidden dim, trajectory matching is just element-wise MSE on the hidden states. No learned projection needed. The teacher is "the same network but with float weights." This is the simplest version and should be the first experiment.
**Tension with width compensation:** If ternary needs 2x width, and we match teacher dim, we're not testing width compensation. Two experiments needed: (a) same width, trajectory distilled; (b) 2x width student, no distillation.

## Node 6: The Backward Pass is Additive, Not Nested
At each timestep t during BPTT, the gradient has two sources:
- dL_task/dh[t]: propagated backward from future timesteps (standard BPTT)
- dL_traj/dh[t]: local trajectory matching loss (2 * (h_student - h_teacher))

These ADD. There's no nesting or chain rule complication between them. This makes the implementation straightforward: the existing backward_step function just gets one extra additive term in dL_dh_new.
**Why it matters:** The implementation is a ~20 line change to the existing training loop, not a rewrite.

## Node 7: Gradient Validation is Non-Negotiable
The analytical backward pass through CfC + shaped gradient + trajectory loss has enough terms that a sign error is likely. Finite-difference gradient checking (perturb each weight by epsilon, measure loss change) must be implemented before any training results can be trusted.
**Why it matters:** EntroMorph was "falsified" precisely because results weren't validated. We can't repeat that pattern.

## Node 8: The Baseline is Wrong
The current float teacher (MSE 0.032 on sine) is mediocre. A well-tuned float CfC should get MSE < 0.001 on sine prediction. If the teacher is weak, distillation is teaching the student to be mediocre. The teacher needs to be good before we measure the distillation gap.
**Why it matters:** We might be measuring "bad teacher" effects and attributing them to "ternary quantization" effects.

## Node 9: Per-Row Scale is the Third Leg
The first `train_sine.c` run without per-row scales had 287x degradation. With per-row scales: 7.2x. The scales carry most of the information. They're float numbers, one per row, that modulate the ternary dot product output. The smoothstep and distillation operate on top of this already-critical mechanism.
**Why it matters:** If scales account for most of the accuracy, the ternary weights themselves are mostly providing SIGN information (direction), and the scales provide MAGNITUDE. The smoothstep should focus on getting the signs right.

## Node 10: Three Independent Variables
We have three techniques that can be tested independently:
1. Smoothstep shaped gradient (vs vanilla STE vs no QAT)
2. Trajectory distillation (vs output-only distillation vs no distillation)
3. Width compensation (16 vs 32 hidden)

A proper experiment tests these in a 2x2x2 factorial. 8 configurations. Each takes seconds to train. We should run all 8 and report the full matrix instead of just the "best" combination.
**Why it matters:** We'll know which technique actually matters, not just whether the combination works.

## Node 11: The Quantizer Boundary at w=0 Is Special
The boundary between trit=0 and trit=+1 (or trit=-1) is at |w|=0.5 (after scaling). But w=0 itself is also special: it's the boundary between positive and negative trits. For ternary, a weight near 0 should map to trit=0 (zero), not oscillate between +1 and -1. The gradient window should be WIDE around 0 (easy to be zero) and NARROW around +/-0.5 (sharp transition between zero and nonzero).
**Tension with Node 3:** A single beta controls all windows. May need separate betas for the zero-boundary and the magnitude-boundary.

## Node 12: Greedy Coordinate Search is a Strong Baseline
Before building all this machinery, we should try the simplest possible post-training fix: iterate through each weight, try all 3 trit values, keep the best. For 609 parameters, this is ~1800 forward passes (~milliseconds). If this alone closes the gap, the smoothstep machinery is unnecessary.
**Why it matters:** Honest evaluation requires comparing against simple baselines, not just the status quo.
