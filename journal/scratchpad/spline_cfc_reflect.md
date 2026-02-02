# Reflections: Smoothstep ProxQuant + Trajectory Distillation

## Core Insight

The problem is not "how do we quantize better." The problem is "what is a ternary weight actually encoding?"

Node 9 revealed it: per-row scales carry magnitude. Ternary weights carry DIRECTION -- which inputs to add, which to subtract, which to ignore. The quantization challenge is fundamentally about getting the sign pattern right, not approximating a float value.

This reframes everything:

- **Smoothstep/ProxQuant** is about refining which weights sit at decision boundaries (zero vs nonzero, positive vs negative). It's a SIGN optimization technique.
- **Trajectory distillation** is about ensuring the sign pattern produces the right DYNAMICS when integrated over time. It's a dynamics verification technique.
- **Width compensation** is about giving the sign pattern enough degrees of freedom. It's a capacity technique.

These are three different axes. They're orthogonal. They should all help independently.

## Resolved Tensions

### Node 1 vs Node 3: Hard Forward vs Soft Forward
**Resolution: Hard forward, shaped gradient.** The forward pass should always use real ternary weights (hard quantized). The smoothstep only shapes the backward gradient. Reasoning:

1. The student's forward pass must match what happens at inference (hard ternary). Soft forward during training means the student never actually practices with the weights it'll use.
2. The trajectory distillation loss compares student hidden states to teacher hidden states. If the student uses soft weights, the trajectory doesn't reflect real ternary behavior, and the distillation teaches the wrong manifold.
3. Hard forward + shaped gradient is much simpler to implement and validate.

The smoothstep's role is to tell the optimizer: "don't waste gradient on weights that have clearly committed. Focus on the ambiguous ones."

### Node 5 vs Width Compensation: Same Width or 2x?
**Resolution: Both. Run the factorial.** Node 10 is right. We have three independent techniques and we should test them independently. The first experiment uses same-width (HIDDEN_DIM=16 for both teacher and student) to isolate the effect of distillation and shaped gradient. The second uses 2x student width.

### Node 11: Asymmetric Boundaries
**Resolution: One beta, asymmetric by construction.** The smoothstep gradient window at |w|=0.5 (zero/nonzero boundary) matters more than the implicit boundary at w=0 (sign boundary). Since we're using hard forward, the sign is always determined by the hard quantizer. The shaped gradient only needs to handle the zero/nonzero decision. Around w=0, the gradient should be LOW (the weight has committed to zero). Around |w|=0.5, the gradient should be HIGH (the weight is deciding between zero and nonzero). A single smoothstep centered at 0.5 with width 1/beta achieves this naturally.

## Remaining Questions

1. **How good does the teacher need to be?** Node 8 says "very." But how do we quantify "good enough"? Proposal: the teacher should achieve MSE < 0.005 on the sine task before distillation begins. If it can't, the task or architecture is wrong, not the distillation.

2. **Can coordinate search alone match smoothstep + distillation?** Node 12 is a valid concern. We should run coordinate search as a baseline. If it matches, the machinery is unnecessary. But my intuition says it won't match for recurrent cells because coordinate search optimizes weights independently, ignoring the temporal interactions that trajectory distillation captures.

3. **What happens when we go beyond sine?** Sine prediction is so simple that brute-force methods might work. The real test is a task with temporal complexity: multi-frequency signals, ECG patterns, chaotic time series. But sine is the right FIRST test because we can verify correctness visually.

## What I Now Understand

The build should proceed in this order:

1. **Fix the teacher.** The float CfC needs to be good. MSE < 0.005 on sine. This may require Adam (not SGD), learning rate scheduling, or more epochs.

2. **Add coordinate search as the first baseline.** Cheapest possible improvement. Measures the gap that remains for fancier methods.

3. **Add shaped gradient (smoothstep STE).** Hard forward, shaped backward. One beta that anneals. Measure improvement over vanilla post-training quantization.

4. **Add trajectory distillation.** Teacher runs in lockstep with student. Trajectory loss at every timestep. Measure improvement over shaped gradient alone.

5. **Run the factorial.** 8 configurations, all combinations of {shaped gradient, trajectory distillation, width compensation}. Report the full matrix.

6. **Gradient validation.** Finite-difference check before trusting any result. This is step 0, not step 6. Every new gradient computation gets validated.

The PRD's complexity is manageable because Node 6 is right: trajectory loss is additive. The backward pass doesn't fundamentally change. We add one term per timestep.

The smoothstep is simpler than I thought: it's a gradient mask, not a quantizer. The implementation is:
```c
float smoothstep_grad_mask(float w_float, float scale, float beta) {
    float w_normalized = fabsf(w_float / scale);  // normalize by row scale
    float t = clampf((w_normalized - 0.5f) * beta + 0.5f, 0.0f, 1.0f);
    return 6.0f * t * (1.0f - t) * beta;  // parabolic bump
}
```

Applied element-wise during backward: `dL/dW_float[i] = dL/dW_quantized[i] * smoothstep_grad_mask(W_float[i], scale, beta)`

That's it. That's the whole smoothstep contribution. Maybe 10 lines of code.

The synthesis should be a concrete implementation spec with exact function signatures and the factorial experiment design.
