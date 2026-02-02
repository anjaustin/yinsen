# Raw Thoughts: Smoothstep ProxQuant + Trajectory Distillation for Ternary CfC

## Stream of Consciousness

I've just written a PRD that combines two ideas: polynomial-based differentiable quantization and trajectory-level distillation. I believe these are both sound. But I need to check my own blade.

The smoothstep quantizer is mathematically clean. 3t^2 - 2t^3, derivative 6t(1-t). No transcendentals. Finite support. I like it. But I'm nervous about the mapping from weight-space to trit-space. The smoothstep maps [0,1] to [0,1]. I need to map the real line to {-1, 0, +1}. That means I need TWO transition regions: one around w = -0.5 (transition between -1 and 0) and one around w = +0.5 (transition between 0 and +1). The function is symmetric so I can work on |w| and restore sign. But wait -- what does the function look like for the whole real line?

For |w| < 0.5 - 1/(2*beta): output is 0 (zero region)
For |w| in transition: smoothstep ramps
For |w| > 0.5 + 1/(2*beta): output is sign(w) * 1 (saturated at +/-1)

Actually, I need to think about this more carefully. The weight shadow values live in float space. I need to define the quantizer Q(w, beta) such that:
- As beta -> 0: Q(w) = w (identity, fully soft)
- As beta -> inf: Q(w) = round_ternary(w) (hard ternary)
- Q is differentiable for finite beta
- Q'(w) has finite support (no infinite tails)

The standard ProxQuant uses tanh(w * beta) which maps R -> (-1, +1). That's binary, not ternary. For ternary, I need a "staircase" with three levels. The tanh version would be something like: tanh(beta * (w - 0.5)) / 2 + tanh(beta * (w + 0.5)) / 2. That's two tanh calls and it's messy.

With smoothstep, I can build the staircase from two smoothstep transitions. Let me think...

Actually, the simplest framing: the ternary quantizer maps w to the nearest of {-1, 0, +1}. The decision boundaries are at w = -0.5 and w = +0.5. Near these boundaries, the smoothstep interpolates. Away from boundaries, it snaps.

So: Q(w, beta) = smoothstep_transition(w, -0.5, beta) * (-1) + smoothstep_transition(w, 0, beta) * ... no, this is getting complicated.

Simpler: decompose into sign and magnitude.
- sign_soft(w) = smoothstep mapping from negative to positive around w=0
- magnitude_soft(w) = smoothstep mapping from 0 to 1 around |w|=0.5

Then: Q(w) = sign_soft(w) * magnitude_soft(|w|)

Wait. That gives a continuous function but is it right? At w = 0.3 (should be 0): sign_soft(0.3) is near +1, magnitude_soft(0.3) is near 0. Product: near 0. Good. At w = 0.8 (should be +1): sign_soft(0.8) near +1, magnitude_soft(0.8) near 1. Product: near +1. Good. At w = -0.7 (should be -1): sign_soft(-0.7) near -1, magnitude_soft(0.7) near 1. Product: near -1. Good.

But the gradient through sign_soft at w=0 is problematic. If w is near 0, I want the gradient to push it toward 0 (the zero trit), not through the sign transition. The sign transition gradient near 0 fights the magnitude gradient near 0.

Hmm. Maybe I should not decompose into sign and magnitude. Maybe the cleaner approach is: just define Q(w) as a piecewise polynomial that directly maps to the staircase.

Another thought: do I even need a smooth quantizer during the backward pass? The forward pass always uses hard quantization (the ternary weights). The backward pass uses the smooth gradient to update the float shadow weights. This is essentially STE but with a shaped gradient instead of the identity. The smoothstep's parabolic bump acts as a window function that focuses gradient signal on weights near decision boundaries and zeroes it for weights that have "committed" to a trit value. That's actually a better framing.

So maybe the implementation is simpler than I'm making it:
- Forward: hard quantize (round to nearest trit, using per-row scale)
- Backward: multiply the incoming gradient by the smoothstep derivative at the pre-quantization value

This is "Shaped STE" -- the gradient estimator is not the identity (standard STE) but a bell-shaped function centered at each decision boundary. Weights far from boundaries get zero gradient (they've committed). Weights near boundaries get full gradient (they're still deciding).

This is actually cleaner and avoids all the complications of the smooth forward pass. And it's probably what your associate meant.

Now trajectory distillation. The core idea is clear: match hidden states, not outputs. But I have practical concerns:

1. The teacher's hidden dim may differ from the student's. If teacher is 32 and student is 16, I need a projection P: R^32 -> R^16. Training P adds parameters. Or I make them the same width and accept that the ternary student has 2x the parameters of a naive design.

2. What if the teacher's trajectory is bad? If the float teacher hasn't converged well on the task, the student is locked to a bad trajectory. Need to ensure teacher quality first.

3. The alpha schedule (trajectory vs task loss) needs tuning. Too much trajectory early -> student copies teacher artifacts. Too much task early -> student ignores teacher's dynamics.

4. Memory: storing all h_teacher[t] for all sequences. For SEQ_LEN=50, HIDDEN_DIM=32, 10 sequences: 50 * 32 * 10 * 4 bytes = 64KB. Fine.

5. Actually, the teacher and student can run simultaneously. Teacher forward pass -> record h. Student forward pass -> compare. No need to store all trajectories in advance. Just run them in lockstep.

What scares me: the backward pass through smoothstep + CfC + trajectory loss is going to have a lot of terms. The chain rule from L_trajectory through h_student[t] through the smoothstep-quantized weights is:

dL/dW = sum_t dL/dh[t] * dh[t]/dh[t-1] * ... * dh/dW_quantized * dW_quantized/dW

That last term (dW_quantized/dW) is the smoothstep gradient. It's element-wise so it's easy. But the BPTT through multiple timesteps with the trajectory loss injecting gradient at every step means I'm accumulating gradient from both the task loss at the end AND trajectory losses at every step. That's actually fine -- it's just additive. At each timestep t:

dL/dh[t] = dL_task/dh[t] (from BPTT from future) + dL_trajectory/dh[t] (from trajectory match at t)

The trajectory gradient at each timestep is just 2 * (h_student[t] - h_teacher[t]). Simple.

OK, I think this is more tractable than I feared. Let me move to nodes.

## Questions Arising

- What exactly is the smoothstep ternary quantizer? Forward is hard quantize? Or soft quantize?
- How do I handle the sign ambiguity at w=0 in the gradient?
- Should teacher and student have same hidden dim?
- Is the shaped-STE (smoothstep derivative as gradient mask) better than true smooth forward pass?
- Can I validate the gradient numerically (finite differences)?
- What's the minimum teacher quality needed before distillation helps?
- Does the per-row scale need its own learning rate?

## First Instincts

- Shaped STE (hard forward, smoothstep gradient) is simpler and probably just as effective as smooth forward
- Same hidden dim for teacher and student avoids the projection problem
- The backward pass is manageable because trajectory loss just adds a term at each timestep
- Finite-difference gradient checking is essential before trusting the analytical backward
- The temperature schedule matters more than the exact polynomial choice
- Width compensation (2x) may matter more than either smoothstep or distillation
