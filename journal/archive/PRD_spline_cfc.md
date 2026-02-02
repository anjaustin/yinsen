# PRD: Smoothstep ProxQuant + Trajectory Distillation for Ternary CfC

**Version:** 1.0
**Date:** 2026-01-31
**Status:** ACTIVE
**Depends on:** PRD_consolidation.md (Phases 1, 2, 4 complete)

---

## Executive Summary

Train ternary CfC networks that actually work. Two techniques, combined:

1. **Smoothstep ProxQuant** -- replace STE's gradient lie with a differentiable polynomial quantizer that anneals from soft to hard ternary over training. No `tanh`, no `exp`. Pure FMA arithmetic.

2. **Trajectory Distillation** -- train a float CfC teacher, then train the ternary student to match the teacher's *hidden state trajectory at every timestep*, not just the final output. CfC cells are ODE solvers; the trajectory is the computation.

Combined, these address the two failure modes observed in `train_sine.c`:
- Post-training quantization destroys learned structure (fixed by training-time quantization)
- Quantization noise drifts recurrent dynamics (fixed by trajectory locking)

---

## Problem Statement

The `train_sine.c` experiment proved the pipeline works but exposed the fundamental gap:

| | Float CfC | Post-Quantized Ternary |
|---|---|---|
| MSE | 0.032 | 0.228 |
| Amplitude | Correct | Compressed to ~0.3x |
| Phase | Correct | Correct |
| Trajectory | Stable | Drifts over time |

The ternary model tracks the right shape but loses magnitude because:
1. Absmean quantization after training doesn't account for what the network *learned* to need
2. Quantization error at each timestep compounds through the recurrence

---

## Technical Design

### Component 1: Smoothstep Quantizer

The smoothstep polynomial `S(t) = 3t^2 - 2t^3` maps [0,1] -> [0,1] with zero derivative at both endpoints. Its derivative is `S'(t) = 6t(1-t)` -- a parabolic bump with finite support.

We use this to build a differentiable ternary quantizer:

```
smoothstep_ternary(w, beta):
    // beta = inverse temperature, starts small (soft), grows large (hard)
    // Quantizes w toward nearest trit {-1, 0, +1}

    For each transition boundary (at -0.5 and +0.5):
        t = clamp((|w| - 0.5) * beta + 0.5, 0, 1)
        transition = 3*t*t - 2*t*t*t
        grad_transition = 6*t*(1-t) * beta

    Forward:  w_q = sign(w) * transition(|w|)  // smooth trit
    Backward: dw_q/dw = grad_transition         // parabolic bump
```

**Temperature schedule:** `beta(epoch) = beta_start + (beta_end - beta_start) * epoch / num_epochs`
- `beta_start = 1.0` (soft, smooth transitions)
- `beta_end = 20.0` (near-hard ternary by end of training)

**Properties:**
- At `beta=1`: weights are smoothly distributed, gradients flow everywhere
- At `beta=20`: weights snap to {-1, 0, +1}, gradients concentrate at decision boundaries
- At `beta=inf`: equivalent to hard ternary quantization (inference mode)
- **No exp() or tanh() anywhere.** Pure polynomial arithmetic.
- Derivative is exactly zero outside the transition window (finite support)
- Compiles to FMA instructions on M4

**Per-row scaling:** Each row maintains a learnable float scale factor `gamma[i]`. The forward pass is:
```
y[i] = gamma[i] * ternary_dot(Q(W_row[i], beta), x) + bias[i]
```
The scale absorbs magnitude information that ternary weights cannot express.

### Component 2: Trajectory Distillation

**Teacher:** A float CfC with 2x width (HIDDEN_DIM=32). Trained with standard SGD on the task until convergence. The teacher's hidden states `h_teacher[t]` are recorded at every timestep.

**Student:** A ternary CfC with the target width (HIDDEN_DIM=16, or matching teacher width for maximum quality). Weights are quantized through smoothstep during training.

**Loss function:**
```
L_total = alpha * L_output + (1 - alpha) * L_trajectory

L_output = MSE(y_student, y_target)           // task loss
L_trajectory = MSE(h_student[t], P(h_teacher[t]))  // trajectory matching
```

Where `P` is a learned or fixed projection from teacher hidden dim to student hidden dim (if dims differ). If dims match, `P = identity`.

**`alpha` schedule:** Start with `alpha=0.3` (mostly trajectory), anneal to `alpha=0.8` (mostly task) as the student stabilizes. Early training locks the manifold; late training refines the output.

**Why trajectory, not just output distillation:**
- CfC cells integrate dynamics over time. Output distillation gives one gradient signal per sequence. Trajectory distillation gives one per *timestep* -- 50x more gradient signal for SEQ_LEN=50.
- Quantization noise at step t compounds to step t+1. Trajectory loss detects and corrects drift immediately, before it amplifies.
- The teacher's hidden state encodes temporal features the teacher learned. The student inherits these features structurally.

### Component 3: Training Pipeline

```
Phase A: Train float teacher (standard SGD, 2x width)
    |
Phase B: Record teacher trajectories on training data
    |
Phase C: Train ternary student
    |   - Smoothstep ProxQuant (beta anneals from 1 to 20)
    |   - Trajectory loss (alpha anneals from 0.3 to 0.8)
    |   - Per-row learned scales
    |
Phase D: Harden and evaluate
    |   - Set beta = infinity (hard ternary)
    |   - Evaluate on test set
    |   - Compare float teacher vs ternary student
```

### Component 4: Greedy Coordinate Refinement (Optional Post-Training)

After Phase D, optionally iterate through each ternary weight:
- Try all 3 values {-1, 0, +1}
- Keep whichever minimizes loss on a calibration set
- O(3 * num_weights) forward passes per sweep
- 1-3 sweeps typically sufficient

For a 609-parameter network: ~1800 forward passes per sweep. Sub-second.

---

## Implementation Plan

### File: `examples/train_sine_v2.c`

A single self-contained C file implementing the full pipeline. No new headers required -- uses existing `ternary.h` and `cfc_ternary.h`.

**New functions to implement:**

1. `smoothstep_quantize(float w, float beta)` -- returns quantized value
2. `smoothstep_grad(float w, float beta)` -- returns gradient multiplier
3. `forward_step_quantized(...)` -- forward pass with smoothstep-quantized weights
4. `backward_step_quantized(...)` -- backward pass with smoothstep gradient chain rule
5. `train_teacher(...)` -- Phase A
6. `record_trajectories(...)` -- Phase B
7. `train_student(...)` -- Phase C with trajectory loss
8. `harden_and_eval(...)` -- Phase D
9. `greedy_refine(...)` -- optional coordinate search

**Estimated LOC:** ~700-900 (the current `train_sine.c` is ~480 and covers Phases A and D only)

### Deliverables

1. `examples/train_sine_v2.c` -- full pipeline implementation
2. Updated `Makefile` with `train-sine-v2` target
3. Results recorded in a comment block at the top of the file
4. If successful: `include/smoothstep_quant.h` -- extract the quantizer as a reusable header

---

## Success Criteria

| Metric | Baseline (current) | Target | Stretch |
|--------|-------------------|--------|---------|
| Ternary MSE on sine | 0.228 | < 0.10 | < 0.05 |
| Ternary/Float MSE ratio | 7.2x | < 3x | < 1.5x |
| Amplitude preservation | ~30% | > 70% | > 90% |
| Trajectory drift at t=100 | Diverged | Bounded | < 2x float |

**The real success criterion:** Can the ternary CfC produce predictions that a human would accept as "sine wave" without being told which is float and which is ternary?

---

## What This PRD Does Not Cover

- **Metal/NEON acceleration** of the smoothstep quantizer (future optimization)
- **Multi-layer CfC** training (single layer first, stack later)
- **Tasks beyond sine** (prove the method, then generalize)
- **STE comparison** (we may add an STE baseline for comparison, but it's not the focus)
- **The KAN/Spline-Weight hybrid** (architecturally interesting but a separate research thread that changes Yinsen's identity as a ternary computation engine)

---

## Risks

1. **Smoothstep may not provide enough gradient signal at high beta.** The parabolic bump narrows as beta increases. Mitigation: cap beta at 20, don't go to infinity during training. Harden only at inference.

2. **Trajectory distillation may over-constrain the student.** If the teacher's trajectory is suboptimal, the student is locked to it. Mitigation: alpha schedule shifts toward task loss late in training.

3. **609 parameters may simply be too few for ternary to express sine dynamics.** Mitigation: test at HIDDEN_DIM=16 and 32. If 32 works and 16 doesn't, that's a clean width-compensation result.

4. **The backward pass through smoothstep + CfC + trajectory loss is complex.** The chain rule has many terms. Mitigation: implement carefully, validate with finite-difference gradient checking.

---

## Changelog

- 2026-01-31: Initial PRD created
