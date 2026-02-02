# Yinsen Diagnostic PRD: Find the Wall, Then Add Geometry

**Version:** 1.0
**Date:** 2026-01-31
**Status:** ACTIVE
**Depends on:** v2 factorial results (completed), LMM synthesis (journal/scratchpad/layers_voxels_synth.md)

---

## Executive Summary

The v2 experiment answered the first question: **ternary CfC can learn.** Flat h=32 with STE + Adam + per-row scales achieves MSE 0.000362 on sine prediction, nearly matching the float teacher (0.000257). Width dominates; smoothstep gradients and trajectory distillation provide no benefit on this task.

The next question: **what breaks ternary CfC, and does geometry fix it?**

This PRD defines three sequential phases. Each depends on the previous.

| Phase | Question | Method |
|-------|----------|--------|
| A | What task breaks flat h=32 ternary? | Run v2 champion against 3 harder tasks |
| B | Does depth help? | 2-layer stacked CfC vs 1-layer, matched params |
| C | Does geometry help? | Voxel CfC (1D ring) vs flat, matched params |

**Critical constraint:** Phase B and C only execute on tasks where Phase A finds >5x degradation vs float. If nothing breaks, that is the finding.

---

## Phase A: Find the Wall

### Motivation

We have no task where ternary CfC demonstrably fails. Without a failure, architectural changes are solutions looking for problems. Phase A is pure diagnostic: same architecture (flat h=32, STE, Adam, per-row scales), harder tasks.

### Three Diagnostic Tasks

**Task 1: Multi-Frequency Prediction**
```
Signal: sin(t) + sin(√2·t) + sin(π·t)
Input:  signal(t)           [1 float]
Output: signal(t + dt)      [1 float]
Sequence length: 50
```
Tests: Can the hidden state track three incommensurate frequencies simultaneously? Ternary gating must maintain decoupled oscillators.

**Task 2: Copy-8-20**
```
Input sequence:  [t1, t2, ..., t8, 0, 0, ..., 0]   (8 tokens + 20 blanks = 28 steps)
Target sequence: [0,  0,  ..., 0,  t1, t2, ..., t8]  (20 blanks + 8 tokens)
Tokens: random ±1
Loss: only on the last 8 timesteps
```
Tests: Information fidelity through recurrence. The hidden state must store 8 values across 20 timesteps of recurrent dynamics. Quantization noise compounds.

**Task 3: Lorenz-x Prediction**
```
System: dx/dt = 10(y-x), dy/dt = x(28-z)-y, dz/dt = xy - 8z/3
Input:  [x(t), y(t), z(t)]    [3 floats]
Output: [x(t+dt), y(t+dt), z(t+dt)]  [3 floats]
Integration: RK4, dt=0.01, sampled every 10 steps (effective dt=0.1)
Sequence length: 50
```
Tests: Sensitivity to quantization noise in a chaotic system. Prediction horizon will be limited; the question is whether ternary's horizon is meaningfully shorter than float's.

### Protocol

For each task:
1. Train float teacher (h=32, Adam, 2000 epochs). Record eval MSE.
2. Train ternary student (STE, no smoothstep, no trajectory distill, per-row scales, 1500 epochs). Record eval MSE.
3. Compute degradation ratio: ternary_MSE / float_MSE.
4. If float teacher fails (MSE > 0.1), the task is too hard for the architecture at this width — try h=64 teacher, or note task difficulty.

### Acceptance Criteria

- [ ] Float teacher MSE < 0.01 on multi-freq and copy tasks
- [ ] Lorenz teacher shows finite prediction horizon (MSE reported over time)
- [ ] Degradation ratio reported for all 3 tasks
- [ ] Tasks with >5x degradation identified as candidates for Phase B/C
- [ ] If no task shows >5x degradation: document "flat ternary sufficient" finding

---

## Phase B: Quick Falsification of Depth

### Motivation

CfC unrolled T=50 is already 50 layers deep through time. Adding spatial layers makes it L×T deep. The prior is negative — depth should hurt or be neutral for ternary recurrence. But priors should be tested, not asserted.

### Design

**2-Layer Stacked CfC:**
- Layer 1: h₁ neurons, processes external input
- Layer 2: h₂ neurons, receives layer 1's hidden state as input
- Output: projection from layer 2's hidden state
- Each layer has its own {W_gate, b_gate, W_cand, b_cand, tau}

**Parameter Matching:**
- Flat h=32 baseline: ~2240 active parameters
- 2-layer target: h₁ = h₂ = 20 → ~1900 active params (close enough)
- Alternative: h₁ = 24, h₂ = 16 (asymmetric)

### Protocol

Only run on task(s) that broke in Phase A (>5x degradation).

1. Train float 2-layer teacher at matched params. Record MSE.
2. Train ternary 2-layer student at matched params. Record MSE.
3. Compare: 2-layer ternary vs 1-layer ternary at equal params.

### Acceptance Criteria

- [ ] 2-layer vs 1-layer comparison at matched parameter count
- [ ] Report: depth helps / hurts / neutral (with magnitude)
- [ ] If depth hurts: confirm gradient path length explanation
- [ ] Phase B completes regardless of outcome (falsification is a valid result)

---

## Phase C: Voxel CfC — The Geometry Experiment

### Motivation

This is where emergence lives.

Ternary weights on a spatial grid become local operators. `[+1, -1]` between neighbors is a discrete gradient. A voxel topology doesn't just impose sparsity — it changes what ternary values *mean*. Combined with CfC's per-neuron time constants, a voxel grid is a reaction-diffusion system: different spatial regions evolve at different speeds, and information propagates as spatial patterns.

### Design: 1D Ring (Minimum Viable Geometry)

Start with the simplest spatial structure — a ring of N neurons where each connects to K neighbors on each side.

```
Neuron layout (N=8, K=1 for illustration):

    0 ── 1 ── 2 ── 3
    |                   |
    7 ── 6 ── 5 ── 4

Each neuron connects to: input + self + K left neighbors + K right neighbors
Connections per neuron: INPUT_DIM + 2K + 1
```

**Concrete configuration:**
- N = 64 neurons
- K = 16 neighbors on each side (window = 2K+1 = 33)
- Connections per neuron: INPUT_DIM + 33 (depending on task input dim)
- Active params per matrix: 64 × (INPUT_DIM + 33) ≈ 2200 for 1-input tasks
- Implementation: dense weight matrix with a binary mask; non-neighbor entries forced to 0

**Why start with 1D ring, not 3D grid:**
- Simplest possible geometry (one topological dimension)
- Easy to visualize (hidden state is a 1D signal on the ring)
- If the ring doesn't show spatial structure, a 3D grid won't either
- Upgrade to 2D/3D is a mask change, not an architecture change

### Parameter Matching

| Architecture | Neurons | Active params (approx) | Structure |
|-------------|---------|----------------------|-----------|
| Flat h=32 | 32 | ~2240 | Dense |
| Ring N=64 K=16 | 64 | ~2200 | Banded/ring |

Same parameter budget, different connectivity. If ring wins, geometry helps.

### Observability: Hidden State Snapshots

Every 100 epochs during training, dump the hidden state at mid-sequence (t=25):
- One float per neuron, written as a row to a CSV file
- Post-hoc analysis: spatial autocorrelation along the ring
  - High autocorrelation = neighboring neurons have similar activation = spatial patterns
  - Low autocorrelation = uniform/random = spatial structure unused
- If autocorrelation is high during training but low at convergence → transient spatial organization (Delta Observer connection)

### Falsifiable Prediction

> "On [failing task from Phase A], ring CfC (N=64, K=16, ~2200 active params) will outperform flat CfC (h=32, ~2200 params) because local connectivity creates decoupled processing regions that prevent quantization noise from propagating globally through the weight matrix."

If the ring loses or ties at matched parameters, the prediction is falsified. Both outcomes are publishable.

### Acceptance Criteria

- [ ] Ring CfC implemented with configurable N and K
- [ ] Ring vs flat comparison at matched parameter count on failing task
- [ ] Hidden state snapshots dumped during training
- [ ] Spatial autocorrelation computed and reported
- [ ] Performance result: geometry helps / hurts / neutral (with magnitude)
- [ ] If spatial patterns emerge: report their temporal evolution (transient vs persistent)
- [ ] If ring loses: document falsification honestly, archive, move on

---

## Phasing and Dependencies

```
Phase A: Diagnostic ──────────────────────────
  "What breaks flat ternary?"               |
  No new architecture. Just harder tasks.    |
  Output: failure task(s) or "nothing breaks"|
                                              |
          ┌── if failure found ──┐            |
          |                       |           |
Phase B: Depth                Phase C: Geometry
  2-layer stacked CfC           Voxel CfC (1D ring)
  Quick falsification           The emergence experiment
  ~1 day                        ~2 days
```

Phase B and C are independent of each other and can run in parallel once Phase A identifies a failing task. If Phase A finds no failure, B and C are deferred — "flat ternary is sufficient at h=32" is the documented finding.

### Estimated Effort

| Phase | Scope | Estimate |
|-------|-------|----------|
| A: Diagnostic | 3 new tasks, reuse v2 training infra | 3-4 hours |
| B: Depth | Stacked CfC forward/backward, matched params | 2-3 hours |
| C: Geometry | Ring topology, masks, hidden state snapshots | 4-5 hours |

**Total: ~1-2 days of focused work.** Less if Phase A reveals no failures.

---

## What This PRD Does Not Cover

- **3D voxel grids.** Only if 1D ring shows promise. Upgrade is a mask change.
- **Attention mechanisms.** Orthogonal to the recurrence question.
- **Larger models.** This is about architecture, not scale. Stay small, stay provable.
- **Biological plausibility arguments.** The reaction-diffusion analogy is suggestive but not a design criterion. Performance is.

---

## Relationship to Previous Work

- **v1 (train_sine.c):** First proof that ternary CfC can learn. SGD, post-training quantization. MSE 0.228.
- **v2 (train_sine_v2.c):** Factorial experiment. Adam, STE, per-row scales. MSE 0.000362. Width dominates. Smoothstep/trajectory distillation don't help.
- **v3 (this PRD):** Diagnostic + architecture. Find what breaks, test whether geometry helps.
- **LMM (layers_voxels_synth.md):** The thinking that produced this PRD.

---

## Changelog

- 2026-01-31: Initial PRD created from LMM synthesis
