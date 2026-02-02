# Reflections: Layers, Voxels, and Architectural Next Steps

## Core Insight

The nodes separate into two clean clusters:

**Cluster A (Nodes 1, 6, 7, 12): We need a failure first.**
Before adding any architecture, we need a task that breaks flat ternary CfC. Without it, we're sharpening an axe that already cuts. The experiment is diagnostic: find the wall, measure its height, then ask what tool clears it.

**Cluster B (Nodes 2, 3, 4, 9, 10): Voxels are more fundamental than layers.**
Layers are a quantitative change (more of the same). Voxels are a qualitative change (different structure of computation). Ternary + voxels creates something new: local spatial operators from discrete sign patterns. This is the emergence the user is looking for — not adding layers to a flat RNN, but discovering what happens when ternary weights have geometry.

The critical resolution: **do A first, then B.** But design A so it naturally leads into B.

## Resolved Tensions

### Node 1 vs Everything Else: Build First or Diagnose First?
**Resolution: Diagnose first. Concretely.**

Run the current h=32 flat ternary (from v2, the champion config 5) against three harder tasks:
1. Multi-frequency: sin(t) + sin(√2·t) + sin(π·t). Gentle step up.
2. Copy task: memorize 8 tokens, wait 20 steps, reproduce. Tests fidelity.
3. Lorenz x-component: one step ahead prediction of dx/dt = σ(y-x). Tests chaos tolerance.

If all three are solved at h=32, ternary CfC is more capable than expected and layers/voxels aren't needed yet. If any fails, we have a target.

### Node 2 vs Node 8: Are Layers Worth It for Recurrent Nets?
**Resolution: Probably not, and here's why.**

Node 8 is the strongest argument. CfC unrolled 50 steps is already 50 layers deep through time. Adding spatial depth on top of temporal depth compounds the gradient path length. Deep LSTMs top out at 2-4 layers for a reason. For ternary, the problem is worse: each spatial layer introduces additional quantization noise that compounds forward and has STE approximation error backward.

The 2-layer CfC experiment (Node 11) is still worth doing as a controlled test, but the prior is that it won't help. If it doesn't, we've falsified "depth helps for ternary CfC on temporal tasks" and can move on.

### Node 3 vs Node 9: Voxels as Sparsity vs Voxels as Geometry
**Resolution: They're the same thing seen from different angles, and that's the point.**

From the efficiency angle: voxel connectivity is sparse, requiring ~10x fewer active weights than dense. From the computational angle: those sparse weights on a grid become local spatial operators. The sparsity isn't a limitation — it's a feature. A ternary weight pattern [+1, -1, 0, +1, -1, 0] on a 3D grid is a discrete gradient operator. You don't need to learn that structure; the grid topology GIVES it to you.

This is analogous to how CNNs don't "waste" parameters on global connections — local connectivity is the right inductive bias for spatial data. The question is whether temporal dynamics on a CfC grid constitute "spatial data." Node 4 says yes: CfC with per-neuron tau on a grid IS a reaction-diffusion system, and reaction-diffusion systems produce spatial patterns (Turing patterns, traveling waves) that carry information.

### Node 7: How to Control for Parameters
**Resolution: Match total ACTIVE parameters, not total neurons.**

- Dense CfC h=32: W_gate is [32, 33], so 1056 params × 2 matrices = 2112 gate+cand params + tau + output. Total ~2200.
- Voxel CfC h=64, 6-neighbor local: W_gate has 64 rows × ~7 nonzero cols (input + 6 neighbors) = 448 params × 2 = 896 gate+cand + tau + output. Total ~1000.
- To match: Voxel CfC h=128 or so, which gives ~2000 active params.

This creates a fair comparison: same parameter budget, different structure. If voxels win at equal parameters, the structure helps. If they lose, the structure hurts.

### Node 10 vs Node 12: Scientific Interest vs Falsifiability
**Resolution: Make the voxel experiment falsifiable AND observable.**

The concrete prediction: "On the multi-frequency task, voxel CfC (4×4×4=64 neurons, 6-neighbor, ~2000 active params) will outperform flat CfC (~2000 params, h≈32) because the spatial structure enables decoupled oscillators — different regions of the grid can track different frequencies without interference through the weight matrix."

The observation: instrument the training loop to dump hidden state snapshots. If spatial patterns emerge (clusters of neurons with similar activation), the voxel structure is being used. If the activation is uniform/random across the grid, the spatial structure is irrelevant.

This gives us two outputs: a performance number (falsifiable) and a visualization (scientifically interesting). Even if voxels lose on performance, the visualization might reveal something about how ternary dynamics work on structured topology.

## Remaining Questions

1. **What's the minimum voxel implementation?** A 3D grid is complex. Could we start with a 1D ring (each neuron connects to 2 neighbors)? That's the simplest spatial structure and still tests the hypothesis.

2. **How does the CfC concat operation work with voxel topology?** Standard CfC concatenates [x; h_prev] and multiplies by a dense W_gate. With voxels, each neuron's gate input should be [x; local_h_prev] where local_h_prev is only the neighboring neurons' hidden states. The input x might broadcast to all neurons or only to a boundary.

3. **Should voxel connectivity be ternary weights, or should the topology just be a mask on a dense ternary matrix?** The cleanest design: the weight matrix is dense ternary, but the voxel topology zeros out non-neighbor connections. The zeros are structural, not learned. The +1/-1 decisions are learned within the topology.

## What I Now Understand

The build should proceed in three phases:

**Phase A: Find the wall.** Run h=32 flat ternary v2 on multi-frequency, copy task, and Lorenz. Report where it breaks. This is pure diagnostic — no new architecture. Reuse the v2 training infrastructure. ~200 LOC of new task definitions, max.

**Phase B: Test depth (quick falsification).** 2-layer stacked CfC at matched parameter count vs 1-layer. If depth doesn't help, we've eliminated one axis and can focus on the other. ~150 LOC modification of v2.

**Phase C: Voxel CfC (the interesting experiment).** Implement the simplest possible spatial CfC — start with a 1D ring, upgrade to 3D grid if it shows promise. Compare against flat at equal active parameters on the task that broke in Phase A. Instrument hidden states for visualization. This is where emergence lives. ~300 LOC new file.

The key insight from the LMM: **the question isn't "would layers or voxels help on sine" — it's "what task reveals whether ternary computation benefits from geometry."** Find the right question before building the answer.
