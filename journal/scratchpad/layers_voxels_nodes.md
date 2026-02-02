# Nodes of Interest: Layers, Voxels, and Architectural Next Steps

## Node 1: The Missing Failure
We have no task where ternary CfC demonstrably fails. v2 showed h=32 matches float on sine (1.4x degradation, MSE 0.00036). Without a failure, any architectural addition is a solution looking for a problem. We cannot evaluate layers or voxels without first finding where the current architecture breaks.
Why it matters: The entire experimental methodology depends on having a measurable gap to close.

## Node 2: Width vs Depth — Orthogonal or Redundant?
Width (more neurons per layer) and depth (more layers) are both ways to add parameters. But they add different KINDS of capacity. Width adds parallel feature detectors at the same representational level. Depth adds sequential transformation — function composition. For ternary, width adds more sign decisions at the same level; depth adds sign decisions about already-processed features.
Why it matters: If they're redundant, there's no point adding layers when you can just widen. If they're orthogonal, the combination should be superadditive.

## Node 3: Voxels = Structured Sparsity
A voxel grid with local connectivity is a ternary weight matrix that is mostly zeros BY CONSTRUCTION (not by learning). A 4×4×4 grid (64 neurons) with 6-neighbor connectivity has at most 6 nonzero weights per row in the recurrent matrix, vs 64 for a dense matrix. That's 90% structural sparsity. Ternary is already about sparsity (0 = "ignore this input"). Voxels align the architecture's inductive bias with the weight representation's natural structure.
Why it matters: This could be more parameter-efficient than dense widening. 64 voxel neurons with 6 connections each = 384 active weights. 64 dense neurons = 4096 weights. Same neuron count, 10x fewer active parameters.
Tension with Node 2: Is this efficiency real or just an illusion? The dense matrix CAN learn sparsity; the voxel matrix MUST be sparse. Imposing structure is a bet that the task's structure matches.

## Node 4: CfC + Voxels = Spatiotemporal Dynamics
CfC neurons have per-element time constants (tau). In a voxel grid, different spatial locations evolve at different speeds. This IS a reaction-diffusion system. Fast neurons near slow neurons create wave-like dynamics — information propagates through the grid as spatial patterns of activation.
Why it matters: This is a different computational paradigm from flat RNNs. Instead of a hidden state vector that transforms as a whole, you get spatial patterns that form, propagate, and interact. This is how biological neural tissue works. It's also how physical systems work. If the task involves spatiotemporal dynamics (which many real-world tasks do), this is a natural fit.

## Node 5: The Copy Task as a Diagnostic
The copy task (memorize N tokens, wait K steps, reproduce them) directly tests information fidelity through recurrence. To succeed, the hidden state must encode the input sequence without corruption over K timesteps. Ternary gating introduces quantization noise at every step. If that noise is random, information degrades as sqrt(K). If the ternary sign pattern is structured (some neurons preserve, others process), information can survive.
Why it matters: This separates two failure modes: (a) ternary can't represent the right function (capacity failure), (b) ternary can't maintain precision through time (fidelity failure). Layers address (a). Voxels might address (b) by creating spatially separated memory and processing regions.

## Node 6: Multi-Frequency Prediction as the Minimum Harder Task
Predicting sin(t) + sin(sqrt(2)*t) + sin(pi*t) requires tracking three independent phases. The hidden state must maintain three decoupled oscillators. With h=32 flat, this should be possible in float but might stress ternary because the three frequency components must not interfere through the quantized weight matrix.
Why it matters: This is the gentlest step up from single-frequency. If ternary CfC handles this easily, we need to go harder. If it struggles, we've found a diagnostic task without going all the way to chaos.

## Node 7: Parameter-Controlled Experiments
The v2 experiment's clearest finding was that width (= parameters) dominates. Any new architecture must be compared at EQUAL PARAMETER COUNT to be meaningful. A 2-layer CfC with h=16 per layer has roughly the same parameters as a 1-layer CfC with h=32. If the 2-layer version wins, depth helps. If it loses, depth hurts. If it ties, depth is irrelevant for this task.
Why it matters: Without parameter matching, we can't distinguish "more parameters helps" from "this architecture helps."

## Node 8: The Temporal Unrolling Argument Against Layers
A single-layer CfC unrolled over T timesteps is already T layers deep through time. Adding L spatial layers makes it L×T effective layers deep. For T=50 (our training sequence length), a single-layer CfC is already 50 layers deep in the computational graph. Adding 2 spatial layers makes it 100 effective layers. Gradient flow through 100 layers is hard. The CfC decay mechanism (exponential forgetting) helps but doesn't eliminate vanishing gradients.
Why it matters: Layers might actually make training HARDER for recurrent networks. This is why deep LSTMs are typically 2-4 layers, not 50. The temporal depth already provides plenty of representational capacity.

## Node 9: Ternary Sign Patterns as Spatial Computation
In a flat hidden state, the ternary weight matrix's sign pattern ({+1, -1, 0} per weight) determines a fixed routing: which inputs get added, subtracted, or ignored for each neuron. In a voxel grid, this routing has geometry — neuron (x,y,z) can only be affected by neighbors (x±1, y±1, z±1). The sign pattern becomes a spatial operator. A row of [+1, -1, +1, -1, 0, 0] on a grid looks like a discrete Laplacian (difference of neighbors). Ternary weights on a grid naturally implement convolution-like operations WITHOUT the convolution machinery.
Why it matters: This suggests voxels don't just impose sparsity — they change what the ternary values MEAN. On a grid, ternary weights become local spatial filters. This is fundamentally different from dense ternary, which has no spatial interpretation.

## Node 10: Delta Observer Connection — Transient Structure
The Delta Observer found that semantic clusters are TRANSIENT during training — they form, scaffold learning, then dissolve. In a voxel CfC, spatial patterns in the hidden state could show the same transient structure. During training, you might see wave-like patterns of activation that organize, guide weight updates, then dissolve as the network converges. The voxel structure makes this observable because the dynamics have geometry.
Why it matters: This connects the theoretical framework (Delta Observer) to a concrete experimental prediction. If we build voxel CfC and instrument the training loop, we can look for transient spatial organization.
Tension: This is scientifically interesting but doesn't directly answer "does it improve MSE." Observability is a separate value from performance.

## Node 11: The Simplest Possible Stacked CfC
Before voxels (which are exotic), the simplest architectural change is a 2-layer stacked CfC: Layer 1 (h1 neurons) processes input, its hidden state h1 becomes the input to Layer 2 (h2 neurons), Layer 2's hidden state h2 is projected to output. Each layer has its own W_gate, W_cand, b_gate, b_cand, tau. This is well-understood (stacked LSTMs/GRUs) and easy to implement as a controlled experiment.
Why it matters: It's the minimum viable "does depth help" experiment.

## Node 12: What Would Make Voxels Falsifiable?
For voxels to be a legitimate scientific proposal (not just cool engineering), we need a prediction: "On task X, voxel CfC with N parameters will outperform flat CfC with N parameters because [specific reason]." The specific reason should be something voxels provide that flat doesn't: structured sparsity, spatial dynamics, local feature extraction. If we can't articulate this prediction, voxels are premature.
Why it matters: EntroMorph was falsified and archived. We should apply the same discipline here.
