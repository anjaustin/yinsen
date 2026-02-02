# Reflections: Linear Discriminant on CfC Hidden States

## Core Insight

The linear discriminant has two orthogonal signals, and the current implementation only uses one at a time. The mean vector captures WHERE the user lives in hidden space (position). The PCA captures HOW the user moves through hidden space (trajectory shape). Probe4 proved both carry real information — mean-only gets 14/20 on the hard case, PCA(5) gets 17/20. But they're measured separately. A combined score would use both.

The reason PCA(1-2) fails is architectural: the current z-score scoring REPLACES position with shape. With only 1-2 PCs, most of the position information is projected away, and the shape information is too low-dimensional to discriminate. At PCA(5), enough of the 8-dimensional space is retained that position leaks through as a side effect. PCA(5) isn't "better PCA" — it's accidental mean recovery through high-dimensional coverage.

This means the right architecture is not "mean OR PCA" but "mean AND PCA" — a hybrid score.

## Resolved Tensions

### Node 1 vs Node 3: Mean vs PCA(5)
Resolution: They're not competing. Mean captures inter-user position difference. PCA captures intra-user trajectory consistency. On easy/medium tests (large speed differences), both work because speed affects both position AND trajectory. On the hard test (same speed, different jitter), position is similar but trajectory differs — that's where PCA adds its +3/20 improvement.

The v3 discriminant should compute both distances and combine them.

### Node 2 vs Node 4: Why low PCs fail but high PCs also degrade
Resolution: This is a dimensionality sweet spot, not a complexity tradeoff. PCA(1-2) fail because they project out the mean signal without enough shape signal to compensate. PCA(7) fails because 7 of 8 dimensions is nearly full-rank — the subspace accepts almost any point, so it can't reject impostors. The sweet spot is where the subspace retains enough enrollment structure to be selective while projecting out enough noise to be robust. For 8-dimensional hidden state, that's ~5 PCs (~62% of dimensions).

Rule of thumb for the v3 spec: N_PCS = ceil(HIDDEN_DIM * 0.6).

### Node 5 vs Node 6: Scoring architecture fix
Resolution: The hybrid score is straightforward:
1. Compute mean distance: Euclidean distance from auth hidden state to enrollment mean, normalized
2. Compute PCA distance: z-score distance in enrolled PC subspace (existing method)
3. Combined score: weighted average, sigmoid mapped

The mean distance acts as a gate — far from the mean, you're out regardless. The PCA distance provides fine-grained discrimination for candidates near the mean.

## What Survives Falsification

1. **CfC hidden state consistency (0.897 cosine)**: Confirmed across all probes. The architecture produces stable representations without training.

2. **Mean-based separation on speed differences**: 19-20/20 on easy and medium. This is rock-solid.

3. **PCA adds value on the hard case**: 14/20 (mean) → 17/20 (PCA5). Consistent across reruns (probe3 showed 15/20 with PCA3 on the same case). The improvement is real.

4. **Control stays at noise floor**: 10-12/20 across all methods. No false separation.

5. **156-244 byte discriminant**: Fits in MCU registers regardless of PC count.

## What's Fragile

1. **The hard case at 75-85%**: This is a simulated hard case with controlled jitter. Real users typing at the same speed would introduce other variation (key preferences, digraph timing, pauses) that simulation doesn't capture. 75-85% could be optimistic OR pessimistic depending on how real timing data distributes.

2. **Single-session enrollment**: All 80 enrollment samples come from one continuous session. Cross-session consistency is untested. A real deployment would need multi-session enrollment.

3. **Untrained CfC weights**: The current separation comes from the CfC architecture (how it maps dt into hidden dynamics), not from learned features. Training could dramatically improve the hard case — or it might not, if the architecture already extracts what's extractable from timing alone.

4. **The sigmoid mapping parameters**: center=2.0, scale=1.0 are hand-picked. They don't affect rank ordering (who wins A vs B) but they affect threshold-based decisions (score > 0.5 → accept). A real deployment needs calibration.

## Remaining Questions

1. Does the hybrid (mean + PCA) score actually improve on the hard case? This is the key experiment for v3.
2. What's the optimal weighting between mean and PCA distances?
3. Does the improvement survive with real timing data (not simulated)?

These questions belong to the v3 implementation and evaluation, not to this LMM cycle. The current cycle's job is to produce a concrete spec.

## What I Now Understand

The linear discriminant is a two-component system:
- **Position** (mean): where in hidden space the user's typing maps to
- **Shape** (PCA): the characteristic trajectory through hidden space

The current implementation tests each component separately. The v3 implementation should combine them. Mean-only is the robust baseline (32 bytes, always works when speeds differ). PCA(5) is the accuracy upgrade (244 bytes, helps when speeds are similar). The hybrid is the full system.

The discriminant size (32-244 bytes) is irrelevant for any realistic deployment target. Choose based on accuracy.

The CfC is doing its job — producing consistent, discriminative hidden states without training. The remaining work is purely in the readout layer.
