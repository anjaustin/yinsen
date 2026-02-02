# Synthesis: Keystroke Biometric v3 — Hybrid Linear Discriminant

## Architecture

```
keystroke (dt) ──► CfC Cell ──► hidden state (8 floats)
                                     │
                                     ├──► Enrollment: collect N samples
                                     │         │
                                     │         ├── compute mean vector
                                     │         ├── compute top 5 PCs (power iteration)
                                     │         ├── project mean onto PCs → enrolled projection stats
                                     │         └── store discriminant (244 bytes)
                                     │
                                     └──► Authentication: per keystroke
                                               │
                                               ├── mean distance: ||h - enrolled_mean|| / enrolled_radius
                                               ├── PCA distance: z-score in enrolled PC subspace
                                               ├── hybrid score: sigmoid(alpha * (1 - mean_dist) + beta * (1 - pca_dist))
                                               └── output: score ∈ [0, 1]
```

## Key Decisions

1. **N_PCS = 5** because probe4 showed PCA(5) is the sweet spot for 8-dim hidden state. PCA(3) underperforms. PCA(7) overfits.

2. **Hybrid scoring** because mean and PCA capture orthogonal information (position vs trajectory shape). Mean-only: 14/20 hard. PCA(5)-only: 17/20 hard. Both together should push higher.

3. **No Welford normalization in CfC path** (confirmed by probe1/probe2 falsification — it destroys hidden state consistency).

4. **10-keystroke warmup** (confirmed sufficient by RAW phase tau analysis: slowest tau 0.80, after 10 steps at dt=0.15 → 85% initial state decay).

5. **Independent seeds** for enrollment vs auth in simulation (the v1 original sin).

6. **Honest reporting** — if separation is below threshold, say so.

## Discriminant Structure (v3)

```c
#define KS_HIDDEN_DIM  8
#define KS_N_PCS       5
#define KS_WARMUP      10

typedef struct {
    float mean[KS_HIDDEN_DIM];                      // 32 bytes — enrollment centroid
    float pcs[KS_N_PCS][KS_HIDDEN_DIM];             // 160 bytes — principal components
    float pc_mean[KS_N_PCS];                         // 20 bytes — mean projection per PC
    float pc_std[KS_N_PCS];                          // 20 bytes — std projection per PC
    float radius;                                     // 4 bytes — mean distance normalizer
    int valid;                                        // 4 bytes — enrollment complete flag
} KeystrokeDiscriminant;  // Total: 240 bytes
```

## Enrollment Algorithm

```
Input: N keystroke dt values (N=80 recommended, minimum 40)
Output: KeystrokeDiscriminant

1. Initialize CfC hidden state h = 0
2. Feed first KS_WARMUP keystrokes (discard hidden states)
3. For keystrokes KS_WARMUP..N:
     a. Step CfC: h = cfc_cell(h, [dt], weights)
     b. Store h in sample buffer
4. Compute mean = average of all stored samples
5. Center samples: samples[i] -= mean
6. Power iteration for 5 PCs:
     For each pc in 0..4:
       v = random unit vector
       For 20 iterations:
         v = (samples^T @ samples) @ v  (via accumulated dot products)
         normalize v
         deflate: samples -= (samples @ v) * v^T
       pcs[pc] = v
7. For each pc, project all (uncentered) samples:
     proj[i] = dot(sample[i] - mean, pcs[pc])
     pc_mean[pc] = average(proj)
     pc_std[pc] = std(proj), min 1e-6
8. Compute radius = average ||sample[i] - mean|| across all samples
     (If radius < 1e-6, set radius = 1.0)
9. Set valid = 1
```

## Authentication Scoring

```
Input: live hidden state h, enrolled KeystrokeDiscriminant D
Output: score ∈ [0, 1]

1. Mean distance:
     mean_dist = ||h - D.mean|| / D.radius
     mean_score = exp(-mean_dist^2 / 2)    // Gaussian falloff, 1.0 at center

2. PCA distance:
     For each pc in 0..4:
       proj = dot(h - D.mean, D.pcs[pc])
       z[pc] = |proj - D.pc_mean[pc]| / D.pc_std[pc]
     pca_dist = average(z)
     pca_score = exp(-pca_dist^2 / 2)      // Same Gaussian falloff

3. Hybrid:
     score = 0.6 * mean_score + 0.4 * pca_score
     // Mean-weighted because it's the stronger signal (19/20 vs 17/20 on medium)
     // PCA provides the fine discrimination on the hard case
```

Why Gaussian falloff instead of sigmoid: `exp(-x^2/2)` gives a natural distance-to-score mapping where 0 distance = 1.0 score, and it falls off smoothly. No hand-picked center/scale parameters. The enrolled radius and pc_std provide natural normalization.

Why 0.6/0.4 weighting: Mean is the stronger signal in 3 of 4 test cases. PCA's advantage is specifically on the hard case. Weighting mean higher makes the system more robust across all conditions while still benefiting from PCA's trajectory discrimination.

## Simulation Mode

Same structure as probe3/probe4:
- User A: enrolled user (consistent timing parameters)
- User B: impostor (different timing parameters)
- 20 independent trials
- Each trial: fresh enrollment for A, fresh auth for both A and B
- Report: avg scores, separation, A-wins count
- 4 difficulty levels: easy (3x speed), medium (1.5x speed), hard (same speed/diff jitter), control

Run the hybrid against mean-only and PCA(5)-only for direct comparison.

## Interactive Mode

Same as v2 but with hybrid discriminant:
1. Enrollment phase: type 80+ characters to enroll
2. Auth phase: type to authenticate, see live score
3. Display: score, mean component, PCA component, verdict

## Success Criteria

- [ ] Easy case: 20/20 (must match or exceed mean-only and PCA5)
- [ ] Medium case: 20/20 (must match probe4 PCA5)
- [ ] Hard case: >= 17/20 (must match or exceed probe4 PCA5 best of 17/20)
- [ ] Control: <= 12/20 (must stay at noise floor)
- [ ] Discriminant size: <= 256 bytes
- [ ] Execution per keystroke: < 200 ns
- [ ] Honest output: reports actual separation, doesn't inflate
- [ ] All 199 existing tests still pass (no regressions)

## What This Does NOT Address

- Multi-session enrollment (future work)
- Trained CfC weights (future work — would need training infrastructure)
- Real keystroke data (future work — needs terminal raw mode)
- Threshold calibration for accept/reject decisions (needs ROC analysis with real data)
- The 75-85% hard case ceiling with untrained weights (fundamental limit until training)

## Files to Modify

- `examples/keystroke_biometric.c` — rewrite scoring pipeline from v2 (cosine template) to v3 (hybrid discriminant)

## Files to Create

None. The v3 logic goes directly into the existing demo file.
