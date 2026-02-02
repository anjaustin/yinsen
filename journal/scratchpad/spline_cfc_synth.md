# Synthesis: Smoothstep ProxQuant + Trajectory Distillation

## Architecture

Three techniques. Three independent axes. One factorial experiment.

```
                    TECHNIQUE A           TECHNIQUE B             TECHNIQUE C
                    Gradient Shape        Distillation            Width
                    ──────────────        ────────────            ─────
                    Vanilla STE           No teacher              16 hidden
                         vs                   vs                     vs
                    Smoothstep mask       Trajectory lock         32 hidden
```

8 configurations. Each trains in seconds. Report the full matrix.

## Key Decisions

**1. Hard forward, shaped backward** (from Nodes 1, 2, 3 — resolved in Reflection)
The student always uses hard ternary weights in the forward pass. The smoothstep only masks the gradient during backward. This ensures the student trains on exactly the weights it will use at inference.

**2. Teacher must be good first** (from Node 8)
Before any distillation experiment, the float teacher must achieve MSE < 0.005 on sine. This likely requires Adam optimizer + LR scheduling. The teacher training is Phase A; it must be validated independently.

**3. Per-row scales are non-negotiable** (from Node 9)
Every configuration uses per-row learned float scales. The experiment varies gradient shape, distillation, and width — not scaling. Scales are the baseline that enables everything else.

**4. Gradient validation before training results** (from Node 7)
Every gradient computation is checked against finite differences before the first real training run. No exceptions. EntroMorph's falsification taught this lesson.

**5. Coordinate search is a baseline, not a final step** (from Node 12)
Run it. Report it. If it wins, the fancier methods are unnecessary. That's a valid finding.

## Implementation Spec

### File: `examples/train_sine_v2.c`

```
SECTION 1: Configuration and types         (~50 LOC)
SECTION 2: Smoothstep gradient mask         (~15 LOC)
SECTION 3: Forward pass (hard quantized)    (~80 LOC, adapted from v1)
SECTION 4: Backward pass (with shaped grad) (~120 LOC, adapted from v1)
SECTION 5: Adam optimizer                   (~40 LOC)
SECTION 6: Teacher training (float)         (~60 LOC)
SECTION 7: Student training (ternary)       (~80 LOC, trajectory loss here)
SECTION 8: Coordinate search baseline       (~40 LOC)
SECTION 9: Factorial experiment driver      (~100 LOC)
SECTION 10: Main + reporting                (~50 LOC)
                                     TOTAL: ~635 LOC
```

### Function Signatures

```c
// --- Smoothstep gradient mask ---
// Returns gradient multiplier for weight w_float with per-row scale.
// beta controls window width (1.0=wide/soft, 20.0=narrow/hard).
static float smoothstep_grad_mask(float w_float, float scale, float beta);

// --- Forward step (hard ternary, with per-row scales) ---
// Quantizes weights to ternary, applies per-row scales, runs CfC step.
// Stores intermediates in cache for backward pass.
static void forward_step_hard_ternary(
    const float* x, const float* h_prev,
    const TrainParams* float_weights,
    const float* gate_scales, const float* cand_scales,
    float beta,                      // for gradient mask (not used in forward)
    StepCache* cache);

// --- Backward step (with shaped gradient and trajectory loss) ---
// dL_dh_new: gradient from future timestep (BPTT) + trajectory loss
// The trajectory loss gradient is added BEFORE calling this function.
static void backward_step_shaped(
    const float* h_prev, const StepCache* cache,
    const TrainParams* float_weights,
    const float* gate_scales, const float* cand_scales,
    float beta,
    const float* dL_dy, const float* dL_dh_new,
    GradParams* grad, float* dL_dh_prev);

// --- Adam optimizer ---
static void adam_step(
    TrainParams* params, const GradParams* grad,
    AdamState* state, float lr, int t);

// --- Gradient validation ---
// Returns max relative error between analytical and numerical gradients.
static float validate_gradients(
    const TrainParams* params, float beta,
    const float* gate_scales, const float* cand_scales);

// --- Coordinate search ---
// Tries all 3 trit values for each weight, keeps best.
// Returns MSE after refinement.
static float coordinate_search(
    TrainParams* params,
    const float* gate_scales, const float* cand_scales,
    int max_sweeps);

// --- Teacher training ---
static float train_teacher(TrainParams* teacher, int num_epochs);

// --- Student training ---
typedef struct {
    int use_shaped_grad;        // 0 = vanilla STE, 1 = smoothstep
    int use_trajectory_distill; // 0 = output only, 1 = trajectory
    int hidden_dim;             // 16 or 32
    float beta_start;           // smoothstep temperature start
    float beta_end;             // smoothstep temperature end
    float alpha_start;          // trajectory weight start
    float alpha_end;            // trajectory weight end
} StudentConfig;

static float train_student(
    const TrainParams* teacher,
    TrainParams* student,
    const StudentConfig* config,
    int num_epochs);
```

### The Factorial Experiment

```c
StudentConfig configs[8] = {
    // shaped_grad, traj_distill, hidden_dim
    {0, 0, 16},  // Baseline: vanilla STE, no distill, narrow
    {1, 0, 16},  // Shaped gradient only
    {0, 1, 16},  // Trajectory distillation only
    {1, 1, 16},  // Both, narrow
    {0, 0, 32},  // Vanilla STE, no distill, wide
    {1, 0, 32},  // Shaped gradient, wide
    {0, 1, 32},  // Trajectory distillation, wide
    {1, 1, 32},  // Both, wide
};
```

Output table:
```
Config | Shaped | Distill | Width | Train MSE | Eval MSE | vs Float | vs Baseline
-------+--------+---------+-------+-----------+----------+----------+------------
   1   |   No   |   No    |  16   |   ...     |   ...    |   ...x   |    1.0x
   2   |  Yes   |   No    |  16   |   ...     |   ...    |   ...x   |    ...x
   3   |   No   |  Yes    |  16   |   ...     |   ...    |   ...x   |    ...x
   4   |  Yes   |  Yes    |  16   |   ...     |   ...    |   ...x   |    ...x
   5   |   No   |   No    |  32   |   ...     |   ...    |   ...x   |    ...x
   6   |  Yes   |   No    |  32   |   ...     |   ...    |   ...x   |    ...x
   7   |   No   |  Yes    |  32   |   ...     |   ...    |   ...x   |    ...x
   8   |  Yes   |  Yes    |  32   |   ...     |   ...    |   ...x   |    ...x
  CS   |  Post-training coordinate search on Config 1      |    ...x
```

### Temperature and Alpha Schedules

```
beta(epoch)  = 1.0 + 19.0 * (epoch / num_epochs)        // 1.0 -> 20.0
alpha(epoch) = 0.3 + 0.5 * (epoch / num_epochs)          // 0.3 -> 0.8
```

### Gradient Validation Protocol

Before any training:
1. Set random weights, random input sequence
2. For each weight w_i:
   - Compute analytical gradient g_a
   - Compute numerical gradient: g_n = (L(w_i + eps) - L(w_i - eps)) / (2 * eps)
   - Relative error: |g_a - g_n| / max(|g_a|, |g_n|, 1e-7)
3. Report max relative error
4. ABORT if max error > 1e-3

## Success Criteria

- [ ] Gradient validation passes (max relative error < 1e-3)
- [ ] Float teacher achieves MSE < 0.005 on sine
- [ ] Full 8-config factorial runs and results are reported
- [ ] At least one ternary config achieves MSE < 0.05 (10x better than v1's 0.228)
- [ ] The factorial reveals which techniques matter (main effects and interactions)
- [ ] Results are reproducible (fixed seed, deterministic)
- [ ] Coordinate search baseline is reported

## What Emergence Looks Like

If the synthesis is right, we should see:
1. Width compensation is the largest effect (Node 9 + BitNet Reloaded)
2. Trajectory distillation is the second largest (Node 4)
3. Shaped gradient provides modest improvement (Node 2)
4. The combination at width=32 approaches float teacher quality (MSE < 0.01)
5. Coordinate search helps but doesn't match trajectory distillation

If we see something different, that's MORE interesting. The point is to measure, not to confirm.
