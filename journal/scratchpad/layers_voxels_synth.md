# Synthesis: Find the Wall, Then Add Geometry

## Architecture

Three sequential experiments. Each depends on the previous.

```
PHASE A: DIAGNOSTIC           PHASE B: FALSIFICATION        PHASE C: EMERGENCE
"What breaks flat ternary?"   "Does depth help?"             "Does geometry help?"
──────────────────────────    ─────────────────────          ──────────────────────
h=32 flat ternary (v2)        2-layer stacked CfC            Voxel CfC (1D ring → 3D)
  vs 3 harder tasks             vs 1-layer, matched params     vs flat, matched params
                                on failing task from A         on failing task from A

Output: which tasks fail      Output: depth helps / hurts    Output: geometry helps / hurts
                                                              + hidden state visualization
```

## Key Decisions

**1. Diagnose before building** (from Nodes 1, 12 — resolved in Reflection)
We have no failure to fix. Phase A creates one. Phases B and C are conditional on Phase A finding a real gap.

**2. Three diagnostic tasks** (from Nodes 5, 6 — resolved in Reflection)
- Multi-frequency: sin(t) + sin(√2·t) + sin(π·t). Tests decoupled oscillator tracking.
- Copy-8-20: memorize 8 random ±1 tokens, wait 20 blank steps, reproduce. Tests information fidelity.
- Lorenz-x: predict x(t+dt) from [x(t), y(t), z(t)] in the Lorenz system. Tests chaos tolerance.
Each is a well-defined failure mode. Multi-frequency tests capacity. Copy tests fidelity. Lorenz tests sensitivity.

**3. Match active parameters, not neurons** (from Node 7 — resolved in Reflection)
All architecture comparisons use approximately equal active parameter counts (~2200 for the flat h=32 baseline). This separates "more parameters" from "better structure."

**4. Layers are probably a dead end, but test anyway** (from Node 8 — resolved in Reflection)
CfC unrolled T=50 is already 50 layers deep through time. Adding spatial depth compounds gradient path. The prior is negative. But a quick falsification experiment is cheap and closes the question.

**5. Voxels change what ternary weights MEAN** (from Node 9 — core insight)
On a grid, ternary sign patterns become local spatial operators. [+1, -1] between neighbors is a discrete gradient. This isn't just sparsity — it's a different computational primitive. The experiment tests whether this matters.

## Implementation Spec

### File: `examples/diagnostic_v3.c`

Phases A, B, and C in one file, run sequentially.

```
SECTION 1: Tasks (multi-freq, copy, lorenz)              (~120 LOC)
SECTION 2: Task-agnostic training loop (adapted from v2)  (~100 LOC)
SECTION 3: Phase A — flat h=32 ternary on all 3 tasks     (~60 LOC)
SECTION 4: Phase B — 2-layer stacked CfC                  (~120 LOC)
SECTION 5: Phase C — Voxel CfC (1D ring)                  (~150 LOC)
SECTION 6: Results table + analysis                        (~50 LOC)
                                                    TOTAL: ~600 LOC
```

### Task Definitions

```c
// Multi-frequency: 3 inputs (individual freqs), 3 outputs (next step of each)
// Or: 1 input (sum), 1 output (next sum). Simpler. Start with 1→1.
#define MF_FREQS  3
static float multi_freq_signal(float t) {
    return sinf(t) + sinf(1.41421356f * t) + sinf(3.14159265f * t);
}
// Task: predict signal(t + dt) from signal(t)
// INPUT_DIM=1, OUTPUT_DIM=1, SEQ_LEN=50

// Copy task: 8 random ±1 tokens, 20 blank steps, reproduce 8 tokens
// INPUT_DIM=1, OUTPUT_DIM=1
// Sequence: [t1, t2, ..., t8, 0, 0, ...(20 zeros)..., 0]  (length 28)
// Target:   [0,  0,  ..., 0,  0, 0, ...(20 zeros)..., t1, t2, ..., t8]
// Loss only on last 8 timesteps
#define COPY_N    8
#define COPY_WAIT 20

// Lorenz system: dx/dt = 10(y-x), dy/dt = x(28-z)-y, dz/dt = xy - 8z/3
// RK4 integration, dt=0.01
// Task: from [x,y,z] at time t, predict [x,y,z] at time t+dt
// INPUT_DIM=3, OUTPUT_DIM=3
#define LORENZ_SIGMA 10.0f
#define LORENZ_RHO   28.0f
#define LORENZ_BETA   2.6667f
```

### Phase B: Stacked CfC

```c
// 2-layer CfC: layer 1 (h1 neurons), layer 2 (h2 neurons)
// h1 + h2 chosen so total active params ≈ flat h=32 params
// Flat h=32: W_gate[32,33] + W_cand[32,33] = 2112, plus output ~2240
// 2-layer: W_gate1[16,17] + W_cand1[16,17] + W_gate2[16,17] + W_cand2[16,17]
//        = 2 * (2 * 16 * 17) = 1088, plus inter-layer + output ≈ 1300
// So h1=h2=20 → 2*(2*20*21) = 1680 + output ≈ 1900. Close enough.

typedef struct {
    Params layer1;
    Params layer2;  // layer2's "input" is layer1's hidden state
    // W_out projects from layer2 hidden state
} StackedParams;

static void forward_stacked_float(
    const float* x, 
    const float* h1_prev, const float* h2_prev,
    const StackedParams* p,
    int h1_dim, int h2_dim,
    StepCache* cache1, StepCache* cache2);
```

### Phase C: Voxel CfC (1D Ring)

```c
// 1D ring: N neurons arranged in a circle
// Each neuron connects to K nearest neighbors (K=2 → left+right)
// Plus input broadcast to all neurons
//
// The recurrent weight matrix is BANDED:
//   W_gate[i, :] has nonzero entries only at positions
//   (i-K/2, ..., i, ..., i+K/2) mod N   (the K+1 neighbors including self)
//   plus INPUT_DIM entries for the input
//
// Connectivity per neuron: INPUT_DIM + K + 1 (input + neighbors + self)
// For N=64, K=4: 64 * (1 + 4 + 1) = 384 recurrent params per matrix
// Two matrices (gate, cand): 768 active params + tau + output
//
// To match flat h=32 (~2200 params):
// N=64, K=8 → 64 * (1+8+1) = 640 * 2 = 1280 + 64*tau + output ≈ 1500
// N=64, K=16 → 64 * (1+16+1) = 1152 * 2 = 2304. Close match.

#define RING_N     64
#define RING_K     16   // neighbors on each side (total window = 2K+1 = 33)

typedef struct {
    // Dense weight arrays, but with a mask: non-neighbor entries forced to 0
    Params weights;
    uint8_t mask_gate[MAX_HIDDEN * MAX_CONCAT]; // 1 if connected, 0 if not
    uint8_t mask_cand[MAX_HIDDEN * MAX_CONCAT];
} VoxelParams;

// Forward: same as flat, but after quantization, apply mask
// Masked weights are always 0 (structural sparsity)
// Gradient: STE as usual, but gradient is zeroed for masked weights

static void init_ring_masks(VoxelParams* vp, int N, int K);
```

### Observability: Hidden State Snapshots

```c
// Every 100 epochs during voxel training, dump hidden state at T=25 (mid-sequence)
// Format: one float per neuron, written as a row in a CSV
// Post-training analysis: look for spatial correlation in the ring
// (Do neighboring neurons have similar activation? → spatial patterns exist)
// (Is activation uniform? → spatial structure unused)

static void dump_hidden_state(const float* h, int N, int epoch, FILE* f);
```

### Success Criteria for Each Phase

**Phase A (Diagnostic):**
- [ ] Float teacher MSE < 0.01 on multi-freq, copy, Lorenz
- [ ] Report flat h=32 ternary MSE on all three tasks
- [ ] Identify which task(s) show >10x degradation (= the wall)
- [ ] If no task fails: report that flat ternary is sufficient, no architectural change needed (valid finding)

**Phase B (Depth):**
- [ ] 2-layer CfC at matched params vs 1-layer on the failing task
- [ ] Report whether depth helps, hurts, or is neutral
- [ ] If depth hurts: explain via gradient path length argument (Node 8)

**Phase C (Geometry):**
- [ ] Ring CfC at matched params vs flat on the failing task
- [ ] Report whether structured sparsity helps, hurts, or is neutral
- [ ] Dump hidden state snapshots
- [ ] Report whether spatial patterns emerge in the ring activation
- [ ] Concrete prediction to falsify: "Ring CfC with N=64, K=16 will outperform flat h=32 on [failing task] at matched parameter count because local connectivity creates decoupled processing regions"

### Temperature / Schedule

Same as v2: Adam optimizer, LR=0.001, 2000 teacher epochs, 1500 student epochs.
No smoothstep (v2 showed it doesn't help). No trajectory distillation (v2 showed it hurts).
Just STE + Adam + per-row scales. The architecture is the variable.

## What Emergence Looks Like

If the synthesis is right:

1. **Phase A reveals copy task as the wall.** Multi-frequency might also stress ternary, but copy is the purest test of fidelity through recurrence. Lorenz might be too sensitive (float teacher may also struggle).

2. **Phase B shows depth is neutral or negative.** The temporal depth already provides enough representational capacity. Spatial depth adds gradient path length without compensating benefit.

3. **Phase C shows ring CfC matches or beats flat.** The local connectivity creates implicit processing regions. Neurons near each other share context; distant neurons are decoupled. For copy, this might allow spatial separation of memory slots (different regions of the ring store different tokens).

4. **Hidden state snapshots show spatial structure.** Neighboring neurons in the ring should have more correlated activations than distant neurons. If this correlation is transient (strong during training, weaker at convergence), we've found the Delta Observer phenomenon in ternary CfC.

If we see something different, that's MORE interesting. The point is to measure.

## What If No Task Breaks?

If flat h=32 ternary handles multi-freq, copy, AND Lorenz with <2x degradation, that's a strong finding:

> "Flat ternary CfC with STE training and per-row scales is sufficient for temporal prediction tasks at h=32. No architectural modification (depth, geometry) is needed."

This would mean ternary CfC's capacity scales with width alone, and the architecture is already well-matched to the weight representation. We document this, archive it, and look for tasks that are genuinely harder (sequence-to-sequence, language modeling, multi-agent dynamics).
