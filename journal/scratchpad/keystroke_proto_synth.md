# Synthesis: Keystroke Biometric Prototype v2

The wood cuts itself.

---

## What the LMM Revealed

The first prototype was structurally correct but operationally wrong. The probes exposed three critical issues:

1. **Welford online normalization destroys CfC hidden state consistency** (intra-class cosine drops from 0.90 to 0.00). Must be removed from the CfC input path.

2. **The Mahalanobis distance penalty is conceptually wrong** for untrained weights. It penalizes auth-phase hidden states for differing from enrollment statistics, but different random sequences from the same distribution produce different hidden states. The penalty hurts the enrolled user as much as the impostor.

3. **The original simulation's "separation" was an artifact** of seed correlation. When tested with independent seeds, the scoring mechanism shows no separation or inverted separation.

**What survived:** The CfC cell produces consistent, information-carrying hidden states (0.90 intra-cosine without norm). The pipeline executes in 251 ns. The architecture is sound. The chips cooperate correctly.

---

## Architecture Decision: Two-Mode Demo

The prototype should have two honest modes:

### Mode 1: Template Matching (no training required)

- Remove Welford normalization from CfC input
- Feed raw (key_code, dt) directly to CfC
- During enrollment: run CfC, store the final hidden state as the template
- During auth: run CfC, compute cosine similarity between live hidden state and template
- Score = cosine similarity (naturally in [-1, 1])

**What this can demonstrate:**
- The pipeline works end-to-end
- The CfC produces consistent dynamics (0.90 intra-class cosine)
- Users with dramatically different typing speeds produce measurably different hidden states
- Execution is 251 ns

**What this CANNOT demonstrate:**
- Fine-grained discrimination (same speed, different rhythm)
- Real biometric authentication accuracy

**Honest framing:** "This demo shows the CfC chip producing consistent temporal representations of typing patterns. With trained weights, these representations become discriminative. This prototype uses untrained weights, so discrimination is limited to gross typing speed differences."

### Mode 2: Trained Weights (future — requires Python trainer)

- Replace hand-initialized weights with Python-trained ternary weights
- Use the full pipeline: norm -> CfC -> projection -> sigmoid
- Normalization uses fixed stats from training distribution (not online Welford)
- This is the production path

**Not built yet.** Requires a Python CfC trainer + ternary quantizer + weight export.

---

## Concrete Changes to the Prototype

### Remove from `keystroke_biometric.c`:
1. Online Welford normalization on CfC input (keep drift detector, move it to a monitoring role only)
2. Mahalanobis distance penalty in scoring
3. Misleading "MATCH/REJECT" thresholds that suggest real authentication

### Add to `keystroke_biometric.c`:
1. Template-based scoring: cosine similarity between live h_state and enrolled h_template
2. Multi-template enrollment: store N hidden state snapshots during enrollment, compare against all, take max cosine
3. Warmup discard: skip first 5 auth keystrokes (cold start transient)
4. Honest output: show cosine similarity as a raw number, show "same user" and "different user" ranges, don't claim authentication accuracy

### Keep:
1. Interactive mode (termios raw input, real timing)
2. Simulation mode (but fix seed independence)
3. Drift detection (but as a monitoring signal, not part of the score)
4. All chip includes (CfC, activation, GEMM — GEMM still needed for CfC internals)
5. The 251 ns execution time measurement

### Pipeline change:
```
BEFORE: keystroke -> raw -> welford_norm -> CfC -> MATVEC_proj -> distance_penalty -> sigmoid -> score
AFTER:  keystroke -> raw -> CfC -> cosine_sim(h_state, h_template) -> score
```

The new pipeline is simpler, more honest, and actually works with untrained weights.

### Simulation redesign:
- Use independent seeds for enrollment and auth
- Run multiple auth sequences per user (10+) and report distribution
- Include a baseline comparison: raw dt mean comparison
- Show the CfC adds value by being more consistent than raw dt alone
- Test at multiple speed differences: 1.1x, 1.5x, 2x, 3x

---

## What the v2 Demo Proves

With untrained weights, the v2 demo proves:

1. **The CfC chip produces consistent temporal representations.** Same user, different runs -> cosine ~0.90. This is the core capability.

2. **Different input dynamics produce different representations.** Fast vs slow typist hidden states differ (cosine 0.85 vs 0.90 intra). The CfC encodes timing information.

3. **The pipeline is fast.** 251 ns for the full stack. That's ~4 million keystrokes/second.

4. **The pipeline is small.** 4 chips, <3 KB code, 32 bytes hidden state.

5. **Training is the bottleneck, not architecture.** The representation quality justifies investing in a Python trainer.

---

## What v2 Does NOT Prove (Honest Boundaries)

- Does NOT prove fine-grained biometric discrimination
- Does NOT prove the system works for real users with real keyboards
- Does NOT prove CfC outperforms a simple statistical baseline for coarse speed differences
- Does NOT prove trained ternary weights will achieve useful accuracy

These are future validation steps, not things to claim in the demo.

---

## Norm Chip: Updated Guidance

The Welford normalization in `norm_chip.h` is not broken — it's misapplied. Online normalization is correct for streaming sensor data where the input distribution is stationary within a session (e.g., accelerometer with fixed mounting). It's wrong for keystroke biometrics where the "distribution" IS the signal you're trying to preserve.

**Updated rule:** Use `ONLINE_NORMALIZE_CHIP` when input distribution should be treated as nuisance (sensor drift, calibration variation). Do NOT use it when input distribution IS the discriminative signal (keystroke timing, CAN bus timing, any behavioral biometric).

Drift detection (`RUNNING_STATS_UPDATE` + `RUNNING_STATS_VARIANCE` for monitoring) remains valid and useful — just don't feed the normalized output into the CfC.

---

## Success Criteria for v2

- [ ] CfC hidden states for same-user runs have cosine > 0.85
- [ ] CfC hidden states for different-user (2x speed) runs have cosine < 0.85
- [ ] No Welford normalization in CfC input path
- [ ] Independent random seeds in simulation
- [ ] Simulation reports raw cosine similarity, not fake auth scores
- [ ] Execution time printed and < 300 ns
- [ ] Interactive mode works with template-based scoring
- [ ] Demo text is honest about what untrained weights can and cannot do
