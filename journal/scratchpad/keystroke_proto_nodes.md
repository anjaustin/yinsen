# Nodes of Interest: Keystroke Biometric Prototype

Extracted from Phase 1 RAW + probe experiment data. The probe falsified the original simulation's apparent success.

---

## Node 1: The Original Simulation Lied (by omission)

The original `--sim` showed separation: User A avg 0.377, User B avg 0.031. The probe, using fresh random seeds for the auth phase, showed the OPPOSITE: User B scores HIGHER than User A in experiments 2-4. Separation is **negative**.

**What happened:** The original simulation used seeds that were continuations of the enrollment seeds. The auth-phase User A seed was a continuation of the enrollment seed, so the random sequence was correlated. When we use independent seeds (as the probe does), the apparent separation vanishes.

**This is a falsification.** The original demo's "SEPARATION DETECTED" was an artifact of seed correlation, not genuine discrimination. The pipeline does NOT discriminate between users with untrained weights.

This is the most important finding. Everything downstream must account for it.

---

## Node 2: The Distance Penalty is Inverted

In the probe's Experiment 5 (distance coefficient sweep), increasing the distance penalty makes things WORSE, not better. At coeff=0.0, User A and User B are tied at ~0.507. As coeff increases, User B scores HIGHER than User A, and the gap grows.

**Why:** The distance penalty compares the auth-phase hidden state to the enrolled mean. But the enrolled mean was computed from a DIFFERENT random sequence than the auth-phase User A. The hidden state depends on the specific input sequence, not just its statistics. Two sequences drawn from the same distribution produce different hidden state trajectories. The distance penalty punishes all auth-phase users roughly equally — it doesn't help the enrolled user.

**The distance mechanism is wrong.** It assumes that matching the statistical distribution of inputs will produce similar hidden states. With untrained weights, that assumption is false.

---

## Node 3: Execution Time is Real — 251 ns

The probe measured 250.7 ns per full pipeline call (normalize + CfC + MATVEC + sigmoid + distance). That's real. On this Apple Silicon machine, that's the full cost of:
- 2D Welford update
- 2D normalization
- CfC cell (2 GEMMs at 8x10, 2 activation passes, decay, mix)
- 1x8 MATVEC projection
- Sigmoid
- 8-element Mahalanobis distance

251 ns for 4 chips cooperating. This number is honest and defensible.

---

## Node 4: The Projection Layer is Useless Without Training

At distance_coeff=0.0 (Experiment 5), both users score ~0.507. The MATVEC + sigmoid projection outputs approximately the same value regardless of input. This is because the output weights are hand-initialized, not trained. The projection has no learned concept of "this hidden state pattern means enrolled user."

**This is expected.** The projection is a linear readout. Without training, it reads noise. But it means the demo cannot demonstrate authentication with hand-initialized weights. Period.

---

## Node 5: The Architecture is Sound; The Weights are Empty

The CfC cell IS producing different hidden states for different users. We can see this because the enrolled hidden state mean is non-zero and structured: `[-0.007, 0.033, 0.068, 0.070, 0.007, 0.014, 0.006, 0.008]`. The CfC responds to the input dynamics. But the response is not yet MEANINGFUL for discrimination because nothing has told the network what to discriminate.

The architecture (norm -> CfC -> projection -> score) is correct. The chip pipeline works. The execution time is 251 ns. What's missing is learned weights that map the CfC dynamics to a discriminative output.

---

## Node 6: The Simulation Design is Flawed

Two problems with the simulation:

1. **Seed correlation:** The original simulation didn't use independent seeds for enrollment vs auth, creating spurious correlation.

2. **Only speed varies:** User A (0.12s) vs User B (0.35s) differ by 3x in typing speed. A threshold on raw dt would separate them. The simulation doesn't test what makes CfC interesting — detecting rhythmic PATTERNS, not average speed.

A proper simulation needs:
- Independent random seeds for enrollment and auth
- Users that differ in rhythm, not just speed
- A baseline comparison (e.g., raw dt threshold) to show CfC adds value

---

## Node 7: What the Demo CAN Show vs What It CLAIMS

**Can show honestly today (untrained weights):**
- The pipeline compiles and runs
- All 4 chips cooperate correctly
- Execution is 251 ns per keystroke
- The CfC produces different dynamics for different input patterns
- Drift detection works

**Cannot show today (needs trained weights):**
- Actual authentication (enrolled user > threshold > impostor)
- That CfC adds value over raw dt statistics
- That temporal patterns are being learned

The demo should be restructured to claim only what it can demonstrate.

---

## Node 8: The Path to Real Discrimination

For the demo to actually authenticate, we need one of:

**Option A: Train in Python, export weights.** This is the plan from the synthesis doc. Train a CfC on real or synthetic keystroke data, quantize to ternary, export as C arrays. This would definitely work but requires building a training pipeline.

**Option B: On-device enrollment via template matching.** Skip the learned projection entirely. During enrollment, record the hidden state trajectory. During auth, compute cosine similarity between the live hidden state trajectory and the enrolled trajectory. This is a template matcher, not a learned classifier. It might work for a demo without training.

**Option C: Hand-tune weights for a specific discrimination.** If we understand the CfC dynamics well enough, we could hand-craft weights that map dt-patterns to distinguishable hidden states. This is fragile but could work for a controlled demo.

---

## Node 9: Welford Normalization is Both Helping and Confounding

The online normalization adapts the input scaling as more keystrokes arrive. During enrollment, the stats stabilize. During auth, if the user is the same, the stats continue smoothly. If the user is different, the stats drift.

**The confound:** The normalization itself is user-dependent. After 80 enrollment keys, the normalization is tuned to User A's distribution. When User B arrives with a different dt distribution, the normalization produces different scaled values not just because of B's different rhythm, but because of the distribution mismatch against A's accumulated stats.

This means the normalization is doing some of the discrimination work — but in a way that's fragile and depends on sequence length.

---

## Connections Map

```
[1: Sim Lied] --> [6: Sim Design Flawed]
     |
     v
[2: Distance Inverted] --> [4: Projection Useless]
     |                          |
     v                          v
[5: Architecture Sound] --> [8: Path to Real]
     |
     v
[3: 251ns Real] --> [7: Can vs Claims]
     |
     v
[9: Normalization Confound]
```

## Boundary Cases (Deltas)

1. **Node 5 vs Node 4:** The architecture works but the weights don't. This is a delta — is it "works" or "doesn't work"? Answer: the PIPELINE works, the DEMO doesn't demonstrate its intended purpose. Be honest about the distinction.

2. **Node 8 Option B:** Template matching in hidden space might work WITHOUT training. This is worth testing before building a full Python training pipeline. If it works, it dramatically simplifies the demo.

3. **Node 9:** Is Welford normalization part of the solution or part of the problem? The answer might be "both" — it helps with input scaling but confounds the discrimination signal. We need to test with and without it.
