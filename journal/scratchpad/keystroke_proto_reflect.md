# Reflections: Keystroke Biometric Prototype

The axe was dull. The probes showed it. Now sharpening.

---

## Core Insight

**The Welford normalization is destroying the CfC's ability to form consistent representations.**

This is the single most important finding. Look at the data:

| Test | Norm | Intra-class cosine (A vs A) | Inter-class cosine (A vs B) |
|------|------|-----------------------------|-----------------------------|
| Test 1 (3x speed diff) | YES | **0.0045** | 0.0307 |
| Test 2 (3x speed diff) | NO  | **0.9020** | 0.8546 |
| Test 5 (control, same) | YES | **-0.0025** | 0.0321 |

**With normalization:** Intra-class cosine is ~0.00. User A's hidden state after 80 keystrokes points in a RANDOM direction each run. The CfC produces no consistent representation. Runs of the same user from different seeds are orthogonal.

**Without normalization:** Intra-class cosine is ~0.90. User A's hidden state points in roughly the same direction regardless of seed. The CfC produces a CONSISTENT representation of the input dynamics. This is what we want.

**Why normalization destroys consistency:** The Welford running mean/variance depends on the exact sequence of inputs. Two runs from the same distribution but with different random draws produce different normalization statistics. By the time the CfC sees the 80th keystroke, the normalized input has been scaled by a sequence-dependent normalizer. The CfC can't form a stable attractor because its input keeps shifting underneath it.

**Without normalization:** The CfC sees raw (key_code, dt) directly. The statistical properties of the input (mean dt, variance of dt) get baked directly into the hidden state trajectory. Different runs from the same distribution converge to similar hidden states because the CfC dynamics are driven by the same raw signal statistics.

---

## Resolved Questions

### Q: Does the CfC separate different users in hidden state space?

**With norm: NO.** Intra-class similarity is ~0, meaning the CfC doesn't even produce consistent states for the SAME user, let alone separable states for different users.

**Without norm, easy case (3x speed): WEAKLY YES.** Intra=0.90, Inter=0.85. The gap is small (0.047) but consistent. The hidden state vectors for fast typists and slow typists point in slightly different directions. The euclidean distance ratio is 1.18x — inter-class distances are 18% larger than intra-class distances.

**Without norm, hard case (same speed, diff jitter): NO.** Intra=0.90, Inter=0.90. No separation. The jitter difference alone doesn't produce distinguishable hidden states with untrained weights.

### Q: Can template matching work without training?

**For the easy case:** Barely. A 0.047 cosine gap and 1.18x euclidean ratio is detectable but fragile. A decision threshold would have low accuracy (lots of overlap).

**For the hard case:** No. The hidden states are indistinguishable.

**Bottom line:** Template matching in hidden state space can weakly separate users who differ dramatically in typing speed. It cannot separate users who differ in subtler ways (rhythm, jitter patterns). For that, we need training.

### Q: Is the architecture fundamentally sound?

**Yes.** The CfC produces consistent hidden states (0.90 intra-class cosine without norm) that carry information about input dynamics. The hidden state IS a representation of the typing pattern. It's just not a DISCRIMINATIVE representation yet — that requires training the weights to amplify between-user differences.

Look at the hidden state samples (Test 7):
```
User A: [0.09, 0.07, 0.04, 0.04, -0.01, 0.02, 0.00, 0.01]
User B: [0.19, 0.17, 0.11, 0.09, -0.00, 0.06, 0.00, 0.03]
```

User B (slow typist) has larger hidden state values. This makes physical sense: larger dt means more time for the CfC to integrate, and the slow tau neurons (tau=0.50, 0.80) accumulate more. The CfC IS encoding timing information. It's just encoding ALL timing information equally, not selectively amplifying discriminative features.

### Q: What does the 251ns execution time include?

The full pipeline: normalize + CfC + MATVEC + sigmoid + distance. This is the real number. Even after we fix the normalization and scoring, the CfC cell execution is the dominant cost and it won't change.

---

## What the Probes Falsified

1. **Original simulation's "SEPARATION DETECTED"**: Artifact of correlated seeds. Falsified by Probe 1 Experiment 2.

2. **"Welford normalization is standard infrastructure"**: It destroys hidden state consistency. Must be removed or replaced for this application.

3. **"Distance penalty helps authentication"**: It hurts. The penalty computes distance from enrollment statistics, but the hidden state depends on the exact sequence, not just the distribution. Probe 1 Experiment 5 showed increasing the penalty worsens separation.

4. **"Template matching can work without training"**: Only for trivial cases (3x speed difference). Falsified for subtle discrimination by Probe 2 Tests 3-4.

---

## What Survived

1. **The CfC cell produces consistent dynamics.** 0.90 intra-class cosine without normalization. The architecture works.

2. **251 ns full pipeline.** Real, measured, honest.

3. **Different input statistics produce different hidden states.** Fast typists and slow typists have distinguishable (if overlapping) hidden state distributions.

4. **The chip pipeline compiles, runs, and cooperates correctly.** Four chips in series, zero bugs.

5. **Drift detection works.** Correctly flags distribution shift.

---

## The Honest State of the Prototype

**What works:** Pipeline, chips, execution time, CfC dynamics, drift detection.

**What doesn't work:** Authentication. The demo cannot separate users reliably without trained weights. The scoring mechanism (projection + distance penalty) is miscalibrated and conceptually wrong for untrained weights.

**What needs to change:**
1. Remove Welford normalization from the CfC input path (or replace with fixed normalization from enrollment statistics)
2. Remove the distance penalty mechanism
3. For an untrained demo: use raw hidden state cosine similarity to a stored enrollment template (will work only for the easy case)
4. For a real demo: train weights in Python and export

---

## Remaining Questions

- If we use fixed normalization (enrollment mean/std applied at auth time, not online), do we recover hidden state consistency?
- What is the minimum speed difference for reliable separation with untrained weights?
- Could we use a simple linear probe (train only W_out on enrolled data) instead of full training?
