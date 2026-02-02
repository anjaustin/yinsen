# Nodes of Interest: Linear Discriminant on CfC Hidden States

## Data Sources
- probe3: Original linear discriminant (3 PCs), 20 trials per test
- probe4: Ablation — mean-only vs PCA(1,2,3,5,7), 20 trials per test, 4 difficulty levels

---

## Node 1: The Mean Does the Heavy Lifting
Mean-only scoring: Easy 20/20, Medium 19/20, Hard 14/20, Control 11/20.
This is the strongest single method. The enrollment mean captures "where in hidden space does this user live?" and that alone separates users when their typing speeds differ.
Why it matters: The simplest possible discriminant (32 bytes — just the mean vector) is already competitive with the full PCA approach.

## Node 2: PCA(1) and PCA(2) Are Worse Than Mean-Only
PCA(1): Easy 11/20, Medium 10/20, Hard 11/20, Control 11/20. That's noise floor across the board.
PCA(2): Easy 13/20, Medium 12/20, Hard 13/20, Control 13/20. Barely above noise.
Why it matters: Low PC counts actively hurt. Projecting onto 1-2 PCs discards the mean information (the scoring uses z-score in PC space, not Euclidean distance from the mean). The PCA scoring path REPLACES the mean comparison, it doesn't supplement it.

## Node 3: PCA(5) Is the Best Overall Configuration
PCA(5): Easy 20/20, Medium 20/20, Hard 17/20, Control 11/20.
This beats both mean-only and PCA(3) on every test. The hard case jumps from 14/20 (mean) to 17/20 (PCA5). The control stays at 11/20 (noise), which is correct — it shouldn't separate identical distributions.
Why it matters: PCA DOES add value beyond the mean, but only at sufficient dimensionality. 5 PCs out of 8 dimensions captures enough of the enrolled subspace to tighten the match.

## Node 4: PCA(7) Shows Diminishing Returns
PCA(7): Easy 20/20, Medium 20/20, Hard 14/20, Control 12/20.
The hard case drops from 17/20 (PCA5) back to 14/20. With 7 PCs in 8 dimensions, the subspace is almost full-rank — it describes everything, so it rejects nothing. The discriminant becomes too permissive.
Why it matters: There's a sweet spot. Too few PCs lose the mean signal. Too many lose the selectivity. 5 is the sweet spot for 8-dimensional hidden state.

## Node 5: The Scoring Architecture Is Wrong
The current PCA scoring computes z-scores in PC space: project onto each PC, compare to enrolled mean projection, normalize by enrolled std. This means the score is entirely about variance-normalized distance in the PC subspace. It ignores the Euclidean distance of the mean.
Tension with Node 1: The mean carries the strongest signal, but PCA scoring doesn't use it explicitly. PCA(5) works well because with 5 of 8 dimensions, the mean difference is implicitly captured in the subspace. But this is accidental — the mean should be an explicit feature.

## Node 6: A Hybrid Score Would Be Optimal
Combine mean distance (Node 1) with PCA subspace distance (Node 3). Two complementary signals:
- Mean distance: "Is this user in the right neighborhood?"
- PCA distance: "Is this user's trajectory shape consistent?"
Why it matters: Mean-only gets 14/20 on hard. PCA(5) gets 17/20. A weighted combination could push higher because they measure different things.

## Node 7: The Control Test Is the Integrity Check
Every method scores 10-13/20 on the control (same distribution for A and B). This is correct — you can't separate identical distributions. Any method that shows >15/20 on the control would be an artifact.
Mean-only: 11/20. PCA(3): 10/20. PCA(5): 11/20. All healthy.
Why it matters: The control validates that separation on other tests is real, not structural bias.

## Node 8: The Discriminant Size Varies With PC Count
- Mean-only: 32 bytes (just the mean vector)
- PCA(3): 156 bytes (mean + 3 PCs + 3 means + 3 stds + flag)
- PCA(5): 244 bytes (mean + 5 PCs + 5 means + 5 stds + flag)
All fit comfortably in MCU registers. The size difference is irrelevant for the target platform.
Why it matters: There's no resource pressure to minimize PC count. Choose based on accuracy, not size.

## Node 9: The Hard Case Is the Real Benchmark
Easy and medium are solved by mean-only. The interesting question is whether we can push the hard case (same speed, different jitter) above 75-85%. This is where PCA's subspace matching should shine — jitter affects the shape of the hidden state trajectory, not just its center.
Why it matters: If the demo only works when users type at very different speeds, it's a speed classifier, not a biometric. The hard case tests whether the CfC captures individual typing rhythm beyond speed.

## Node 10: PCA Captures Within-Session Trajectory
As noted in the RAW phase: the 80 enrollment samples are consecutive hidden states from a single session. PCA on time-correlated data captures the trajectory shape, not session-to-session variation. This is actually a feature for this application — the trajectory shape IS the biometric signature.
Tension with Node 9: But the trajectory shape might be dominated by typing speed. If the CfC trajectory is a spiral whose rate is proportional to dt, then "trajectory shape" and "speed" are the same thing, and the hard case can't improve.

## Node 11: The RAW Phase's Hypothesis Was Confirmed
RAW predicted: "The mean is doing most of the heavy lifting. PCA might be marginal." Probe4 confirms the mean is strong but PCA(5) adds real value on the hard case (+3 wins out of 20). The RAW phase also predicted PCA would capture within-session drift, not between-session variation — and this is consistent with PCA(1-2) being useless (they capture only the dominant drift axis, which is the same for all users).
