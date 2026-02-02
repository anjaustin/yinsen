# Raw Thoughts: Linear Discriminant on CfC Hidden States

## Stream of Consciousness

The results look great. That's what scares me. Last time the results looked great (v1 simulation) they were a seed artifact. I need to be more careful this time.

100% separation on the easy case. 100% on the medium case. 75% on the hard case. 60% on the control. The control is correct — 60% is coin-flip-plus-noise, which is what you'd expect when A and B are drawn from the same distribution. The hard case (75%) is above control (60%). That's real signal. But 75% is not great — it means 1 in 4 times, the impostor scores higher than the enrolled user.

Wait. Let me look at what the linear discriminant is actually doing. During enrollment, we collect 80 hidden state vectors from the CfC. We compute their mean and top 3 principal components. At auth time, we center the live hidden state by subtracting the enrollment mean, project onto the PCs, and measure how many standard deviations the projection is from the enrollment mean projection.

What is this actually measuring? It's measuring whether the auth hidden state lives in the same region of hidden space as the enrollment hidden states. The mean captures "where is the center." The PCs capture "what directions does the enrolled user vary in." The z-score measures "are you in the expected subspace."

But here's a problem I didn't think about: the PCs capture INTRA-user variation, not INTER-user differences. The directions that User A varies along might be the SAME directions that User B varies along. The PCs describe the shape of the cloud, not what distinguishes it from other clouds. If User A and User B both vary primarily along the same axis (say, the hidden dimension most sensitive to dt), the PCA subspace is the same for both, and the discriminant can only rely on the MEAN difference.

That would explain the results perfectly:
- Easy (3x speed): Mean differs a lot -> 100% separation
- Medium (1.5x speed): Mean differs moderately -> 100% separation
- Hard (same speed, diff jitter): Mean is similar, but variance structure differs slightly -> 75%
- Control (same everything): Mean and variance identical -> 60% (noise)

So the PCA isn't really adding much beyond the mean. The mean alone might do most of the work. I should test mean-only scoring to see if PCA is adding signal or just complexity.

Another concern: the enrollment seeds in Tests 4 and 5 are the same (both use the same `100 + r * 31` seeds for enrollment). The auth seeds are also the same. So Tests 4 and 5 should produce IDENTICAL results. They do — both show avg 0.705/0.697 with A wins 12/20. That's good for consistency but it means Test 5 is not an independent control — it's literally the same test as Test 4 with a different label. I should have used different enrollment seeds for the control. Sloppy.

The discriminant is 156 bytes. That's: mean (32) + 3 PCs (96) + 3 means (12) + 3 stds (12) + valid flag (4). Tiny. On an MCU this fits in registers.

The enrollment cost: power iteration is 3 PCs * 20 iterations * (80 samples * 8-dim dot + 80 * 8 accumulate) = roughly 3 * 20 * 80 * 16 = 76,800 FLOPs. At 138ns per CfC step (which involves ~200 FLOPs), the power iteration is equivalent to about 380 CfC steps. Under 1ms on this machine. Negligible.

Something bugs me about the warmup. I skip the first 10 keystrokes for both enrollment and auth. The enrollment sees keystrokes 11-90. The auth sees keystrokes 11-60 (10 warmup + 50 scored). But the warmup states are from a cold start (h=0). After 10 steps the CfC has warmed up to a trajectory that depends on the input. Is 10 enough? The CfC's slowest tau is 0.80. At mean dt=0.15s, 10 keystrokes = 1.5s. The slowest decay exp(-1.5/0.8) = exp(-1.875) = 0.15, meaning 85% of the initial state has decayed. That seems sufficient. But the faster taus (0.05) decay in 2-3 steps. So after 10 steps, all neurons are fully driven by the input. Warmup of 10 seems fine.

One thing I didn't consider: the order of samples matters for PCA. The CfC hidden state is a TIME SERIES — consecutive samples are correlated because the hidden state evolves smoothly. PCA on time-correlated samples doesn't give you the same thing as PCA on i.i.d. samples. The first PC might just capture "the hidden state drifts in this direction over time" rather than "the hidden state varies in this direction across typing sessions." Since I'm collecting ALL 80 samples from a single enrollment session, the PCA is dominated by the within-session trajectory, not between-session variation.

This might actually be useful — the within-session trajectory IS characteristic of the user. But it also means the discriminant is sensitive to the enrollment session length and starting conditions in ways I haven't tested.

What if enrollment consisted of multiple short sessions instead of one long one? E.g., 4 sessions of 20 keystrokes, each starting from h=0. The PCA would then capture between-session variation (what stays consistent across sessions) rather than within-session drift. That might be more robust.

The sigmoid mapping: `score = sigmoid(2.0 - dist)`. When dist=0 (perfect match), score=sigmoid(2)=0.88. When dist=2, score=sigmoid(0)=0.5. When dist=4, score=sigmoid(-2)=0.12. This is aggressive — 2 standard deviations from the mean already drops the score to 50%. In the easy case, User B's distance is large enough to push scores near 0. In the hard case, the distances are close enough that the sigmoid can't separate well. The sigmoid parameters (center=2.0, scale=1.0) were hand-picked. Different values might change the separation quality.

Actually, what matters isn't the absolute scores — it's the separation. A wins 100% in the easy case regardless of sigmoid tuning, because the underlying distance difference is large. The sigmoid only affects the score magnitudes, not the ranking.

For the hard case (75% A wins), could we do better with more PCs? I used 3. What about 5 or 7 (approaching full dimensionality of 8)? More PCs capture more of the enrolled variance, giving a tighter description of the enrolled subspace. But they also increase overfitting risk — with 80 samples in 8 dimensions, you can describe the data very tightly, and then any new sample (even from the same user) will look like an outlier.

What about the opposite: just 1 PC? If the mean does most of the work, 1 PC might be sufficient and more robust.

## Questions Arising
- How much does the mean alone contribute? Test mean-only scoring.
- How much does each additional PC add? Sweep N_PCS from 1 to 7.
- Is within-session PCA the right choice, or should enrollment use multiple sessions?
- Does the sigmoid mapping matter, or only the rank order?
- What happens with more enrollment keystrokes? 160? 320?
- What happens when the same user comes back after a break (new session, different text)?
- Is there a way to use the PCs for INTER-class separation rather than just INTRA-class description?

## First Instincts
- The mean is doing most of the heavy lifting. PCA might be marginal.
- The 75% on the hard case is real but fragile. It won't hold up in production.
- The 100% on the easy and medium cases is solid and defensible.
- Multi-session enrollment would be more robust than single-session.
- The approach is sound for the demo. Don't oversell the hard case.
- 156 bytes is a beautiful number. Keep it small.
