# Raw Thoughts: Improving ISS Telemetry with Real WebSocket Data

## Stream of Consciousness

The falsification probes gutted the demo. Let me be honest about what's left standing and what's rubble.

What's standing: the pipeline works. 8 independent CfC channels, discriminant per channel, cross-channel correlation, stdin streaming protocol, Python Lightstreamer shim. 3,328 bytes total. 55 ns/channel. The plumbing is real and it's clean.

What's rubble: the claim that the CfC provides meaningful temporal anomaly detection beyond what a threshold detector would do. The probes showed no advantage over 3-sigma on any tested anomaly pattern. The tau tuning narrative is vacuous. CMG and CabinP hidden states are degenerate — H-Std of 0.0007 and 0.0003 respectively. The CfC can't exercise its nonlinearity on inputs that live in a 0.001-wide range when its sigmoid/tanh saturate around [-3, 3].

But wait. The probes tested SYNTHETIC data. The whole point of the WebSocket shim is to get REAL ISS telemetry. Real data has structure that synthetic data doesn't. Orbital thermal cycling isn't a clean sinusoid — it has eclipses, attitude changes, experiment activations, crew activities. Real CMG data has bearing harmonics, not Gaussian noise on a DC offset. Real cabin pressure has airlock cycles, Progress docking events, CDRA pump-down transients.

The question isn't "does the CfC detect anomalies in our toy sine generator." The question is "does the CfC hidden state develop richer dynamics on real ISS telemetry than on synthetic, and if so, does that richness translate to detection of real operational events?"

Here's what scares me: it might not. The CfC has UNTRAINED weights. The gate and candidate matrices were hand-initialized with structurally meaningful but ultimately arbitrary values. The CfC architecture (gate + candidate + exponential decay) creates temporal dynamics, but those dynamics are only useful if the weights create a meaningful mapping from input space to hidden space. With random-ish weights, the CfC might just be a fancy lowpass filter.

But the keystroke probes showed the CfC DOES produce consistent representations without training — 0.897 intra-class cosine. The architecture itself provides structure. The question is whether that architecture-level structure is enough for ISS telemetry or whether the input scale problem kills it.

The input scale problem is the real issue. CMG vibration ~0.001g. After passing through `W_gate @ [x, h]`, the input contribution is `0.1 * 0.001 = 0.0001`. That's essentially zero compared to the bias term of -0.5. The gate is completely dominated by the bias and the hidden state recurrence. The input barely affects anything. This is why the hidden state is trapped at ~0.5014 — it's the fixed point determined by the bias, and the input perturbation is too weak to shift it.

The fix is obvious: pre-scale the input. If CMG goes from 0.001 to 0.002, that's a range of 0.001. Scale it to [0, 1] by subtracting the min and dividing by the range. Then the CfC sees a full-range input and its nonlinearities actually engage.

But wait — we learned from keystroke v1 that normalization destroys CfC hidden state consistency. The cosine dropped from 0.90 to 0.00 when Welford was applied. So can we normalize at all?

The keystroke lesson was specifically about ONLINE normalization during execution — the running mean/variance shifted as new data arrived, which caused the CfC to track the normalizer state rather than the signal. The fix isn't "never normalize" — it's "normalize with FIXED statistics from enrollment." Learn the min/max or mean/std during enrollment, freeze them, then apply the same fixed transform during execution. The normalization becomes a constant, not a moving target.

This is what the Lightstreamer data gives us that synthetic can't: real signal statistics for calibration. We can enroll on real ISS data, learn the per-channel ranges, and use those frozen ranges as the input scaling. Then the CfC hidden states should actually move.

Another thought: the Lightstreamer feed pushes updates at irregular intervals. Some parameters update every second, some every 10 seconds, some only when they change. This means dt varies per channel and per update. The CfC was designed for variable dt — that's the whole point of the exponential decay `exp(-dt/tau)`. On synthetic data, dt was constant (10s), so this capability was untested. Real ISS data with variable dt would be the first real test of the CfC's temporal dynamics.

The ISS Mimic project has a telemetry database. I should look at what parameters are actually streamed and what their update rates are. The CMG parameters might update faster than the thermal parameters. If CMG updates at 1Hz and coolant updates at 0.1Hz, the per-channel dt values span a 10x range. That's exactly the kind of multi-timescale scenario the CfC architecture was built for.

What about the spike invisibility problem? Single-sample spikes scored identically to normal because we score at the end of the sequence. The fix: score EVERY sample, not just the final hidden state. Keep a running score and flag when it drops below threshold. The current stdin mode already does this (scores on every incoming sample). The simulation was broken because it only scored at sequence end.

Actually, looking at the simulation code more carefully — Phase 3 does score at every step and tracks detection time. The spike test in probe2 was poorly designed: it injected a spike at 3/4 mark but only looked at the FINAL score. Of course the final score recovered — the spike was 75 samples ago. The right test would check the score AT the spike. The spike IS visible in the hidden state for a few time constants after injection — we just didn't look at the right moment.

## Questions Arising

- What ISS telemetry parameters actually stream via Lightstreamer and at what rate?
- What are the real value ranges for CMG, coolant, cabin, O2 parameters?
- How much does fixed pre-scaling (from enrollment stats) improve hidden state dynamics compared to raw input?
- Does variable dt from real ISS updates engage the CfC's temporal discrimination where constant dt didn't?
- Can we detect real ISS operational events (docking, EVA, attitude maneuvers) as anomaly score changes?
- What does the CfC hidden state cosine consistency look like on real ISS data?
- Is per-sample scoring (not end-of-sequence) sufficient to catch transients?
- Should enrollment be done on a specific "quiet period" (no events), or on representative operational data?

## First Instincts

- The input pre-scaling problem is the #1 fix. It explains the CMG degenerate attractor completely.
- Fixed normalization from enrollment stats is safe (unlike Welford in the execution path).
- Real ISS data with variable dt will be a far better test of the CfC than synthetic constant-dt data.
- The spike detection failure was a test design bug, not a system architecture bug.
- The tau ablation failure might flip with pre-scaled inputs — if the hidden state actually moves, tau timescale should matter.
- We should re-run all falsification probes after the pre-scaling fix.
- The Lightstreamer feed is confirmed working (ISS Mimic, MIT license). The shim is already built. We just need to run it.
- Enrollment on real ISS data during a "quiet" orbit, then detection during an orbit with known events, would be the definitive test.
