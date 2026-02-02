# Reflections: Improving ISS Telemetry with Real WebSocket Data

## Core Insight

The falsification didn't break the architecture. It broke the INPUT PIPELINE. The CfC cell, the discriminant, the cross-channel scoring — all proven sound by probes 1 and 2. What failed is that raw ISS telemetry values (0.001g for CMG, 101.3 kPa for cabin pressure) live outside the CfC's useful dynamic range. The sigmoid gate can't distinguish `sigmoid(-0.5 + 0.0008)` from `sigmoid(-0.5 + 0.0000)`. Both return ~0.378. The input is invisible to the CfC.

This is the same class of problem as keystroke v1's Welford normalization — a pre-processing error that made the signal invisible — but with the opposite sign. v1 normalized too aggressively and destroyed the signal. The ISS demo doesn't normalize at all and the signal never arrives.

The fix is a frozen affine transform per channel: `x_scaled = (x - center) / scale`, where center and scale are learned during a calibration phase and never change during execution. This is safe (keystroke probes proved FIXED transforms are fine; MOVING transforms are lethal) and sufficient (it maps the channel's operating range into the CfC's useful input range).

## Resolved Tensions

### Node 1 vs Node 2: Scaling Need vs Normalization Danger

Resolution: These are not in tension. The keystroke lesson was about ONLINE normalization — Welford stats that shift during execution. The ISS fix is OFFLINE calibration — statistics learned once from data and frozen. One is a moving target; the other is a constant. The CfC hidden state is consistent with constants; it's inconsistent with drifting transforms.

Concrete: During calibration, compute `mean_i` and `std_i` for each channel. During execution, apply `x_scaled = (x - mean_i) / (std_i + epsilon)`. These 4 floats per channel (16 bytes) are frozen after calibration, just like the discriminant is frozen after enrollment.

### Node 3 vs Node 5: Variable dt Testing vs Tau Ablation Failure

Resolution: The tau ablation failure occurred under CONSTANT dt = 10s. With constant dt, `exp(-dt/tau[i])` is just a per-neuron constant multiplied by the hidden state at every step. There's no temporal information in the decay — it's the same decay at every step. Different tau values just produce different constant multipliers, which the discriminant absorbs into its learned mean and PCA.

With VARIABLE dt, each step has a DIFFERENT decay profile. A CMG update 1 second after the last one produces `exp(-1/5) = 0.82` on the fastest neuron and `exp(-1/600) = 0.998` on the slowest. A thermal update 60 seconds later produces `exp(-60/5) = 0.000` on the fastest and `exp(-60/600) = 0.905` on the slowest. The fast neurons respond to recent updates; the slow neurons integrate over minutes. This temporal decomposition is precisely what multi-scale tau provides — and it's invisible under constant dt.

Prediction: With real Lightstreamer data (variable dt), the tau ablation test should show ISS tau outperforming keystroke tau. Keystroke tau (0.05-0.8s) would make ALL neurons respond only to the most recent update because `exp(-10/0.8) = 0.000` even for the slowest neuron at 10-second dt. The temporal memory would be erased at every step.

This is testable: re-run the tau ablation probe with variable dt from logged ISS data. If tau still doesn't matter, the CfC's temporal memory genuinely doesn't help. If it does matter, we've identified the test condition that exposes the architecture's value.

### Node 4 vs Node 6: Spike Test Design Bug vs Real Limitation

Resolution: Partly test design bug, partly real limitation. The probe2 spike test measured only the final score, which is wrong — a single-sample spike would affect the hidden state for only a few tau time constants, then decay away. The score AT the spike moment (or within one time constant) would show the impact.

But there IS a real limitation: the CfC is a lowpass system. A single-sample spike of duration dt gets mixed into the hidden state with weight `gate * candidate`, where gate is sigmoid-bounded in [0, 1]. The maximum hidden state change from one sample is bounded by the gate magnitude. For large spikes (20°C on a 15°C baseline, with proper pre-scaling), the gate should saturate at ~1.0, producing a maximal hidden state update. The question is whether that maximal update is large enough to push the score below threshold.

The right test: inject a spike, measure the score on the spike sample, and on each subsequent sample until recovery. Map the "anomaly visibility window" — how many samples does it take for the score to return to normal?

### Node 8 vs Node 10: Orbital Coverage vs Practical Enrollment Duration

Resolution: Enrollment needs to cover the dominant period of the signal. For ISS, that's orbital (~92 min). But we don't need to run enrollment in real-time. The two-phase approach (Node 10) solves this: calibrate by logging 2 orbits to disk (~3 hours, can run unattended overnight), then replay through the CfC for enrollment in seconds.

The ISS_MAX_SAMPLES = 500 limit is a code constraint, not a physics constraint. For replay enrollment, we can use all logged samples. For live enrollment, we need either a larger buffer or a running enrollment algorithm that incrementally updates the discriminant.

The practical approach: overnight calibration log + offline enrollment. Then live detection next day. This is how real monitoring systems work — you don't calibrate during an emergency.

## What Survives This Reflection

1. **Fixed pre-scaling is the #1 fix.** Every failure traces back to inputs being outside the CfC's useful range. Fix the scaling and re-test everything.

2. **Variable dt is the CfC's real advantage.** Constant-dt testing disabled the mechanism that makes CfC different from a simple RNN. Real ISS data with irregular updates is the true test.

3. **The two-phase approach (calibrate → enroll → detect) is operationally sound.** Separate data collection from model fitting from execution.

4. **Per-sample scoring is correct for transient detection.** The architecture supports it; the probe test design was wrong.

5. **3-sigma baseline must accompany every CfC result.** If we can't beat the simplest possible detector, we should say so honestly.

## What's Fragile

1. **The prediction that tau will matter under variable dt.** This is a hypothesis. It might fail. If keystroke tau and ISS tau still perform identically on variable-dt real ISS data, the CfC's temporal machinery is genuinely not contributing.

2. **The assumption that real ISS data has richer structure than synthetic.** It probably does, but "richer" doesn't guarantee "CfC-exploitable." The richness might be in frequency domain features (bearing harmonics) that the CfC's point-value input can't capture without FFT pre-processing.

3. **Untrained weights.** Pre-scaling fixes the input range problem, but the weight matrices still map inputs to hidden states via arbitrary-ish projections. The CfC will produce CONSISTENT representations (architecture guarantees this), but not necessarily USEFUL ones. Training remains the long-term path to real performance.

## What I Now Understand

The ISS telemetry demo is a pipeline validation, not an anomaly detection system. The pipeline is sound — data flows from Lightstreamer through Python to C, through the CfC, through the discriminant, out as anomaly scores. What's missing is the signal quality at the CfC input layer.

The path from pipeline validation to real detection has three steps:
1. **Pre-scaling** (fixes input range — required, no ML)
2. **Real data** (exercises variable dt — reveals whether architecture matters)
3. **Trained weights** (maps inputs to useful hidden features — future work, requires Python training loop)

Steps 1 and 2 can be done now with the tools we have. Step 3 is future work that requires infrastructure we haven't built yet.

The honest narrative: "We built a multi-channel streaming CfC pipeline that processes ISS telemetry at 55 ns/channel in 3,328 bytes. Falsification exposed an input scaling problem that made 4 of 8 channels degenerate. The fix (frozen pre-scaling from calibration data) is designed and ready to implement. Real ISS data via Lightstreamer will be the first test of the CfC's variable-dt temporal discrimination — the capability that makes it architecturally distinct from a threshold detector."
