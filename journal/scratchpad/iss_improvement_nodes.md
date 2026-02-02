# Nodes of Interest: Improving ISS Telemetry with Real WebSocket Data

## Data Sources
- probe1: Seed independence, null discriminant, cross-channel, tau ablation, random disc, ROC
- probe2: Magnitude sweep, slow drift vs fast spike, CMG attractor analysis, same-subsystem discrimination, CfC vs 3-sigma
- ISS Mimic Lightstreamer: confirmed real, MIT license, `push.lightstreamer.com/ISSLIVE`
- Keystroke falsification (probes 1-4): normalization destruction proof, linear discriminant discovery

---

## Node 1: The Input Scale Problem Is the Root Cause

CMG input range: ~0.001g. After W_gate multiplication (largest weight 0.8): 0.0008. Bias: -0.5. The gate sigmoid sees `sigmoid(-0.5 + 0.0008 + h_contribution)`. The input contributes 0.16% of the pre-activation. The gate is entirely determined by the bias and recurrence.

Hidden state analysis confirms: H-Std = 0.0007, H-Range = 0.0001. The hidden state is a fixed point, not a trajectory.

Coolant (input range ±8°C) works because `0.8 * 8.0 = 6.4` — the input actually pushes through the sigmoid's active region. CoolA H-Std = 0.032, H-Range = 0.103. That's 150x more hidden state variance than CMG.

Why it matters: This is not a CfC architecture problem. It's a scaling problem. The CfC was designed for inputs in roughly [-1, 1]. CMG at 0.001g is three orders of magnitude below the useful range. Fix the scaling, and the architecture should engage.

## Node 2: Fixed Pre-Scaling Is Safe; Online Normalization Is Not

Keystroke probe2 proved that Welford online normalization destroys hidden state consistency (cosine 0.00 vs 0.90). The reason: the normalizer's running statistics SHIFT during execution, so the CfC tracks the normalizer's state rather than the signal's state.

But FIXED normalization — learn min/max or mean/std during enrollment, freeze, apply unchanged during execution — does not have this problem. The transform is a constant affine map: `x_scaled = (x - offset) / scale`. Constants don't shift. The CfC sees a stable, well-scaled input.

Why it matters: The path forward is enrollment-phase calibration. Use the first N samples to learn per-channel statistics, freeze them, then apply during execution. This is the same pattern as the discriminant — learn at enrollment, freeze at execution.

## Node 3: Real ISS Data Has Variable dt — Untested CfC Capability

Simulation used constant dt = 10s. The CfC's exponential decay `exp(-dt/tau)` reduces to a constant per-tau value when dt is fixed. The entire variable-dt machinery — the reason CfC exists instead of a simple RNN — was never exercised.

Lightstreamer pushes updates at irregular intervals. Some parameters update every 1-2 seconds (attitude, power), others every 10-60 seconds (thermal, atmospheric). The CfC's decay computation `exp(-dt/tau[i])` with per-neuron tau values creates different forgetting rates for different neurons at each step. With variable dt, the 8 hidden neurons span different temporal windows at every update.

Why it matters: Variable dt is the CfC's raison d'etre. Testing it with constant dt was like testing a sports car on a treadmill. The tau ablation failure (ISS/keystroke/constant tau all equal) might reverse with variable dt — different tau configurations would produce different temporal resolution profiles on the same irregular update stream.

## Node 4: Per-Sample Scoring Already Exists in stdin Mode

The spike invisibility finding was partly a test design artifact. Probe2 injected a spike and measured the FINAL score. Of course the final score recovered — the spike was dozens of time constants in the past.

The stdin streaming mode already scores every incoming sample: `channels[ch_id].last_score = channel_score(&channels[ch_id])`. If a spike caused a hidden state excursion, the score AT that moment would drop, then recover as the hidden state returns to baseline.

The simulation's Phase 3 also scores per-step and tracks first detection time. The probe2 spike test was just poorly designed — it only looked at end-of-sequence scores.

Why it matters: The architecture handles transients correctly IF you look at per-sample scores. The spike test should be rerun measuring per-sample score minimum, not final score.

## Node 5: The Lightstreamer Feed Provides Ground Truth Events

ISS operations have publicly documented events: docking/undocking, EVAs, attitude maneuvers, solar array repositioning, CMG desaturation burns, CDRA cycling. Many of these are logged with timestamps and are visible in the telemetry.

A CMG desaturation event (momentum dump using RCS thrusters) shows up as: CMG wheel speeds changing simultaneously + brief attitude disturbance + thruster firing indicators. This is a KNOWN event type that should produce a detectable pattern in the CfC hidden states.

Enrollment on a "quiet" orbital period (no events) and then monitoring through a period with known events would give us ground truth for detection.

Why it matters: Ground truth is the difference between "does it detect synthetic anomalies we designed to be detectable" and "does it detect real operational events we didn't design for."

## Node 6: The 3-Sigma Baseline Is the Real Benchmark

Probe2 showed CfC has no advantage over 3-sigma on coolant drift. Both fail equally. This is the honest bar: if CfC + discriminant doesn't outperform `if (value > mean + 3*std) flag()`, the entire apparatus is unjustified overhead.

The CfC SHOULD outperform 3-sigma on:
- Multi-variate correlations (3-sigma is per-channel; CfC hidden state captures temporal context)
- Gradual distribution shifts (3-sigma compares to a fixed threshold; the discriminant compares hidden state trajectory shape)
- Variable-rate temporal patterns (3-sigma is memoryless; CfC has exponential memory with variable time constants)

But these advantages are theoretical until demonstrated on real data with pre-scaled inputs.

Why it matters: Every result should be reported alongside the 3-sigma baseline. If CfC doesn't beat it, say so.

## Node 7: Channel Selection Should Match What Lightstreamer Actually Provides

The Python shim uses hardcoded AMPS identifiers: `USLAB000060` through `USLAB000063` for CMG, `S6000008`/`P6000008` for ETCS, `AIRLOCK000049` for cabin pressure, `NODE3000005` for O2. These were sourced from the ISS Mimic codebase but not verified against the live feed.

Lightstreamer subscription may not return data for all of these. Some might be disabled, renamed, or return stale values. The first thing the live test should do is verify which channels actually produce fresh updates.

Why it matters: We could build a beautiful multi-channel detector and discover that half the channels return NULL or stale data. Verify the data source before building on it.

## Node 8: Enrollment Duration Should Cover at Least One Full Orbital Cycle

ISS orbital period: ~92 minutes. Thermal cycling (the dominant signal on coolant and CMG) follows this period. To learn "normal," enrollment must see at least one full cycle of the dominant signal. Partial-cycle enrollment would learn an incomplete picture and flag normal orbital variation as anomalous.

At 1Hz update rate: 5520 samples per orbit. At 0.1Hz: 552 samples. ISS_MAX_SAMPLES is currently 500, which is tight for 1Hz data.

Why it matters: The enrollment buffer size and duration directly determine whether the discriminant captures the full operational envelope. Too short = false positives on normal cycling.

## Node 9: The Discriminant Structure Proved Sound

Despite the failures, the discriminant mechanism itself is validated:
- Channel specificity: cross-channel scores drop to 0.000 (probe1 test 3)
- Random discriminant: scores 0.08-0.12 vs real 0.73-0.87 (probe1 test 5)
- Same-subsystem: can separate CMG1 from CMG4 by -0.077 (probe2 test 4)
- Seed independence: delta <0.003 between continuous and independent seeds (probe1 test 1)

The discriminant learns and applies correctly. The problem is upstream — the CfC isn't producing rich enough hidden states because the inputs aren't scaled.

Why it matters: The fix is in the input pipeline, not in the discriminant or scoring. The machinery downstream of the hidden state works.

## Node 10: Two-Phase Approach — Calibrate on Real Data, Then Detect

The v2 approach should be:
1. **Calibration run**: Connect to Lightstreamer, stream 2 orbits (~3 hours). Log raw (channel_id, timestamp, value) to disk. Compute per-channel min/max/mean/std. Verify which channels are live. This is pure data collection — no CfC involvement.
2. **Enrollment run**: Replay logged data through the CfC with pre-scaling applied. Learn discriminants. This happens offline, instantly (microseconds per sample).
3. **Live detection**: Connect to Lightstreamer, apply frozen pre-scaling, run CfC + scoring in real-time. Compare to 3-sigma baseline.

Separating calibration from enrollment prevents the chicken-and-egg problem: you need statistics to scale, but you need to scale to get good enrollment.

Why it matters: This is the concrete sequence of operations. Each step has a clear input, output, and success criterion.

## Node 11: The Memory Footprint Claim Needs Revision

The original claim was 1,084 bytes per channel. Probe falsification showed the actual execution-phase per-channel state is 324 bytes (no enrollment buffer). Adding per-channel scaling parameters (min, max, or mean, std — 2 floats per input dim = 16 bytes) barely changes this.

But if enrollment buffer is needed for re-enrollment (adapt to seasonal changes), the full struct is much larger: 500 samples * 8 floats * 4 bytes = 16,000 bytes per channel. That's still tiny (128KB total for 8 channels — fits in L2) but the "L1 cache" narrative applies to execution only, not enrollment.

Why it matters: Be precise about which phase the memory claim applies to.
