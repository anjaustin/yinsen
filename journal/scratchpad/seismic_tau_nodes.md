# Seismic Tau Ablation — NODES

## Date: February 2026

## Node 1: Tau matters, but only at the margin
The signal is clear: for large earthquakes (M3+), all tau configs produce identical detection timing. The earthquake signal is so strong relative to background that the CfC's hidden state gets knocked far from its enrolled mean regardless of decay rates.

But at M2.0 and especially M1.5, seismic tau consistently detects first:
- M2.0: seismic 0.38s vs ISS/constant 0.82s (2.2x faster)
- M1.5: seismic 6.04s vs ISS/constant 14.31s (2.4x faster)

This is a clean result: same weights, same data, same discriminant structure. Only tau differs. The advantage emerges precisely where you'd expect — at the noise floor, where fast decay neurons (tau=0.01-0.05s) can track the P-wave's high-frequency onset while slow neurons are still integrating.

## Node 2: Higher H-Std doesn't mean better detection
ISS tau has 60% higher hidden state variability (0.186 vs 0.116). This is because ISS tau values (5-600s) cause neurons to retain state across many samples at 100 Hz. The CfC "remembers" longer, creating more variance in its trajectory.

But this extra variance is noise, not signal. When the earthquake arrives, the ISS-tau CfC was already wandering — its hidden state deviation from the enrolled mean is noisier, making the anomaly harder to distinguish.

Seismic-tau CfC has lower background variance but sharper response to genuine transients. The fast neurons (0.01s) decay to zero between samples in quiet conditions, then light up when a P-wave arrives. This creates a cleaner signal-to-noise in hidden state space.

## Node 3: CfC and STA/LTA have complementary strengths
CfC wins on sharp onsets (4/5 tests). STA/LTA wins on weak, gradual signals (M1.5). The mechanisms are different:

- CfC detects *deviation from learned state trajectory*. Reacts to instantaneous changes. Memory is in the hidden state + discriminant.
- STA/LTA detects *energy ratio change*. Integrates over its windows. Memory is in the LTA buffer (30s of energy history).

For a P-wave with a sharp onset, CfC reacts within samples. STA/LTA needs to accumulate enough energy to shift the ratio.

For a weak, distant earthquake with gradual energy buildup, STA/LTA's 30s integration window accumulates the evidence. CfC's hidden state may not deviate enough from the enrolled pattern in any single timestep.

## Node 4: The decay math explains the mechanism
At dt=0.01s (100 Hz):
- tau=0.01s: decay = exp(-0.01/0.01) = 0.368 → forgets 63% per step
- tau=0.05s: decay = exp(-0.01/0.05) = 0.819 → retains 82%
- tau=1.0s:  decay = exp(-0.01/1.0)  = 0.990 → retains 99%
- tau=5.0s:  decay = exp(-0.01/5.0)  = 0.998 → retains 99.8%
- tau=600s:  decay = exp(-0.01/600)  = 0.99998 → essentially no decay

With ISS tau values, ALL neurons retain >99.8%. They're all effectively constant — exp(-dt/tau) is approximately 1.0 for all neurons. There's no temporal differentiation.

With seismic tau, neurons 0-1 (0.01, 0.05s) genuinely forget between samples while neurons 6-7 (15, 30s) retain. This creates a frequency-sensitive filter bank where different neurons respond to different timescales of the input. This IS the multi-timescale mechanism working as intended.

## Node 5: Memory advantage is dramatic
CfC total: 1,768 bytes (3 channels, everything).
STA/LTA: 37,284 bytes (3 channels, circular buffers).

CfC is 21x smaller. On an edge device with 64KB L1, STA/LTA uses 57% for buffers alone. CfC uses 2.7%.

Both can trivially keep up at 100 Hz (CfC: 67 ns/sample, STA/LTA: 3 ns/sample). But at higher sample rates or more channels, STA/LTA's buffer memory becomes the bottleneck.

## Node 6: The constant dt caveat
This experiment uses constant dt=0.01s. The ISS tau ablation failed under constant dt=10s. Here, seismic tau wins under constant dt=0.01s. The difference isn't variable dt — it's that the tau values are properly matched to the signal timescales.

The earlier ISS failure was because ISS tau values (5-600s) and ISS dt (10s) created decay values where exp(-10/5)=0.135 to exp(-10/600)=0.983. This is a narrower range of decay rates than seismic: exp(-0.01/0.01)=0.368 to exp(-0.01/30)=0.9997. The seismic case has 3 orders of magnitude more dynamic range in decay.

The key insight: tau matters when it creates **sufficient dynamic range in decay rates** relative to the signal's temporal structure. This happens when tau spans the signal's characteristic timescales.
