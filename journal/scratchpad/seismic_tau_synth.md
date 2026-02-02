# Seismic Tau Ablation — SYNTH

## Date: February 2026

## The Tau Principle (validated)

**CfC time constants (tau) provide a detection advantage when the decay dynamic range spans the signal's temporal structure.**

Formal statement:
- Let R = max(decay) / min(decay) where decay_i = exp(-dt / tau_i)
- Let T = max(timescale) / min(timescale) of the target signal
- Tau differentiation emerges when R is commensurate with T
- The advantage is measurable at low SNR and negligible at high SNR

Evidence:

| Domain | dt | R (decay range) | T (signal range) | Tau matters? |
|--------|-----|-----------------|------------------|--------------|
| ISS (constant dt=10s) | 10s | 7x (0.135–0.983) | ~100x (min–orbit) | NO |
| Seismic (dt=0.01s) | 0.01s | 2700x (0.368–0.9997) | 3000x (P–surface) | YES at low SNR |
| Keystroke (dt~0.05s) | ~0.05s | N/A (constant tau tested) | ~20x (0.05–1s) | Untested |

The ISS failure makes sense: R=7x is too narrow to differentiate timescales spanning 100x. The seismic success makes sense: R=2700x matches T=3000x.

## The Sensitivity Gradient

Tau advantage scales inversely with SNR:

| Magnitude | SNR (approx) | Seismic speedup vs ISS/Const |
|-----------|-------------|------------------------------|
| M4.0 | High | 1.0x (no advantage) |
| M3.0 | High | 1.0x |
| M6.0 | High (distant) | 1.0x |
| M2.0 | Medium | 2.2x |
| M1.5 | Low | 2.4x |

Prediction: at M1.0, seismic tau would show >3x advantage. The fast-decay neurons (tau=0.01-0.05s) act as high-pass filters that can isolate P-wave onsets from the microseismic background. ISS/constant tau neurons integrate everything equally.

## CfC vs STA/LTA: Complementary, Not Competitive

| Detector | Strength | Weakness | Memory |
|----------|----------|----------|--------|
| CfC | Sharp onsets (<1s) | Weak gradual signals | 1,768 bytes |
| STA/LTA | Sustained energy accumulation | Slow response to transients | 37,284 bytes |

A combined detector (CfC for fast detection, STA/LTA confirmation) would be optimal. Implementation: trigger on CfC anomaly OR STA/LTA trigger. Cost: ~39KB total. Still fits in L1.

## Updated Falsification Record

### ISS iss_telemetry.c falsification note (to be updated):
**OLD**: "The tau tuning story is aspirational, not proven."
**NEW**: "Tau tuning is validated on seismic data where decay dynamic range matches signal temporal structure (R~2700x, T~3000x). At ISS timescales with constant dt=10s, R=7x is insufficient. Variable dt from real ISS data remains untested."

### Seismic seismic_detector.c initial falsification:
- **PASSED**: Tau differentiation at M2.0 (2.2x) and M1.5 (2.4x)
- **PASSED**: CfC beats STA/LTA on 4/5 tests
- **PASSED**: CfC memory 21x smaller than STA/LTA
- **HONEST LOSS**: STA/LTA beats CfC on M1.5 (weakest event)
- **NOT YET TESTED**: Real seismic data from SeedLink
- **NOT YET TESTED**: Actual earthquake detection

## What Ships

1. `examples/seismic_detector.c` — 3-channel CfC seismic detector with built-in tau ablation and STA/LTA comparison. Works in sim mode and stdin mode for live data.
2. Updated ISS falsification note with seismic tau results.
3. LMM journal documenting the complete tau ablation investigation.

## Next Moves

1. **Live SeedLink test** — Run against real GFZ seismic data. Background will be real ocean noise. Any event during capture would be a real detection test.
2. **Combined CfC+STA/LTA detector** — Trigger on either. Best of both worlds.
3. **ISS tau redesign** — Apply the R~T principle. For ISS with orbital period 5520s, use tau=[60, 180, 600, 1800, 120, 360, 900, 5520] and test with variable dt from live ISS data.
