# Seismic Tau Ablation — RAW

## Date: February 2026

## Context
The tau ablation question has been open since ISS probe 1 (test 4). Under constant dt=10s, all tau configurations produced identical results. The honest documentation said: "The tau tuning story is aspirational, not proven."

Seismic data at 100 Hz with genuine multi-timescale structure (P-wave 0.01-0.1s, S-wave 0.1-1s, surface wave 1-30s) is the strongest possible test.

## What I Built
`examples/seismic_detector.c` — 3-channel (HHZ/HHN/HHE) CfC with built-in tau ablation. Runs 3 independent CfC channel sets on identical synthetic seismic data, each with different tau:

- **Seismic**: [0.01, 0.05, 0.2, 0.5, 2.0, 5.0, 15.0, 30.0] — matched to seismic timescales
- **ISS**: [5, 15, 45, 120, 10, 30, 90, 600] — wrong scale (500x-60000x too slow)
- **Constant**: [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0] — null hypothesis

Plus STA/LTA (STA=1s, LTA=30s, trigger=3.0) as the seismology standard baseline.

## Simulation Model
- Background: ocean microseism (0.15 Hz primary, 0.30 Hz secondary) + site noise
- Earthquakes: P-wave onset (5 Hz, sharp envelope), S-wave (2 Hz, later arrival), surface wave (0.1 Hz, large amplitude, arrives last)
- Travel times: P at 6 km/s, S at 3.5 km/s, surface at 3 km/s
- 5 test events: M4.0@100km, M3.0@50km, M6.0@5000km, M2.0@20km, M1.5@100km

## Raw Results

### Hidden State Diagnostics (H-Std)
| Channel | Seismic | ISS | Constant |
|---------|---------|-----|----------|
| HHZ | 0.116 | 0.186 | 0.182 |
| HHN | 0.104 | 0.174 | 0.170 |
| HHE | 0.109 | 0.179 | 0.175 |

Counterintuitive: ISS tau has HIGHER H-Std. But does that mean better detection?

### Detection Results
| Test | Seismic | ISS | Constant | STA/LTA |
|------|---------|-----|----------|---------|
| M4.0 @ 100km | 0.87s | 0.87s | 0.87s | 3.06s |
| M3.0 @ 50km | 0.50s | 0.50s | 0.50s | 4.99s |
| M6.0 @ 5000km | 0.56s | 0.56s | 0.56s | 1.21s |
| M2.0 @ 20km | **0.38s** | 0.82s | 0.82s | 4.28s |
| M1.5 @ 100km | **6.04s** | 14.31s | 14.31s | 4.85s |

### CfC vs STA/LTA
| Test | Winner | Margin |
|------|--------|--------|
| M4.0 | CfC | 2.19s faster |
| M3.0 | CfC | 4.49s faster |
| M6.0 | CfC | 0.65s faster |
| M2.0 | CfC | 3.90s faster |
| M1.5 | STA/LTA | 1.19s faster |

### Performance
- CfC: 58 ns/channel/step, 67 ns total per sample
- STA/LTA: 3 ns/channel/update
- Budget at 100 Hz: 10ms. Actual: 67 ns. Headroom: 148,810x real-time.
- CfC memory: 1,768 bytes. STA/LTA: 37,284 bytes.

## Key Observations
1. For M3+ earthquakes, tau doesn't matter — signal overwhelms
2. For M2 and below, seismic tau detects 2x faster than ISS/constant
3. CfC beats STA/LTA on 4/5 tests
4. STA/LTA beats CfC on the weakest event (M1.5) — its energy integration window helps
5. Higher H-Std doesn't mean better detection (ISS H-Std > Seismic H-Std, but Seismic detects faster)
