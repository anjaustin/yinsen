# Synthesis: Chip Stack Application Roadmap

The clean cut. Three applications, ordered by buildability. One architecture experiment. One infrastructure default.

---

## Key Decision: We are building temporal sensor primitives, not an inference engine.

The competition for "run ML on MCUs" is crowded (TFLite Micro, CMSIS-NN, Edge Impulse, etc.). We don't win that fight. We win a different fight: **continuous-time temporal reasoning in the smallest possible footprint with deterministic, auditable behavior.**

Every application below is chosen because it sits in the intersection of:
- Tiny hardware (where the competition can't fit)
- Temporal signals (where CfC has a categorical advantage)
- A real problem someone will pay to solve

---

## Application 1: Keystroke Biometrics (THE DEMO)

### Why first
- No external hardware required. Any computer has a keyboard.
- Variable-dt IS the feature — not an argument, a fact.
- Privacy-preserving (on-device, no cloud) is a selling point.
- Low regulatory burden. No safety implications.
- End-to-end demo proves the entire chip pipeline.

### Chip Pipeline
```
Keyboard event -> (key_code, dt_since_last) -> [norm_chip_welford: track typing stats]
                                              -> [cfc_cell_chip: hidden_dim=8]
                                              -> [softmax_chip or raw sigmoid: auth score]
```

### Specifications
| Parameter | Value |
|-----------|-------|
| input_dim | 2 (key_code normalized, dt) |
| hidden_dim | 8 |
| output_dim | 1 (authentication score) |
| Estimated execution | <200 ns (4x8 = 97ns baseline, +overhead) |
| Model size | 8*(2+8+1)*2 bits = 22 bytes of ternary weights |
| Total .text | <3 KB (CfC chip + norm chip + softmax chip) |
| RAM | <256 bytes (hidden state + Welford accumulators) |
| Training | Offline in Python, export ternary weights as C array |

### What to build
1. `examples/keystroke_biometric.c` — Main demo program
   - Enrollment mode: record N keystrokes, compute timing features
   - Authentication mode: score incoming keystrokes against enrolled profile
   - Uses `cfc_cell_chip` with pre-trained weights (hardcoded C array)
   - Uses `norm_chip_welford` for input normalization
   - Prints running authentication confidence

2. `scripts/train_keystroke.py` — Training script (simple)
   - Record keystroke timings from terminal
   - Train small CfC on positive (enrolled user) + negative (random timing) examples
   - Quantize to ternary weights
   - Export as C header

### What's missing from the forge
- Nothing critical. All chips needed exist.
- Minor: may want a `threshold_chip` macro for binary decision from score. Trivially added.

### Success criteria
- [ ] Demo enrolls a user in <30 seconds of typing
- [ ] Demo correctly authenticates enrolled user with >90% true positive
- [ ] Demo rejects random typing with >90% true negative
- [ ] Total compiled binary <8 KB
- [ ] Execution per keystroke <1 us

---

## Application 2: Vibration Predictive Maintenance (THE MARKET)

### Why second
- Largest addressable market ($10B+ PdM industry).
- Uses the richest chip pipeline (FFT + norm + CfC + softmax).
- Targets disconnected environments where cloud PdM fails.
- Demo is buildable in pure simulation (synthetic vibration data).

### Chip Pipeline
```
Accelerometer (256 samples @ 25.6 kHz = 10ms window)
    |
    v
[fft_chip_forward: 256-point FFT]
    |
    v
[fft_chip_band_energy: 8 frequency bands]
[fft_chip_dominant_freq: peak frequency]
    |
    +---> [norm_chip_welford: running stats on each band]
    |          |
    |          v (mean_drift, var_growth as auxiliary features)
    |
    v
[concat: 8 band_energies + 1 dominant_freq + 2 drift_features = 11 inputs]
    |
    v
[cfc_cell_chip: hidden_dim=16, dt=10ms fixed]
    |
    v
[softmax_chip: 5 classes]
    |
    v
Classification: {normal, inner_race, outer_race, ball_fault, misalignment}
```

### Specifications
| Parameter | Value |
|-----------|-------|
| input_dim | 11 (8 bands + dominant_freq + mean_drift + var_growth) |
| hidden_dim | 16 |
| output_dim | 5 (fault classes) |
| FFT size | 256-point |
| Sample rate | 25.6 kHz |
| Window | 10 ms (256 samples) |
| Execution per window | FFT: ~50us + CfC: ~2us + softmax: <1us = ~53us total |
| Model size | 16*(11+16+5)*2 bits = 128 bytes ternary weights |
| Total .text | <6 KB (FFT + norm + CfC + softmax chips) |
| RAM | ~2.5 KB (FFT buffers + hidden state + Welford accumulators) |
| Target hardware | Cortex-M0+ (32KB flash, 8KB SRAM) — fits easily |

### What to build
1. `examples/vibration_pdm.c` — Simulation demo
   - Generates synthetic vibration: base frequency + fault harmonics + noise
   - Processes through full chip pipeline
   - Outputs per-window classification and confidence
   - Includes drift detection: flags when input distribution shifts

2. `scripts/gen_vibration_data.py` — Synthetic data generator
   - Simulates healthy bearing + 4 fault modes
   - Uses standard bearing fault frequency formulas (BPFO, BPFI, BSF, FTF)
   - Outputs training data for Python CfC trainer

3. `scripts/train_vibration.py` — Training + ternary quantization
   - Trains CfC on synthetic vibration features
   - Quantizes to ternary
   - Exports C weight arrays

### What's missing from the forge
- Nothing. FFT, norm, CfC, and softmax chips all exist.
- May want a convenience macro `PIPELINE_VIBRATION` that wires all chips together, but this is sugar, not substance.

### Success criteria
- [ ] Demo correctly classifies 4 fault types on synthetic data (>85% accuracy)
- [ ] Drift detection flags distribution shift when fault severity changes
- [ ] Total compiled binary <12 KB
- [ ] Full pipeline per window <100 us
- [ ] Runs on simulated Cortex-M0 memory budget (8KB SRAM)

---

## Application 3: CAN Bus Anomaly Detection (THE TECHNICAL FIT)

### Why third
- Best technical fit for our differentiators (variable-dt + determinism + auditability).
- Real regulatory demand (UN R155 vehicle cybersecurity).
- Harder to demo (needs CAN bus data or simulator).
- Harder to market (automotive OEM sales cycles are 3-5 years).

### Chip Pipeline
```
CAN message -> parse (msg_id, timestamp, payload)
    |
    v
[per-message-id timer: dt = timestamp - last_seen[msg_id]]
    |
    v
[norm_chip_welford: per-ID timing statistics]
    |
    v
[feature vector: dt, dt/expected_period, payload_delta, timing_jitter]
    |
    v
[cfc_cell_chip: hidden_dim=8, dt=variable (genuine irregular)]
    |
    v
[sigmoid output: anomaly_score 0.0-1.0]
    |
    v
Threshold -> alert / log
```

### Specifications
| Parameter | Value |
|-----------|-------|
| input_dim | 4 (dt, dt_ratio, payload_delta, jitter) |
| hidden_dim | 8 |
| output_dim | 1 (anomaly score) |
| Execution per message | <200 ns |
| Model size | 8*(4+8+1)*2 bits = 26 bytes ternary weights |
| Total .text | <3 KB |
| RAM | <512 bytes (hidden state + per-ID timers for top-N msg IDs) |
| Target hardware | Automotive-grade Cortex-M0/M3 |

### What to build
1. `examples/can_anomaly.c` — Simulation demo
   - Simulates normal CAN traffic (periodic messages with jitter)
   - Injects anomalies (timing deviation, missing messages, replay attacks)
   - CfC processes each message with true variable-dt
   - Reports anomaly score timeline

### What's missing from the forge
- A simple per-ID timer/tracker utility. Not a chip per se, just bookkeeping. ~50 lines of C.

### Success criteria
- [ ] Demo detects timing anomalies (message delays >3 sigma) with >90% recall
- [ ] Demo detects replay attacks (duplicate messages at wrong time) with >95% recall  
- [ ] False positive rate <5% on normal traffic
- [ ] Per-message execution <500 ns

---

## Architecture Experiment: Multi-Timescale CfC

### Purpose
Validate or kill the hypothesis that multi-timescale CfC outperforms single CfC for signals with both fast and slow dynamics.

### Design
```
Test signal: sin(2*pi*5*t) + 0.3*sin(2*pi*0.01*t)  [5 Hz fast + 0.01 Hz drift]

Architecture A (baseline): Single CfC, hidden_dim=16
Architecture B (multi):    Fast CfC (hidden=4, tau=0.01) + Slow CfC (hidden=4, tau=10.0)
                           -> concat(fast_h, slow_h) -> Decision CfC (hidden=4)

Task: Predict next value. Measure MSE after 1000 steps.
```

### What to build
- `experiments/multi_timescale.c` — Pure C experiment
- Pre-computed weights for both architectures (can be random init for comparison, or pre-trained)
- Report: which architecture achieves lower MSE? At what parameter count?

### Decision gate
- If multi-timescale wins with fewer total parameters: pursue as architecture paper + add to examples
- If single CfC wins or ties: park the idea, don't add complexity

---

## Infrastructure Default: Drift Detection in Every Pipeline

### Decision
Include `norm_chip_welford` in every application pipeline by default. Track input distribution statistics and expose them as:
1. Auxiliary features to the CfC (the model can learn to use distribution shift as a signal)
2. A monitoring output (alert when inputs drift beyond training distribution bounds)

### Implementation
Every example should include:
```c
// After N warmup samples, check for distribution shift
if (welford.count > WARMUP_N) {
    float mean_drift = fabsf(welford.mean - training_mean);
    float var_ratio  = welford.variance / training_variance;
    if (mean_drift > DRIFT_THRESHOLD || var_ratio > VAR_THRESHOLD) {
        // Flag: operating outside training distribution
    }
}
```

This costs ~20 bytes of RAM per monitored feature and negligible compute. There is no reason not to include it.

---

## Roadmap Summary

```
NOW (Week 1-2):
  [x] Complete LMM exploration (this document)
  [ ] Build keystroke biometric demo (examples/keystroke_biometric.c)
  [ ] Hardcode reasonable ternary weights for keystroke demo
      (hand-tuned or Python-trained)

NEXT (Week 3-4):
  [ ] Build vibration PdM simulation (examples/vibration_pdm.c)
  [ ] Run multi-timescale experiment (experiments/multi_timescale.c)
  [ ] Write synthetic data generators (scripts/)

THEN (Week 5-8):
  [ ] Build CAN bus anomaly demo (examples/can_anomaly.c)
  [ ] If multi-timescale validated: integrate into vibration pipeline
  [ ] Hardware validation: run keystroke demo on actual M0 dev board
  [ ] Performance profiling: measure real power consumption

LATER (Month 3+):
  [ ] Domain partnerships for vibration PdM
  [ ] Automotive cybersecurity partnership for CAN bus
  [ ] Control output chip (PID replacement prep)
  [ ] Medical pathway research (FDA predicate search)
```

---

## What We're NOT Doing

- **Not building a general ML framework.** We're building specific application pipelines.
- **Not competing with TFLite Micro on Cortex-M4+.** We play where they can't fit.
- **Not entering medical or automotive directly.** We prove the technology in lower-risk domains first.
- **Not building training in C.** Training stays in Python. C is for execution only.
- **Not adding chips unless a concrete application demands them.** The forge is frozen until a demo reveals a gap.

---

*The wood cuts itself when you understand the grain.*
*Three applications. Two axes (tiny hardware, temporal signals). One temporal sensor primitive.*
