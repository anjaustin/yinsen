# Nodes of Interest: Chip Stack Applications

Extracted from Phase 1 RAW. Each node is a discrete idea that stands alone.
Connections and tensions noted. No solutions yet — just mapping the grain.

---

## Node 1: The Variable-dt Differentiator

CfC treats irregular time intervals as a first-class feature, not a bug to work around. Most edge ML (TFLite LSTM, CMSIS-NN RNN) assumes fixed sample rates and panics or interpolates when timestamps are irregular.

**Where this is real:** GPS (1-15s fixes), CAN bus (event-driven), BLE RSSI (opportunistic), event cameras (brightness-triggered), medical ECG with dropout.

**Why it matters:** This isn't a marginal improvement. It's a categorical difference. You either handle irregular dt natively or you don't. We do.

**Tension with Node 7:** Variable-dt shines in sparse, event-driven signals. But some of our best applications (vibration monitoring, audio) are actually regularly sampled. Variable-dt is a differentiator *where it applies*, not everywhere.

---

## Node 2: Spectral-Temporal Pipeline (FFT -> CfC)

FFT chip extracts frequency content at one moment. CfC tracks how that content evolves over time. This is an incremental, on-device spectrogram with memory.

**What's new here:** The FFT chip already gives us band energy, dominant frequency, and log magnitude. Feed those as features into CfC, and you get a system that tracks spectral evolution — frequency shifts, harmonic changes, broadband noise growth — without storing a full spectrogram.

**Concrete pipeline:** Raw signal (N samples) -> `fft_chip_forward` -> `fft_chip_band_energy` (K bands) -> K-dimensional input to `cfc_cell_chip_forward` -> classification/anomaly score.

**Why it matters:** This is the predictive maintenance pipeline. Bearing wear shows up as spectral change over time. Gear mesh faults shift harmonics. Pump cavitation grows broadband noise. All temporal-spectral signatures.

---

## Node 3: Auditability as Regulatory Moat

Ternary weights are human-readable: +1, 0, -1. You can literally print the weight matrix and a domain expert can reason about it. "This input has positive influence on this output" is not an approximation — it's exact.

**Where this matters:**
- FDA (medical devices: 510(k) and De Novo pathways want model interpretability)
- ISO 26262 (automotive functional safety: determinism + traceability)
- DO-178C (aviation software: every code path must be testable)
- IEC 61508 (industrial safety: verifiable algorithms)

**What we have:** Bit-identical determinism (proven in bench_cfc_chip.c). 49KB total source (auditable in a day). 199 tests (exhaustive verification). Weights are literally a lookup table of {-1, 0, +1}.

**Tension with Node 10:** Auditability is a moat but not a product. Nobody buys "auditable AI." They buy a solution to a problem that happens to be auditable. The moat only works once you're already solving something.

---

## Node 4: Multi-Timescale Architecture

Chain CfC cells with different tau values: fast cell (tau=0.01s) for transient detection, slow cell (tau=60s) for trend tracking, both feeding a decision cell.

**Why this is interesting:** Biological neural systems operate at multiple timescales. Reflexes (ms), motor control (100ms), learning (minutes), memory (hours+). Our chip stack already supports this — call `cfc_cell_chip_forward` twice with different parameters. The `decay_chip_precompute` makes the exp() cost near-zero for fixed tau.

**Concrete architecture:**
```
sensor -> FFT -> [fast_cfc(tau=0.01)] -----> [decision_cfc]
                 [slow_cfc(tau=60.0)] -----> [decision_cfc]
```

**What's missing:** We have no example of this. We don't know if it actually learns better than a single CfC with more hidden units. This is an architecture hypothesis, not a proven capability.

**Tension with Node 12:** Multi-timescale adds complexity. If the chip stack's advantage is simplicity and auditability, adding architectural complexity works against that.

---

## Node 5: PID Replacement / Learned Control

CfC output is continuous. It doesn't have to be a classifier — it can be a controller. Replace hand-tuned PID loops with a learned controller that adapts to plant dynamics.

**Where PID is fragile:**
- Nonlinear plants (PID is a linear controller)
- Systems with variable dynamics (load changes, temperature drift)
- Multi-input-multi-output coupling (PID is SISO)
- Systems where tuning takes days of expert time

**Why CfC fits:**
- Continuous-time dynamics match control theory naturally
- Variable dt handles irregular control loops
- Small enough to run in the control loop (97ns at 4x8)
- Deterministic = certifiable

**What's missing:** A PID output chip. Our current chips classify or predict. A control chip needs: clamped output range, anti-windup equivalent, rate limiting. This is a gap in the forge.

**Tension with Node 3:** Control is the hardest domain to certify. "Auditable" helps but doesn't solve the fundamental challenge of verifying learned controllers against all possible plant states.

---

## Node 6: Keystroke / Behavioral Biometrics

Typing rhythm is a temporal signal with natural irregular intervals (time between keystrokes varies). CfC can model a user's typing dynamics as a continuous-time signature.

**Why it fits:** Variable dt is the feature, not a nuisance. The inter-keystroke timing IS the biometric. CfC sees this natively. Hidden state accumulates a "typing fingerprint" over a session.

**Practical details:**
- Input: (key_id, dt_since_last_key) pairs
- Hidden dim: probably 8-16 is enough (typing patterns aren't high-dimensional)
- Output: similarity score to enrolled template
- Latency budget: generous (100ms+ is fine, we run in <1us)

**What's interesting:** This runs entirely on-device. No cloud. No network. Privacy-preserving biometrics. The model is tiny enough to store per-user on a smartcard or secure element.

---

## Node 7: Predictive Maintenance (Vibration/Acoustic)

The most mature market for edge temporal ML. Bearings, motors, pumps, compressors — they all fail with spectral-temporal signatures that develop over hours to weeks.

**Our pipeline:** Accelerometer/microphone -> FFT chip -> band energies -> CfC -> anomaly score / fault classification.

**Market reality:**
- Companies already spend $100K+ per monitored asset on cloud-based systems
- The value proposition of on-device is: no connectivity, no data egress, no cloud cost
- STM32/ESP32-class MCUs are already deployed on vibration sensors
- The gap: they run threshold alarms, not learned models

**What we offer:** A model that fits in the sensor node's existing MCU. No cloud. No connectivity dependency. Processes data where it's generated.

**Tension with Node 1:** Vibration data is regularly sampled (e.g., 25.6 kHz accelerometer). Variable-dt is not the differentiator here. The differentiator is code density + determinism + no-cloud.

---

## Node 8: In-Sensor Compute / Smart Transducer

The chip stack is small enough to embed in the sensor package itself. Not "edge" as in a gateway — edge as in the sensor's own MCU.

**The numbers:** CfC 4x8 = 97ns. Total chip source = 49KB. Compiled .text for CfC alone = 2,232 bytes. This fits in an Cortex-M0+ with 32KB flash and 8KB SRAM.

**What this enables:** The sensor outputs a decision, not raw data. An accelerometer that outputs "bearing fault type 3, confidence 0.87" instead of raw vibration waveforms. A current sensor that outputs "motor winding degradation, 72% remaining life" instead of amperage readings.

**Why it matters:** Data reduction at source. A 25.6 kHz accelerometer generates ~50KB/s of raw data. A classification every second is 4 bytes. That's a 12,500x data reduction. This changes the networking/storage architecture completely.

---

## Node 9: CAN Bus Anomaly Detection

Vehicle CAN bus messages arrive at irregular intervals. Each message type has its own nominal timing. Deviations in timing patterns can indicate: ECU failures, intrusion attempts (security), bus loading issues, intermittent wiring faults.

**Why CfC fits:** The input IS irregular timestamps. Each CAN message ID has a different expected interval. CfC's variable-dt models the expected timing distribution and flags deviations.

**Pipeline:** CAN message -> (message_id, dt_since_last_same_id, payload_delta) -> CfC -> anomaly score.

**Market context:** Automotive cybersecurity (UN R155) now requires intrusion detection. Current approaches use fixed-window statistics. CfC could learn more nuanced timing patterns.

**Tension with Node 3:** Automotive is ISO 26262 territory. Getting a learned model into a safety-relevant system requires ASIL qualification. Our auditability helps but doesn't solve the full certification challenge.

---

## Node 10: Medical Wearable (ECG/PPG Arrhythmia)

ECG signals have natural temporal dynamics. Arrhythmias are fundamentally about timing — irregular R-R intervals, prolonged QT, variable P-wave morphology.

**Why CfC fits:** Heart rhythms ARE temporal dynamics with clinically meaningful variable intervals. The closed-form solution means we can express "time since last R-peak" as a first-class feature.

**Regulatory reality:** FDA 510(k) for a new arrhythmia detector requires predicate device, clinical validation, and substantial equivalence. De Novo pathway for novel classification. Both take 6-18 months and $200K-$2M.

**Tension with Node 3:** This is where auditability has the highest value but the regulatory path is longest. You can't shortcut FDA with "but the weights are interpretable." You still need clinical trials.

**Tension with Node 8:** Medical wearables have the strictest power/size constraints, which is where we excel. But they also have the strictest safety requirements, which is where we lack certification history.

---

## Node 11: Power Grid / Mains Frequency Monitoring

Mains frequency (50/60 Hz) drifts under load. These drifts carry information about grid events: generator trips, load steps, inter-area oscillations. FFT of mains frequency gives low-frequency oscillation modes.

**Pipeline:** Mains voltage -> zero-crossing timer -> frequency estimate (irregular dt due to measurement jitter) -> CfC -> event classification.

**Why it's interesting:** This is a real industrial need (grid monitoring), the hardware is dead simple (a voltage divider and a comparator), and the signal is inherently variable-dt (zero-crossing timing jitter).

**Market reality:** PMUs (Phasor Measurement Units) cost $5K-$50K. A microcontroller doing CfC-based frequency analysis costs $5. The accuracy gap matters, but for distribution-level monitoring (not transmission), the economics could work.

---

## Node 12: Code Density as the Actual Moat

The chip forge compiles to 2,232 bytes .text for CfC. The entire chip source is 49KB. This is small enough to:
- Fit in L1 instruction cache entirely (no instruction cache misses during inference)
- Audit the complete source in a single sitting
- Formally verify with bounded model checking tools
- Deploy on the smallest microcontrollers ($0.50 parts)

**The insight from the benchmark:** Scalar ternary is 1.6-4.8x slower than float for raw ALU work. The advantage isn't compute speed — it's working set size. When your entire model + inference engine fits in L1 cache, you avoid the memory hierarchy penalties that dominate large model inference.

**This reframes everything:** We're not competing on FLOPS. We're competing on bytes. The question isn't "how fast is our multiply" but "how much of the system fits in the fastest memory?"

---

## Node 13: Drift Detection via Online Statistics

The norm chip's Welford streaming normalization tracks running mean and variance. These statistics are themselves features. If mean is drifting: sensor calibration issue or environmental change. If variance is growing: system becoming unstable.

**Pipeline:** Sensor -> `norm_chip_welford_update` (track stats) -> normalized input to CfC, BUT ALSO -> drift magnitude as auxiliary input to CfC.

**Why it matters:** This is self-monitoring. The system knows when its input distribution is changing. This is a prerequisite for safe deployment — a model that knows when it's operating outside its training distribution.

**Connection to Node 3:** Drift detection + distribution shift awareness is part of the auditability story. "The model flagged that its inputs have drifted outside training bounds" is exactly what regulators want to hear.

---

## Connections Map

```
       [1: Variable-dt] -----> [6: Keystroke]
             |                  [9: CAN Bus]
             |                  [11: Power Grid]
             v
       [2: FFT->CfC] -------> [7: Pred. Maintenance]
             |                  [11: Power Grid]
             v
       [4: Multi-timescale] -> [7: Pred. Maintenance]
             |                  [5: PID Control]
             v
       [3: Auditability] ---> [10: Medical]
             |                  [9: CAN/Automotive]
             |                  [5: PID Control]
             v
       [12: Code Density] --> [8: In-Sensor]
             |                  [ALL applications]
             v
       [13: Drift Detection] -> [7: Pred. Maintenance]
                                 [10: Medical]
                                 [8: In-Sensor]

  ENABLERS (horizontal):  Nodes 1, 2, 3, 4, 12, 13
  APPLICATIONS (vertical): Nodes 5, 6, 7, 8, 9, 10, 11
```

---

## Boundary Cases (The Deltas)

Per the Laundry Method — what sits at bucket boundaries and needs extra scrutiny:

1. **Node 5 (PID Control)** sits between "application" and "architecture gap." We don't have a control output chip. This is either the highest-value thing to build next, or a rabbit hole.

2. **Node 4 (Multi-timescale)** sits between "enabler" and "unproven hypothesis." It's architecturally clean but empirically unknown. Could be the key unlock or could add complexity for no gain.

3. **Node 10 (Medical)** sits between "highest value" and "highest barrier." Auditability is maximally valuable here but the regulatory path is maximally expensive.

4. **Node 7 (Pred. Maintenance)** is the "safest" application but also the most crowded market. The delta: our differentiator (no cloud, on-device) matters most where connectivity is worst — underground mines, offshore platforms, remote infrastructure.

