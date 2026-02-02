# Reflections: Chip Stack Applications

Sharpening the axe. Taking each node seriously. Looking for hidden assumptions.

---

## Core Insight

After sitting with the 13 nodes, a pattern emerges: **we are not building an inference engine. We are building a temporal sensor primitive.**

The distinction matters. An inference engine competes on throughput, accuracy, and model compatibility. We lose that fight — TFLite Micro, CMSIS-NN, and TinyML frameworks all run bigger models faster on the same hardware.

A temporal sensor primitive competes on: how tightly can you couple temporal reasoning with the sensing element? How small is the gap between "data acquired" and "decision made"? Can the entire perception-to-action pipeline fit inside the sensor's own resources?

Our advantage is not "we run neural networks on MCUs." Lots of people do that. Our advantage is: **we run continuous-time temporal dynamics in 97 nanoseconds, in 2KB of code, with bit-identical determinism and human-readable weights.** That's a different thing.

---

## Node-by-Node Honesty Check

### Node 1 (Variable-dt): REAL but narrow

Variable-dt is a genuine differentiator, but only in a subset of applications. Most industrial sensors sample regularly. Where variable-dt is the *primary* differentiator:

- **CAN bus / event-driven protocols**: YES. Timing IS the signal.
- **Keystroke biometrics**: YES. Inter-event timing IS the feature.
- **BLE/RF proximity**: YES. RSSI arrives opportunistically.
- **ECG with dropout**: PARTIAL. Regular sampling with occasional gaps.
- **Vibration monitoring**: NO. Fixed sample rate. Variable-dt is irrelevant.

**Honest assessment:** Variable-dt is a moat for event-driven applications, a convenience for gapped data, and irrelevant for fixed-rate sensors. Don't oversell it.

### Node 2 (FFT->CfC): REAL and buildable today

This is the most immediately implementable pipeline. We have both chips. We have measured latency numbers. The only missing piece is a trained model.

**Smallest viable demo:** Generate synthetic vibration data with injected bearing fault signatures (inner race, outer race, ball, cage frequencies). FFT -> band energies -> CfC -> fault classification. Entirely in simulation. No hardware needed. This proves the pipeline works before we touch any physical sensor.

**Hidden assumption check:** We assume FFT band energies are sufficient features. In practice, vibration fault detection also uses time-domain features (kurtosis, crest factor, RMS). Our pipeline might need a mixed feature vector: some time-domain, some frequency-domain. The norm chip's running statistics (mean, variance) could provide the time-domain features.

**Revised pipeline:**
```
raw signal -> norm_chip_welford (running RMS, kurtosis proxy)
           -> fft_chip (band energies, dominant freq)
           -> concat features -> cfc -> classification
```

This is richer and uses three chips cooperatively. It's also still tiny.

### Node 3 (Auditability): REAL but not a product

This came up in the NODES tension and it's worth dwelling on. Auditability is a *property*, not a *product*. Nobody wakes up and says "I need auditable AI." They say "I need arrhythmia detection that I can get through the FDA."

The auditability story works when it's the tiebreaker: "We solve your problem AND you can inspect every weight AND the output is deterministic AND the codebase is auditable in a day." That's compelling. "We're auditable" alone is not.

**Where auditability is the tiebreaker:**
- Anywhere two solutions have comparable accuracy but one is a black box and one is transparent
- Safety-critical systems where the certifier *requires* model interpretability
- Regulated industries where audit trails are legally mandated

**Honest assessment:** Auditability accelerates adoption in regulated markets. It does not create markets. Build the solution first, let auditability be the wedge that gets you through the door that competitors can't open.

### Node 4 (Multi-timescale): HYPOTHESIS — needs testing

This is the most intellectually exciting node and the least validated. The idea is clean: fast CfC for transients, slow CfC for trends, decision CfC to combine them. But:

- Does it actually learn better than a single CfC with larger hidden dim?
- What's the training procedure? Do you train end-to-end or in stages?
- The decision cell's input dimension is (fast_hidden + slow_hidden). Does this push us into larger networks that lose the code density advantage?

**What would change my mind:** A synthetic experiment where multi-timescale CfC provably outperforms single-timescale on a signal with both fast and slow components. E.g., a sine wave (fast) with a drifting baseline (slow). If multi-timescale learns this with hidden_dim=4+4 where single-timescale needs hidden_dim=32, that's a win.

**Honest assessment:** Park this for now. Build it as an experiment after we have a working single-CfC application. Don't let architectural elegance distract from shipping something.

### Node 5 (PID Replacement): REAL but premature

The PID replacement idea is genuinely interesting because:
1. PID tuning is a real pain point (billions of PID loops in the world, most hand-tuned)
2. CfC's continuous-time dynamics are a natural fit for control
3. The latency (97ns) is fast enough for any control loop
4. Determinism matters enormously in control

But:
- We have no control output chip (output clamping, rate limiting, anti-windup)
- Control requires a plant model or a real plant to train against
- Stability guarantees for learned controllers are an open research problem
- We'd be competing against established model-predictive control (MPC) approaches

**Honest assessment:** This is a Phase 2 application. Build the monitoring/classification use case first (lower risk, clearer value prop). Then extend to control once we have a deployed monitoring system that proves the temporal dynamics work on real hardware. The monitoring-to-control progression is natural: first you learn to observe the plant, then you learn to control it.

### Node 6 (Keystroke Biometrics): REAL, small, and demonstrable

This is an underappreciated gem. Here's why:

1. **The feature IS variable-dt.** Inter-keystroke timing is the biometric signal. We don't need to argue that variable-dt matters — it's the entire input.
2. **The model is tiny.** Typing patterns probably need hidden_dim=8 at most. That's sub-microsecond inference.
3. **Privacy is built in.** The model runs on-device. Keystrokes never leave the machine. This is a selling point in enterprise security.
4. **No hardware needed for demo.** Any computer has a keyboard. The demo IS the product.
5. **Low regulatory burden.** Biometric authentication is not safety-critical.

**Smallest viable demo:** A C program that records keystroke timings during an enrollment phase, trains (or pre-loads) a CfC model, then continuously scores incoming keystrokes against the enrolled profile. Output: authentication confidence score.

**Hidden assumption:** We need to train the CfC on keystroke data. This means either: (a) we need training code (we don't have in-C training), or (b) we pre-train in Python and export weights. Option (b) is realistic and honest.

**Honest assessment:** This is the best demo application. Low barrier, compelling narrative, uses our genuine differentiator, and can be built end-to-end without any external hardware.

### Node 7 (Predictive Maintenance): REAL, crowded, but has a niche

The PdM market is real ($10B+ and growing) but competitive. The cloud PdM players (Augury, Senseye, SparkCognition) have raised hundreds of millions. We don't compete with them directly.

**Where we win:** Where there is no connectivity. Underground mines. Offshore oil platforms. Remote pipelines. Agricultural equipment in rural areas. Mobile machinery (construction, mining). These are places where cloud-based PdM doesn't work because there's no reliable network.

**The pitch:** "Your sensor node already has an MCU. Our software makes that MCU smart enough to detect faults on its own, with no cloud connection, in 2KB of code."

**Honest assessment:** This is the highest-revenue application but requires domain partnerships. We can't sell directly to mines — we sell to the sensor manufacturers who supply mines. The demo needs to be compelling enough that a sensor OEM wants to integrate it.

### Node 8 (In-Sensor Compute): ENABLER, not application

Node 8 is really a property of Nodes 7, 9, 10, and 11. "In-sensor compute" is how we deploy, not what we solve. The 2,232-byte .text and 97ns latency are the enabling specs, not the value proposition.

**Honest assessment:** Stop talking about "in-sensor compute" as an application. It's the deployment model for every application. Reframe: "Our software turns a $1.50 sensor MCU into a smart sensor that outputs decisions instead of data."

### Node 9 (CAN Bus): REAL but certification-heavy

CAN bus anomaly detection is a real need (UN R155 cybersecurity regulation). Variable-dt is genuinely the differentiator (message timing patterns ARE the signal). But:

- Automotive OEMs move slowly (3-5 year design cycles)
- AUTOSAR compliance is a major integration barrier
- The certification burden (ISO 26262) is heavy

**Honest assessment:** This is a strong application but a hard market to enter. Best approached via an automotive cybersecurity company that needs an embedded anomaly detection engine, not by selling directly to OEMs.

### Node 10 (Medical): HIGHEST VALUE, HIGHEST BARRIER

Everything about medical wearables is correct: CfC is a natural fit, auditability is maximally valuable, the market is enormous. But:

- FDA clearance costs $200K-$2M and takes 6-18 months
- Clinical validation requires patient data under IRB
- You need a 510(k) predicate device or De Novo classification
- Liability insurance for medical devices is expensive

**Honest assessment:** This is a Year 2+ application. The technology fits perfectly but the go-to-market requires capital, regulatory expertise, and clinical partnerships. Don't start here. Build credibility with industrial applications, then bring the proven technology to medical.

### Node 11 (Power Grid): INTERESTING NICHE

Mains frequency monitoring using a microcontroller + CfC is genuinely clever. The hardware is trivial (voltage divider + comparator) and the signal is naturally variable-dt (zero-crossing jitter).

**But:** The power grid monitoring market is dominated by established players (GE, Schweitzer, ABB) and the accuracy requirements for transmission-level monitoring are beyond what a $5 MCU can achieve. Distribution-level monitoring is less demanding but also less funded.

**Honest assessment:** Interesting science project. Not a first product. Could become relevant if smart grid initiatives expand to distribution-level monitoring at scale.

### Node 12 (Code Density): THE ACTUAL INSIGHT

Revisiting this node last, because it reframes everything above.

The user's insight was: "The ultra-edge or L1 cache is where the money ball is." The benchmark proved it: scalar ternary is slower than float for raw compute. The advantage is working set, not ALU speed.

**What this means practically:** Our advantage maximizes when the alternative (a larger model with a framework) exceeds the target's cache/memory budget. On a Cortex-M0 with 8KB SRAM, TFLite Micro *cannot run at all*. We can. That's not a marginal advantage — it's categorical.

**Reframe:** We don't compete with TFLite Micro on Cortex-M4. We enable applications on Cortex-M0 that nobody else can touch. The market is the billions of tiny MCUs that are currently "dumb" because no inference engine fits.

### Node 13 (Drift Detection): REAL and underappreciated

Online distribution shift detection is not just a nice feature — it's a safety requirement. A model that doesn't know when its inputs have drifted outside training distribution is a liability.

The Welford streaming stats in the norm chip give us this almost for free. The computational cost is negligible. The value is disproportionate: "This system monitors its own input distribution and alerts when conditions have changed" is a sentence that makes safety engineers happy.

**Honest assessment:** This should be a standard feature of every deployment, not an application. Include it in every pipeline by default.

---

## Resolved Tensions

### "Variable-dt is our differentiator" vs "Many applications use fixed-rate sensors"
**Resolution:** Variable-dt is THE differentiator for event-driven applications (CAN bus, keystrokes, BLE). For fixed-rate applications (vibration, audio), our differentiator is code density + determinism. Don't force the variable-dt story where it doesn't fit.

### "Auditability is a moat" vs "Auditability doesn't sell"
**Resolution:** Auditability is a wedge, not a product. It gets you through doors (regulatory, procurement) that competitors can't open. But you need a product to carry it through the door.

### "Multi-timescale is elegant" vs "We should ship something simple"
**Resolution:** Ship single-CfC applications first. Multi-timescale is an experiment for after we have working demos. If the single-CfC pipeline doesn't work well enough, multi-timescale won't save it.

### "Medical is highest value" vs "Medical is hardest to enter"
**Resolution:** Build credibility in industrial first. The technology transfers directly to medical. The reverse doesn't work (you can't enter industrial with a medical regulatory strategy).

---

## What I Now Understand

**The wood's grain runs along two axes:**

**Axis 1: Deployment size.** Our advantage is maximum on the smallest targets. Cortex-M0, 8KB SRAM, $0.50 MCUs. This is where nobody else can play. As targets get larger (M4, M7, A-class), our advantage shrinks because the competition (TFLite, CMSIS-NN) fits too.

**Axis 2: Signal type.** Our advantage is maximum for event-driven, irregular-time signals. This is where variable-dt is categorical. For fixed-rate signals, we still have code density and determinism, but those are quantitative advantages rather than categorical ones.

**The sweet spot:** Applications that combine BOTH axes — tiny hardware AND event-driven signals. That's: CAN bus on automotive ECUs, keystroke biometrics on secure elements, BLE proximity on sensor tags, event-camera processing on vision MCUs.

**The broader market:** Applications where code density matters even if the signal is regular — vibration monitoring on sensor-node MCUs, current monitoring on motor controllers. Here we compete on "fits where others don't" rather than "does something others can't."

**Three applications rise to the top:**

1. **Keystroke biometrics** — Best demo. No hardware needed. Variable-dt is the feature. Privacy-preserving. Low regulatory burden. Proves the pipeline end-to-end.

2. **Vibration predictive maintenance** — Biggest market. Uses FFT->CfC pipeline. Targets disconnected environments. Needs a domain partner but the demo is buildable in simulation.

3. **CAN bus anomaly detection** — Best technical fit (variable-dt + auditability + determinism in automotive). Hard market to enter but strong partnership play.

**One architecture experiment to run:**
- Multi-timescale CfC on synthetic data. Quick and cheap. Either validates or kills the hypothesis.

**One infrastructure investment to make:**
- Drift detection (Node 13) should be standard in every pipeline. Build it into the example code.

