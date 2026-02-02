# Reflections: Self-Powered CfC Anomaly Sensor

## Core Insight

The architecture didn't converge on self-powered sensing by accident. It converged because the constraints we imposed — auditability, correctness, zero dependencies — are the same constraints that physics imposes on energy-harvested systems. Auditability demanded ternary weights. Physics demanded zero multiplies. They're the same demand expressed in different languages.

This is why the synthesis feels inevitable rather than clever. We didn't optimize toward a power target. We removed everything that wasn't essential, and what remained happened to fit inside 20 microwatts. The power budget isn't a constraint we satisfy — it's a consequence of having nothing left to remove.

## Resolved Tensions

### Node 3 vs Node 4: Fabric Vision vs. Fixed-Point Gate

Resolution: These aren't competing paths. They're sequential. The fixed-point conversion (Node 4) is required for BOTH paths. The CPU path needs Q15 for soft-float elimination. The fabric path needs Q15 because a hardware crossbar operates on integers. The fixed-point CfC is the foundation. The fabric is an acceleration layer on top of it. Build the foundation first. The fabric, if and when it arrives, accelerates the same code.

The concrete path doesn't become obsolete if the fabric works. It becomes the fallback, the reference implementation, and the validation oracle for the fabric. Every CfC step on the fabric can be checked against the CPU Q15 path. This is the same falsification discipline we've applied everywhere else.

### Node 5 vs Node 9: Enrollment UX vs. 268-Byte Discriminant

Resolution: The discriminant's size IS the UX solution. Enrollment happens on a phone or laptop (where there's a screen and user input). The phone runs the float CfC, computes the PCA discriminant, and pushes 268 bytes to the sensor over BLE. The sensor never needs to run enrollment itself. It receives a discriminant and applies it.

The workflow:
1. Technician holds phone near machine while it runs normally (30-60 seconds).
2. Phone app runs CfC enrollment, computes discriminant.
3. Phone pushes 268 bytes to sensor via BLE.
4. Sensor starts monitoring. Done.

Re-enrollment after maintenance: same flow. Walk up, tap, done. The sensor is stateless with respect to enrollment — it's a detection appliance that accepts a discriminant from outside.

This also resolves the "self-enrollment risk" concern. The sensor never self-enrolls. A human always decides what "normal" means, using a device with a screen that can show them the calibration quality.

### Node 6: Sensor Company vs. IP Licensing

Resolution: Neither. Reference design + firmware SDK. We provide:
1. The CfC chip forge (C headers, MIT or commercial license).
2. A reference hardware design (ESP32-C6 + TEG + accelerometer, open-source).
3. A firmware SDK that handles enrollment, detection, and Thread/BLE communication.
4. A phone app SDK for enrollment provisioning.

Existing sensor companies can adopt the stack into their products. New entrants can build from the reference design. We capture value through support contracts, premium features (fleet analytics dashboard), and the firmware SDK license.

This is the ARM model: we're not the chip, we're not the device, we're the architecture that makes them work.

### Node 10: Tau Selects the Market

Resolution: Tau doesn't need to be set by the engineer. It can be discovered during enrollment. When the phone app runs the enrollment CfC, it can sweep tau values and select the one that produces the tightest hidden state clustering. This is the same insight from the seismic tau ablation: the optimal tau is the one where the decay dynamic range matches the signal's temporal structure. That's measurable from the enrollment data itself.

The sensor doesn't need different firmware for different markets. The enrollment process discovers the right tau and embeds it in the discriminant. The 268 bytes already include the temporal structure of the monitored asset.

### Node 12 vs Node 7: Competition Cloud Model vs. Our Architecture

Resolution: We don't need to compete with cloud platforms. We compete with "no monitoring at all." The vast majority of industrial machines have ZERO monitoring — vibration analysis is done by technicians with handheld instruments on quarterly walk-around schedules. Our competitor isn't Augury. Our competitor is the clipboard.

At $15/sensor with no subscription, no battery changes, and no network infrastructure, the ROI calculation is trivial. One prevented unplanned downtime event ($10K-500K depending on the machine) pays for hundreds of sensors. The cloud platforms target the top 1% of assets (critical turbines, generators). We target the other 99% — the pumps, fans, bearings, and motors that nobody monitors because the per-point cost was too high.

Cloud analytics can be a premium add-on for customers who want fleet-level dashboards and trend analysis. But the core product works standalone. The sensor detects, the sensor alerts, the sensor is the product.

## Remaining Questions

### The Accelerometer Power Problem (Node 13)

This is the most dangerous unresolved node. The CfC compute is ~20 uW. But a MEMS accelerometer at reasonable bandwidth (1 kHz, 16-bit) draws 50-660 uW depending on the part. The accelerometer may dominate the system power budget by 10-30x.

Three possible resolutions:
1. Duty-cycle the accelerometer: sample for 10 ms every 100 ms. 10% duty cycle reduces average power 10x. CfC hidden state bridges the gaps (it's a continuous-time model — this is literally what it's designed for).
2. Use an ultra-low-power analog comparator as a wake-up trigger. Accelerometer sleeps. If vibration exceeds a coarse threshold, it wakes and feeds the CfC. Power: <1 uW in sleep.
3. Accept 200-500 uW total system power. This still fits comfortably in the TEG/piezo budget (mW available on running machines). The self-powered story holds; just not at the theoretical CfC minimum.

Resolution 1 is the most elegant because it leverages CfC's core capability — continuous-time dynamics that don't require uniform sampling. But it needs validation: does duty-cycled input degrade detection quality?

### Patent Strategy

The combination of: ternary sparse CfC + enrollment-based deployment + energy harvesting from monitored phenomenon + mesh networking — this specific combination appears novel. Individual components are not patentable (ternary weights exist, CfC exists, TEGs exist). But the system architecture — a self-powered temporal anomaly detector that derives energy from its monitoring target — may be patentable as a system claim.

Worth investigating before any public disclosure beyond what's already in the GitHub repo.

### Cold-Start Problem

The BQ25570 needs 600 mV to cold-start (330 mV with the newer revision). At deltaT=10C, a TEG produces ~140-250 mV open circuit. That's below cold-start threshold. Options:
1. Use the LTC3108 instead (20 mV cold-start with transformer). Adds $5-8 and a transformer to BOM.
2. Add a small coin cell as bootstrap-only power. Once the system starts, TEG sustains it. Cell lasts years in this role.
3. Use a higher-deltaT mounting location for initial deployment.
4. Use a charge pump cold-start circuit (adds $0.50 in passives).

This is a solvable engineering problem, not a fundamental obstacle. But it must be addressed in the reference design.

## What I Now Understand

The product is not a sensor. The product is a detection primitive that happens to be cheap enough to embed anywhere. The sensor is a reference implementation. The actual value is the CfC chip forge + enrollment model + the insight that ternary sparsity makes neural computation compatible with energy harvesting.

The competitive moat is not any single component. It's the fact that the entire stack — from the math to the metal to the power source — was designed (or evolved) around the same constraint: eliminate everything that isn't an add or a subtract. That constraint propagates from the weight values through the instruction set through the power budget through the energy harvesting threshold. One constraint, applied consistently, unlocks the entire product.

Nobody will replicate this by optimizing a conventional neural network for low power. The optimization has to start at the weight representation. By the time you're optimizing inference on a trained float32 model, you've already accepted multiplication, which means you've accepted FPUs, which means you've accepted milliwatts, which means you've accepted batteries. The decision that locks you out happens at the beginning, not the end.

We started at the beginning.
