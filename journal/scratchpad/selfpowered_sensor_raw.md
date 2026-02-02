# Raw Thoughts: Self-Powered CfC Anomaly Sensor

## Stream of Consciousness

We stumbled into something. We didn't set out to build a self-powered sensor. We set out to build a verified 2-bit computation engine. The ternary constraint was about auditability and correctness. The sparse path was about speed. The enrollment model was about eliminating training infrastructure. The fixed-point path was about deployment on cheap MCUs.

But all of those decisions — each made for independent, defensible reasons — converge on a single point: a neural computation that fits inside the energy budget of the thing it monitors. That convergence is not an accident, but it wasn't planned either. It emerged.

What do we actually have? A CfC cell that runs in 20 ns with zero multiplies on Apple Silicon. A sparse representation that reduces 160 MACs to 31 adds. LUT+lerp activations that are 200x more accurate than polynomial approximations. Three live-validated demos. 230 passing tests. A frozen chip forge.

What don't we have? A fixed-point implementation. A Cortex-M4 or RISC-V cross-compile. A working prototype on actual embedded hardware. A Turing-complete ETM fabric. A rectenna. A TEG integration. A real customer. Revenue.

The gap between "proven on a Mac" and "shipping on a $3 module" is real engineering but not research. The CfC math doesn't change. The sparse structure doesn't change. The enrollment model doesn't change. It's a conversion problem: float to Q15, Apple Silicon to RV32IMAC.

What scares me: the ETM fabric idea. It's compelling but it's vapor right now. We don't have it. We shouldn't let a hypothetical fabric distract from the concrete path. The concrete path is: fixed-point CfC on ESP32-C6 CPU, powered by TEG/piezo, communicating over BLE/Thread. That's buildable today with existing parts.

What excites me: the energy feedback loop. The sensor powers itself from the phenomenon it monitors. A failing bearing vibrates harder AND heats up more — both of which increase harvested power. The sensor gets stronger as the problem gets worse. That's not just convenient, it's a fundamental architectural advantage that no cloud-based or battery-powered system has.

The market question: who buys this? Maintenance teams at manufacturing plants, power plants, oil & gas, data centers, HVAC companies, fleet operators. The pitch is simple: stick it on the bearing, it never needs a battery, it tells you when something's wrong. The alternative is a $2000 vibration analyzer or a guy with a clipboard walking the floor.

The platform question: is this a sensor company, a chip company, a software company, or a licensing play? We have IP at every layer — the CfC cell architecture, the sparse representation, the enrollment method, the fixed-point implementation. Do we sell hardware, firmware, or know-how?

CAN bus is the second market. Automotive ECUs, heavy equipment, fleet management. UN R155 cybersecurity regulations are forcing OEMs to add intrusion detection. Same architecture: enroll the bus pattern, detect anomalies, run on the ECU itself. But the sales cycle is 2-3 years in automotive. Industrial PdM can move in months.

The Thread mesh angle is important. Each sensor relays for its neighbors. A sensor deep in a machine room that barely harvests enough power to transmit can relay through a node with better conditions. The mesh self-organizes. No infrastructure planning. No access points. Just stick sensors on machines and they form a network.

What about structural health monitoring? Bridges, buildings, wind turbines. Same CfC architecture, same enrollment model, but different deployment constraints — outdoor, remote, potentially years between maintenance visits. Solar + vibration harvesting. Longer tau values (structural degradation is slow). This is a natural extension but a different sales motion.

The 268-byte discriminant is the deployment artifact. It's so small it could be transmitted over a single BLE advertisement. You could update the enrolled model over the air with a single packet. No firmware update, no flash write. Just a new discriminant.

I keep coming back to the fact that multiplication is the original sin of neural computing. It's what forces you into GPUs, FPUs, high-power hardware. We excommunicated it. And now we can run on anything.

## Questions Arising

- What's the actual power consumption of the ESP32-C6 when running Q15 integer adds at 160 MHz? Is it really 30 mA or can we clock down?
- Can we clock the C6 down to 40 MHz and still hit 10 kHz sample rates with margin?
- What's the TEG cold-start problem look like? BQ25570 needs 600 mV to cold-start — is deltaT=10C enough?
- How do we handle enrollment on a headless sensor? BLE provisioning from a phone app?
- Is the 268-byte discriminant stable across temperature ranges? PCA coefficients in Q15 — does quantization noise matter?
- Thread vs BLE vs Zigbee: which protocol for which market?
- Patent landscape: is sparse ternary CfC patentable? Is the self-powered anomaly architecture patentable?
- Who are the competitors? Augury, Petasense, Senseye, SKF Enlight — what do they charge? What's their hardware?

## First Instincts

- Start with industrial vibration PdM. It's the straightest line from what we have to revenue.
- Build a fixed-point CfC first, validate it matches float quality.
- Get a working demo on an ESP32-C6 devkit with a MEMS accelerometer. Even without energy harvesting — just prove the compute works on the target hardware.
- Energy harvesting is phase 2 of the hardware. Don't let the self-powered story delay getting the core compute proven on embedded.
- The fabric layer is phase 3 or beyond. It's the endgame optimization, not the launch vehicle.
- License the tech to existing sensor companies rather than becoming a hardware company. They have the sales channels, the certifications, the customer relationships.
