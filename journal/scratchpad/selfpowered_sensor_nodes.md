# Nodes of Interest: Self-Powered CfC Anomaly Sensor

## Node 1: The Convergence Was Emergent, Not Designed

Every architectural decision — ternary weights, sparse representation, LUT activations, enrollment-not-training, precomputed decay — was made for its own reason (correctness, speed, accuracy, deployability, simplicity). None were made with "self-powered sensor" in mind. Yet they all converge on exactly the properties needed for one: zero multiplies, sub-microwatt compute, tiny memory, no training infrastructure.

Why it matters: This suggests the architecture found a natural minimum, not a local optimum. Designs that emerge from independent constraints tend to be more robust than designs that are engineered toward a single target.

## Node 2: The Energy Feedback Loop

A degrading machine vibrates harder and runs hotter. Both increase harvested energy. The sensor gets more powerful as the problem gets worse. This is structurally unprecedented — battery-powered and cloud-connected systems degrade under stress (battery drain, network congestion). This one gets stronger.

Why it matters: This isn't a feature. It's an architectural property. It changes the reliability model of the entire system. No competing architecture has this because no competing architecture is cheap enough to run on harvested energy.

## Node 3: The Concrete Path vs. The Fabric Vision

There are two paths forward. The concrete path: fixed-point CfC on ESP32-C6 CPU, TEG/piezo power, BLE/Thread comms. Buildable with off-the-shelf parts today. The visionary path: Turing-complete ETM fabric, 50 parallel neurons, CPU-as-supervisor. The fabric doesn't exist yet.

Tension: The fabric vision is more compelling but more speculative. The concrete path is less exciting but shippable. Pursuing the fabric risks delaying a product that could ship. Ignoring the fabric risks building something that a future fabric makes obsolete.

## Node 4: The Fixed-Point Conversion Is the Critical Gate

Nothing else matters until CFC_CELL_SPARSE works in Q15 integer arithmetic on RV32IMAC. Not energy harvesting. Not Thread mesh. Not the TEG BOM. The entire product thesis collapses if the fixed-point conversion introduces unacceptable quality loss or doesn't actually hit the power target.

Why it matters: This is the one piece of unproven engineering in the stack. Everything else is either validated (CfC math, enrollment, live data) or commodity (ESP32-C6, TEG modules, BQ25570). The Q15 conversion is the remaining risk.

## Node 5: Enrollment UX on a Headless Sensor

The enrollment model is our biggest advantage ("no training needed"), but enrollment still requires a calibration phase. On a laptop, that's trivial. On a headless $15 sensor glued to a bearing, how does the user trigger calibration? How do they know it succeeded? How do they update the discriminant?

Tension: The engineering is elegant but the UX is undefined. A sensor that requires a laptop to enroll defeats half the value proposition. A sensor that self-enrolls risks learning the wrong baseline.

## Node 6: Sensor Company vs. IP Licensing

We can build hardware (high margin per unit, capital-intensive, need certifications, warranty, supply chain). Or we can license the CfC chip IP to existing sensor companies (lower margin, zero hardware risk, faster to market, leverage their channels). Or both — reference design + license.

Tension: Licensing is faster but gives away control. Hardware is slower but captures full value. The reference design + license model is a middle path but requires both engineering (reference design) and business development (license deals).

## Node 7: The BOM Has Three Tiers

Tier 1 ($3): ESP32-C6 module + supercap + passives. Battery or USB powered. Proves the compute. No energy harvesting. Development tool or short-deployment sensor.

Tier 2 ($15): Add TEG + BQ25570 + heat sink. Self-powered on any warm machine. The core product for industrial PdM.

Tier 3 ($25): Add piezo/EM harvester + rectenna + dual-source PMIC. Multi-source harvesting. Maximum reliability. Premium positioning for critical assets.

Why it matters: These tiers map to different customers, different sales motions, and different certification requirements. Tier 1 can ship in months. Tier 2 needs thermal design validation. Tier 3 needs vibration tuning per application.

## Node 8: Thread Mesh Changes the Deployment Model

Thread (802.15.4) enables self-organizing mesh networks. Each sensor is a router. No access points to install. No network planning. You stick sensors on machines and they find each other. A sensor with marginal power budget relays through a neighbor with surplus.

Why it matters: This eliminates the largest deployment cost in industrial IoT — network infrastructure. Existing wireless sensor systems (ISA100, WirelessHART) require dedicated gateways and network planning. Thread makes the sensors the network.

## Node 9: The 268-Byte Discriminant Is a Deployment Primitive

The PCA discriminant is so small (268 bytes) it fits in a single BLE advertisement (max 254 bytes in extended advertising). You can update the enrolled model over the air with one packet. No firmware update. No flash write. No reboot. Just overwrite 268 bytes of RAM.

Why it matters: This enables remote re-enrollment. A technician with a phone app walks the floor, taps each sensor, and it re-enrolls to current machine state. Or: a gateway pushes new discriminants over Thread after a maintenance event changes the machine's baseline.

## Node 10: The Tau Principle Selects the Market

CfC's time constant (tau) must match the temporal structure of the signal. Fast tau (ms) for vibration. Medium tau (seconds) for thermal drift. Slow tau (minutes-hours) for structural degradation. The tau value determines which signals the sensor can detect.

Tension with Node 7: A single hardware platform can serve multiple markets by changing tau and enrollment. But each market requires different domain knowledge to set tau correctly. "Universal sensor" is a sales story. "Sensor tuned to your machine" is an engineering reality.

## Node 11: Regulatory and Certification Landscape

Industrial sensors need CE marking (EU), FCC Part 15 (US), potentially ATEX/IECEx for hazardous environments (oil & gas, chemical plants). The ESP32-C6 module is pre-certified for FCC/CE/IC for radio. The harvesting and sensor circuitry need additional certification. ATEX is expensive and slow ($50-100K, 6-12 months).

Why it matters: ATEX locks out oil & gas (a huge PdM market) until certification is done. Non-hazardous industrial (manufacturing, data centers, HVAC) can ship with just FCC/CE. Market entry strategy must account for this.

## Node 12: The Competition Is Cloud-Dependent

Augury, Petasense, Senseye (now Siemens), SKF Enlight all require cloud connectivity for their ML models. Their sensors are battery-powered with WiFi or cellular backhaul. Their value is in the cloud analytics platform, not the sensor. Monthly subscription pricing ($50-200/point/year).

Our architecture inverts this: intelligence is on the sensor. Cloud is optional (for dashboarding, fleet aggregation). The sensor works standalone. No subscription required for basic anomaly detection. Cloud adds value but isn't mandatory.

Tension: Subscription revenue is recurring and high-margin. Hardware revenue is one-time. By making the sensor self-sufficient, we may cannibalize the revenue model that funds the competitors.

## Node 13: The Accelerometer Selection Matters

The MEMS accelerometer is the actual sensing element. Its bandwidth, noise floor, and power consumption constrain the entire system. Low-noise accelerometers (ADXL355: 25 ug/rtHz, 200 uA) vs. ultra-low-power (LIS2DW12: 90 ug/rtHz, 50 uA). The accelerometer might consume more power than the CfC compute.

Why it matters: The story of "20 uW compute" is misleading if the accelerometer draws 200 uA at 3.3V (660 uW). Total system power, not just compute power, determines whether self-powering is viable. The accelerometer choice may be the actual power constraint.
