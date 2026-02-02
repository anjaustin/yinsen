# Synthesis: Self-Powered CfC Anomaly Sensor

## The Product

A self-powered, maintenance-free anomaly detection sensor that harvests energy from the machine it monitors and runs a temporal neural network using only addition and subtraction. No batteries, no cloud dependency, no training infrastructure. Deploys with a phone tap.

**Name suggestion:** Yinsen Sentinel (the sensor reference design)

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    YINSEN SENTINEL                               │
│                                                                  │
│  ┌──────────────┐   ┌────────────┐   ┌───────────────────────┐  │
│  │ Energy        │   │ Sensing    │   │ Compute + Comms       │  │
│  │               │   │            │   │                       │  │
│  │ TEG (30x30mm) │──▶│ MEMS Accel │──▶│ ESP32-C6              │  │
│  │   +           │   │ (duty-     │   │                       │  │
│  │ BQ25570 PMIC  │   │  cycled)   │   │  CFC_CELL_SPARSE_Q15 │  │
│  │   +           │   │            │   │  PCA discriminant     │  │
│  │ 100uF supercap│   │ Optional:  │   │  Thread/BLE mesh      │  │
│  │               │   │ Temp sense │   │                       │  │
│  └──────────────┘   └────────────┘   └───────────────────────┘  │
│                                                                  │
│  Total BOM: $15 (volume)                                         │
│  Size target: 40x40x15mm (matchbox)                              │
│  Weight: <20g                                                    │
│  Power budget: <500 uW avg (TEG provides 0.5-3 mW)              │
│  Deployment: phone app → BLE → 268-byte discriminant             │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

## Implementation Roadmap

### Phase 1: Fixed-Point CfC (WEEKS 1-3)

The critical gate. Everything depends on this.

**Deliverables:**
- `CFC_CELL_SPARSE_Q15` — Q15 fixed-point sparse CfC cell
- Q15 LUT tables (sigmoid, tanh) — 512 bytes each, integer index + integer lerp
- Q15 mixer using shifts instead of multiplies where possible, integer multiply (RV32 M extension) where not
- Validation: bit-level comparison against float CFC_CELL_SPARSE across all 3 demo domains (ISS, seismic, keystroke)

**Success criteria:**
- [ ] Anomaly detection quality within 5% of float path (measured by ROC AUC)
- [ ] Zero floating-point operations in the hot path
- [ ] Compiles and runs on RV32IMAC (ESP-IDF toolchain)
- [ ] Cycle count measured on ESP32-C6 devkit

**Files to create:**
- `include/chips/cfc_cell_q15.h` — Q15 sparse CfC cell
- `include/chips/activation_q15.h` — Q15 LUT sigmoid/tanh
- `test/test_q15.c` — comparison tests against float path

### Phase 2: ESP32-C6 Proof of Life (WEEKS 3-5)

Prove the compute runs on the target hardware. No energy harvesting yet.

**Deliverables:**
- ESP32-C6-DevKitC-1 running CFC_CELL_SPARSE_Q15
- MEMS accelerometer (LIS2DW12 or ADXL355) on SPI/I2C
- Real-time vibration anomaly detection on a small motor or fan
- Cycle count and power measurement (INA219 on supply rail)
- Duty-cycled accelerometer validation (10ms on / 90ms off)

**Hardware BOM (dev phase):**
- ESP32-C6-DevKitC-1: $10
- LIS2DW12 breakout (ultra-low-power) or ADXL355 breakout (low-noise): $15
- INA219 current sensor breakout: $8
- Small vibration motor or desk fan as test target: $10

**Success criteria:**
- [ ] Detects simulated bearing degradation (increasing vibration amplitude)
- [ ] Total system power measured and documented
- [ ] Duty-cycled sensing validated (no quality loss vs continuous)
- [ ] Enrollment via serial/BLE from laptop

### Phase 3: Energy Harvesting Integration (WEEKS 5-8)

Add self-powering. The sensor cuts the cord.

**Deliverables:**
- TEG module (Marlow TG12-2.5-01L or equivalent) integrated with BQ25570
- Cold-start solution validated (LTC3108 or coin cell bootstrap)
- Supercap sizing validated for BLE burst transmit
- Thermal interface design (mounting clip + thermal pad)
- Continuous operation demonstrated on a running machine (motor, pump, or compressor)

**Success criteria:**
- [ ] Sensor starts from cold (zero stored energy) on deltaT >= 15C
- [ ] Continuous operation for 72+ hours with no external power
- [ ] BLE anomaly alerts transmitted successfully
- [ ] Energy balance positive (harvest > consume) documented

### Phase 4: Mesh and Enrollment App (WEEKS 8-12)

The deployment experience.

**Deliverables:**
- Thread mesh firmware (ESP-IDF OpenThread)
- Multi-sensor mesh validated (minimum 5 nodes)
- Phone app (iOS/Android or cross-platform) for:
  - BLE scan and discovery of Sentinel nodes
  - Enrollment: 60-second calibration → discriminant push
  - Status dashboard: per-node anomaly score, battery/energy level
  - Re-enrollment after maintenance
- OTA discriminant update over Thread mesh

**Success criteria:**
- [ ] 5-node mesh self-organizes within 60 seconds
- [ ] Enrollment completes in under 90 seconds
- [ ] Discriminant OTA update under 1 second
- [ ] App shows real-time anomaly scores from all nodes

### Phase 5: Reference Design and SDK (WEEKS 12-16)

The product for customers.

**Deliverables:**
- Open-source reference hardware design (KiCad)
- Firmware SDK (ESP-IDF component)
- Phone app SDK (enrollment + fleet management)
- Documentation: integration guide, API reference, application notes
- 3D-printable enclosure design (IP65 target)

**Success criteria:**
- [ ] A third party can build a working Sentinel from the reference design
- [ ] SDK integrates into existing ESP-IDF projects with `idf.py add-dependency`
- [ ] Three application notes: motor PdM, pump monitoring, HVAC

## Key Decisions

### 1. Q15 Fixed-Point, Not Q8

Q15 (16-bit) gives ~15 bits of fraction, enough for the PCA discriminant without quantization noise issues. Q8 would halve memory but risks quality loss at the discriminant decision boundary. The ESP32-C6 has 512 KB SRAM — we're using <4 KB. Memory isn't the constraint; precision is.

### 2. Duty-Cycled Accelerometer

The accelerometer duty-cycles at 10% (10ms on, 90ms off). CfC's continuous-time dynamics bridge the gaps — this is the tau principle applied to sampling. Average accelerometer power drops from ~200 uW to ~20 uW, bringing total system power to ~40-50 uW. Well within TEG budget.

Validation required: must prove detection quality doesn't degrade. If it does, fall back to continuous sensing and accept ~500 uW total (still within TEG budget on any warm machine, just not on marginal sources).

### 3. Phone-Based Enrollment, Not Self-Enrollment

The sensor never enrolls itself. A human with a phone decides what "normal" means. This prevents the sensor from learning an already-degraded state as baseline. The phone runs the float CfC (plenty of compute), generates the Q15 discriminant, and pushes 268 bytes over BLE.

### 4. Thread for Mesh, BLE for Enrollment

Thread (802.15.4) for always-on mesh communication between sensors and to the gateway. BLE for point-to-point enrollment from the phone app. The ESP32-C6 supports both simultaneously. Thread has better power characteristics for mesh routing than BLE mesh.

### 5. TEG as Primary Harvester, Piezo as Optional

TEG is the default energy source because thermal gradients exist on virtually all industrial equipment and don't require resonance tuning. Piezo/EM harvesters are application-specific (must match vibration frequency). The reference design includes a TEG. Piezo is a documented add-on for high-vibration applications.

### 6. Reference Design + SDK, Not a Hardware Product

We are not a sensor company. We are the architecture. The reference design proves the concept. The SDK lets others build products. Value capture through SDK license (commercial use), support contracts, and a premium cloud dashboard for fleet analytics.

## Target Markets (Ordered by Time-to-Revenue)

### Market 1: General Industrial PdM (Months 1-6)
- Motors, pumps, fans, compressors in manufacturing
- Non-hazardous environments (FCC/CE only)
- Entry: reference design + SDK for system integrators
- Price point: $15-30/sensor (BOM), $0 subscription for basic detection

### Market 2: Data Center Cooling (Months 3-9)
- CRAC/CRAH units, cooling fans, pumps
- High density of identical machines (easy enrollment)
- High cost of downtime ($8K-300K/hour)
- Thread mesh maps naturally to row/aisle topology

### Market 3: CAN Bus Anomaly Detection (Months 6-12)
- Same CfC architecture, different input (CAN message timing)
- Fleet management, heavy equipment
- UN R155 compliance driver
- Higher margin, longer sales cycle

### Market 4: Structural Health Monitoring (Months 12-18)
- Bridges, buildings, wind turbines
- Longer tau, lower sample rates
- Solar + vibration harvesting (no TEG — outdoor, no hot surfaces)
- Government/infrastructure customers, long procurement cycles

## Bill of Materials: Sentinel v1 Reference Design

| Component | Part Number | Function | Unit Cost |
|---|---|---|---|
| MCU | ESP32-C6-WROOM-1-N4 | Compute + Thread + BLE | $2.00 |
| Accelerometer | LIS2DW12TR | 3-axis MEMS, ultra-low-power | $1.50 |
| TEG | Marlow TG12-2.5-01L | Thermal energy harvesting | $8.00 |
| PMIC | TI BQ25570RGRR | Boost charger + buck regulator | $3.50 |
| Cold-start | LTC3108EMS (or coin cell bootstrap) | TEG cold-start | $2.00 |
| Supercap | 100 uF 3.3V MLCC bank | Burst TX energy storage | $1.00 |
| Thermal pad | Graphite TIM | TEG-to-machine interface | $0.50 |
| Heat sink | Aluminum finned, 30x30mm | TEG cold side | $1.50 |
| Passives + PCB | Resistors, caps, 4-layer PCB | — | $2.50 |
| Enclosure | 3D printed or injection molded | IP65 housing | $1.50 |
| **Total** | | | **$24.00** |
| **At 10K volume** | | | **~$14-16** |

## Risk Register

| Risk | Severity | Mitigation |
|---|---|---|
| Q15 quality loss exceeds 5% | **HIGH** — blocks everything | Validate early. Fall back to Q24 if needed. |
| Accelerometer power dominates budget | MEDIUM | Duty-cycle validation. Fall back to continuous (still viable on TEG). |
| TEG cold-start at low deltaT | MEDIUM | LTC3108 backup. Coin cell bootstrap option. Document minimum deltaT. |
| Thread mesh instability | LOW | Mature ESP-IDF OpenThread stack. BLE fallback for single-sensor deployments. |
| Patent conflict with existing art | MEDIUM | Prior art search before public launch. Focus on system claims, not component claims. |
| Competitor copies architecture | LOW | 18-month head start. Architecture requires ternary CfC expertise — not trivially replicable. |

## The One-Sentence Pitch

A $15 maintenance-free sensor that powers itself from the machine it monitors and detects failures before they happen — no batteries, no cloud, no training, no multiplies.

## Next Action

Order an ESP32-C6-DevKitC-1 and a LIS2DW12 breakout board. Write `cfc_cell_q15.h`. The wood is ready to cut.
