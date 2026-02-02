# Synthesis: ISS Telemetry v2 — Pre-Scaled Inputs + Real WebSocket Data

## Architecture (v2)

```
PHASE 0: CALIBRATION (offline, one-time, ~3 hours)
─────────────────────────────────────────────────────
  Lightstreamer ──► iss_websocket.py --calibrate
                         │
                         ├── Log raw (ch_id, timestamp, value) to CSV
                         ├── Verify which channels are live
                         ├── Compute per-channel: min, max, mean, std, update_rate
                         └── Output: calibration.json (or header line in CSV)

PHASE 1: ENROLLMENT (offline, seconds)
─────────────────────────────────────────────────────
  Logged CSV ──► iss_telemetry --enroll calibration.csv
                         │
                         ├── Apply frozen pre-scaling: x_s = (x - mean) / std
                         ├── Run CfC with variable dt from timestamps
                         ├── Collect hidden states after warmup
                         ├── Learn discriminant per channel (mean + PCA)
                         ├── Learn 3-sigma baseline per channel (for comparison)
                         └── Output: enrollment.bin (discriminants + scaling params)

PHASE 2: LIVE DETECTION (real-time, continuous)
─────────────────────────────────────────────────────
  Lightstreamer ──► iss_websocket.py ──► iss_telemetry --live enrollment.bin
                                              │
                                              ├── Apply frozen pre-scaling
                                              ├── CfC step with real variable dt
                                              ├── Score per sample (not end-of-sequence)
                                              ├── Cross-channel aggregate
                                              ├── 3-sigma baseline comparison
                                              └── stdout: scores + 3-sigma + verdict
```

## Key Decisions

1. **Frozen pre-scaling from calibration** because probe2 proved inputs must reach the CfC's useful range (Node 1) and keystroke probes proved fixed transforms are safe while online normalization is not (Node 2).

2. **Three-phase operation (calibrate → enroll → detect)** because you need statistics before you can scale, and you need scaled data before you can enroll (Node 10). Separating the phases eliminates the chicken-and-egg problem.

3. **Variable dt from real timestamps** because constant-dt testing disabled the CfC's core temporal mechanism. The tau ablation test must be re-run under variable dt to determine if tau matters (Node 3).

4. **Per-sample scoring** because transient detection requires examining the score at the moment of the anomaly, not at the end of the sequence (Node 4).

5. **3-sigma baseline at every comparison point** because we must demonstrate CfC advantage over the simplest possible detector or honestly report that no advantage exists (Node 6).

6. **Channel verification before enrollment** because Lightstreamer may not serve all expected parameters (Node 7).

## Per-Channel Calibration Structure

```c
typedef struct {
    float input_mean;     /* 4 bytes — calibration mean */
    float input_std;      /* 4 bytes — calibration std */
    float input_min;      /* 4 bytes — observed minimum */
    float input_max;      /* 4 bytes — observed maximum */
    float update_rate;    /* 4 bytes — mean updates/sec */
    int n_calibration;    /* 4 bytes — samples used for calibration */
    int live;             /* 4 bytes — did this channel produce data? */
} ChannelCalibration;     /* 28 bytes per channel */
```

Added to the `TelemetryChannel` struct, the pre-scaling step becomes:

```c
static float prescale(float raw_value, const ChannelCalibration *cal) {
    return (raw_value - cal->input_mean) / (cal->input_std + 1e-8f);
}
```

This maps the channel's operating range to roughly [-3, +3], centered on zero. The CfC's sigmoid and tanh will exercise their full nonlinear range.

## Modified CfC Input Path

```c
/* BEFORE (v1 — raw input, degenerate for CMG): */
float input[2] = { raw_value, dt };

/* AFTER (v2 — pre-scaled input): */
float input[2] = { prescale(raw_value, &ch->cal), dt / ch->cal.update_rate };
```

The dt is also normalized by the channel's typical update rate. This maps dt to roughly [0.5, 2.0] for normal update intervals, which keeps it in the CfC's useful range for the tau computation.

## Calibration Script Additions (`scripts/iss_websocket.py`)

Add `--calibrate` mode:

```
python scripts/iss_websocket.py --calibrate --duration 3h --output calibration.csv
```

Outputs:
1. Raw data CSV: `channel_id,timestamp,value` (every sample for 3 hours)
2. Summary header or JSON with per-channel statistics

The C processor adds `--enroll <calibration.csv>` mode that reads the CSV, computes pre-scaling parameters, runs CfC enrollment, and saves the frozen state.

## Falsification Probes for v2

Re-run ALL probes from probe1 and probe2 with pre-scaling enabled:

### Probe 3: Pre-Scaling Validation

1. **CMG hidden state variance**: Must increase from 0.0007 to at least 0.01 (15x improvement). If the hidden state is still degenerate after pre-scaling, the CfC architecture genuinely can't represent CMG dynamics.

2. **CabinP hidden state variance**: Same check. Must increase from 0.0003.

3. **Tau ablation re-test with pre-scaled inputs**: This is the critical test. If ISS tau still doesn't outperform keystroke tau after pre-scaling, then tau tuning genuinely doesn't matter and the claim should be permanently dropped.

4. **Null discriminant re-test**: CMG noise-enrolled scored 0.799 in probe1. After pre-scaling, noise-enrolled should score much lower (real ISS patterns should be discriminable from noise).

5. **Subtle anomaly detection**: Re-run the magnitude sweep at 20%, 10%, 5% with pre-scaling. Pre-scaling should make smaller anomalies detectable because the hidden state now actually moves.

6. **3-sigma comparison**: At each anomaly level, compare CfC detection rate to 3-sigma detection rate. Report honestly.

### Probe 4: Variable-dt Tau Ablation (requires real data)

1. Log 2 orbits of real ISS data via Lightstreamer
2. Replay through CfC with ISS tau, keystroke tau, and constant tau
3. Compare hidden state variance, discriminant quality, and detection sensitivity

If this probe shows no tau differentiation under variable dt with pre-scaled inputs, the CfC's temporal mechanism provides no advantage over a simple RNN for this application. That would be an important negative result.

### Probe 5: Spike Visibility Window

1. Inject a spike at a known time in the pre-scaled data
2. Measure the anomaly score at every sample from spike to recovery
3. Map the "visibility window" — duration in samples and seconds
4. Compare across tau configurations

## Concrete Steps (in order)

### Step 1: Add pre-scaling to simulation (`iss_telemetry.c`)

Add `ChannelCalibration` struct. In simulation mode, compute calibration from the first N samples, then apply pre-scaling for the rest. This lets us test pre-scaling without needing real Lightstreamer data first.

**Success criterion**: CMG hidden state H-Std increases from 0.0007 to >0.01.

### Step 2: Re-run falsification probes with pre-scaling

Create `experiments/iss_probes/iss_probe3.c` that repeats probe1 and probe2 tests with pre-scaled inputs. Report side-by-side: v1 (raw) vs v2 (pre-scaled).

**Success criteria**:
- [ ] CMG H-Std > 0.01
- [ ] Noise-enrolled CMG score < 0.50 (down from 0.799)
- [ ] Tau ablation shows at least 0.05 delta between ISS and keystroke tau
- [ ] 20% CMG anomaly: ISS tau detects >15/20 (currently 20/20 — check if pre-scaling changes anything)
- [ ] Spike visibility: score drops below threshold for at least 1 sample

### Step 3: Add `--calibrate` mode to Python shim

Extend `scripts/iss_websocket.py` to log raw data for calibration. Output per-channel statistics.

**Success criterion**: Successfully connects to Lightstreamer, receives data from at least 4 channels, logs for 10+ minutes.

### Step 4: Calibrate on real ISS data

Run calibration overnight. Collect 2+ orbits of real data. Verify channel coverage.

**Success criterion**: Calibration CSV with >5000 samples per live channel.

### Step 5: Enroll on real ISS data

Replay calibration data through the CfC with pre-scaling. Learn discriminants. Save frozen state.

**Success criterion**: All live channels produce valid discriminants (disc.valid == 1).

### Step 6: Live detection with 3-sigma baseline

Run live detection during a period with known ISS events (CMG desaturation, CDRA cycle, etc.). Report CfC scores alongside 3-sigma baseline.

**Success criterion**: Honest report. If CfC detects events 3-sigma misses, document it. If it doesn't, document that too.

## What This Does NOT Address

- **Trained weights**: The CfC weights are still hand-initialized. Training requires a Python training loop with backprop through the CfC, which is future infrastructure work.
- **FFT pre-processing**: Vibration bearing health monitoring really needs spectral features (bearing fault frequencies). The FFT chip is built but not integrated into the ISS pipeline. This is a future enhancement.
- **Multi-channel correlation learning**: The cross-channel score is a simple count of anomalous channels. Learning which channel combinations are diagnostic (e.g., CMG vibration + bearing temp) requires a higher-level model.
- **Seasonal/long-term drift**: ISS telemetry drifts over months (thermal coating degradation, equipment aging). The frozen calibration will eventually go stale. Re-calibration cadence is not addressed.

## Files to Modify

- `examples/iss_telemetry.c` — Add `ChannelCalibration` struct, pre-scaling in channel_step, calibration computation in simulation mode
- `scripts/iss_websocket.py` — Add `--calibrate` mode with CSV logging and statistics output

## Files to Create

- `experiments/iss_probes/iss_probe3.c` — Pre-scaling validation probe (repeat all probe1/probe2 tests with scaling)

## Success Criteria Summary

- [ ] CMG hidden state H-Std > 0.01 (15x improvement over current 0.0007)
- [ ] Noise-enrolled CMG discriminant scores < 0.50 on real CMG data
- [ ] At least one tau configuration outperforms another under variable dt
- [ ] CfC detects at least one anomaly type that 3-sigma misses (or honestly reports it doesn't)
- [ ] Per-sample scoring catches a spike within 1 tau-window
- [ ] Live Lightstreamer connection produces data from at least 4 channels
- [ ] All existing 204 tests still pass (no regressions)
