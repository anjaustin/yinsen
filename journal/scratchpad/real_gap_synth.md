# The Real Gap — SYNTH

## Date: February 2026

## What Emerged

The four gaps collapsed into one: **composition**. Every piece works in isolation. The system has never been assembled end-to-end. The REFLECT revealed something more important: the thing that emerged (enrollment-based temporal anomaly detection) is not what was planned (evolution-trained ternary CfC), and the thing that emerged is the product.

This SYNTH produces an executable spec. Three experiments, one positioning statement, one decision gate. No feature list. Hypotheses with kill criteria.

## Experiment 1: Ternary Quantization of CfC Weights

### Hypothesis
The CfC cell's hidden state dynamics are robust to weight quantization from float32 to ternary ({-1, 0, +1}). Detection quality (discriminant score) will degrade by less than 15% across all three demo domains.

### Method
1. Take the existing hand-initialized float32 weights from each demo (keystroke, ISS, seismic)
2. Quantize each weight to ternary: w > threshold → +1, w < -threshold → -1, else → 0. Sweep threshold in [0.1, 0.3, 0.5, 0.7]
3. Run the identical enrollment + detection pipeline with ternary weights
4. Measure: discriminant score on normal data, detection latency on anomaly data, false positive rate

### Success Criteria
- Discriminant convergence within 15% of float baseline (keystroke >0.76, ISS >0.71, seismic >0.74)
- Detection latency within 2x of float baseline on all anomaly tests
- At least one threshold value works across all three domains

### Kill Criteria
- If NO threshold preserves >50% of discriminant score on ANY domain, ternary CfC is falsified for enrollment-based detection. Document honestly. The float CfC is still the product.

### Implementation
- New file: `experiments/ternary_quantization/quant_probe1.c`
- Uses existing `trit_encoding.h` for quantization, existing `cfc_cell_chip.h` for the cell
- Side-by-side: float CfC vs ternary CfC on identical input streams
- Output: table of (threshold, domain, float_score, ternary_score, degradation%)

### What This Resolves
- Node 1 (Two Yinsens): Proves whether Thing A composes into Thing B
- Node 3 (Quantization Test): The bridge experiment
- Remaining Question 1 from REFLECT: "Does ternary quantization preserve discriminant quality?"

## Experiment 2: Ternary Dot Composition

### Hypothesis
The verified `ternary_dot_chip.h` can replace the float dot products inside the CfC cell's gate computations, producing identical (or acceptably close) hidden state trajectories.

### Dependency
Only meaningful if Experiment 1 passes. If ternary weights don't preserve detection quality, there's nothing to compose.

### Method
1. Modify `cfc_cell_chip.h` (or create a `cfc_cell_ternary_chip.h` variant) to use `ternary_dot()` from `ternary_dot_chip.h` for the W*x and W*h products
2. Weight storage uses packed 2-bit trit format from `trit_encoding.h`
3. Bias and hidden state remain float32 (only weights are ternary)
4. Run all three demos with the composed ternary CfC cell

### Success Criteria
- Output matches Experiment 1 ternary results within floating-point tolerance
- Execution time per step is within 2x of current float CfC (currently 58-79 ns/channel)
- Memory for weights is reduced (8 weights packed into 2 bytes vs 32 bytes float)

### What This Resolves
- The full primitive composition: trit_encoding → ternary_dot → CfC cell → enrollment → detection
- Proves the chip forge primitives work together, not just individually
- Every operation in the hot path touches verified ternary code

## Experiment 3: Cross-Compile to Embedded Target

### Hypothesis
The seismic detector (smallest footprint: 1,768 bytes state, ~20KB binary estimate) will cross-compile to ARM Cortex-M4 (STM32F4) with zero code changes and execute within real-time constraints at 100 Hz.

### Method
1. Install `arm-none-eabi-gcc` toolchain
2. Cross-compile `examples/seismic_detector.c` with `-mcpu=cortex-m4 -mfloat-abi=hard -mfpu=fpv4-sp-d16`
3. Measure: binary size, static memory footprint (from .map file)
4. If a dev board is available: flash, run sim mode, measure wall-clock execution time per step

### Success Criteria
- Cross-compiles with zero code changes (pure C99 + math.h)
- Binary < 32KB (fits in smallest STM32F4 flash)
- Static memory < 8KB (fits in smallest STM32F4 SRAM with room for stack)

### What This Resolves
- Node 5 (Deployment Proximity): Proves it, doesn't just claim it
- Remaining Question 2 from REFLECT: "What's the minimum viable target hardware?"
- Produces a concrete artifact: a .bin or .elf file for a real microcontroller

### Fallback
If STM32F4 toolchain is problematic, ESP32 (Xtensa or RISC-V) via ESP-IDF is the backup target. Same hypothesis, different toolchain.

## The Positioning Statement

Based on what actually emerged, not what was planned.

### For technical audiences:
**"Enrollment-based temporal anomaly detection. 1,768 bytes. No training. No runtime. One function call."**

The CfC cell is a temporal feature extractor that builds distinguishable hidden-state trajectories from streaming sensor data. Enrollment learns what "normal" looks like (268-byte PCA discriminant). Deviation is scored in real-time. Validated on live ISS telemetry, seismic waveforms, and keystroke dynamics. 58-79 ns per channel per step. Entire state fits in L1 cache.

### For regulated industry audiences:
**"The anomaly detector a regulator can audit."**

Every weight is +1, -1, or 0. The decision boundary is a mean vector and 5 principal components — printable on one page. No black box. No opaque model file. Falsification record included: here is every way we tried to break it, and what survived.

### What this is NOT:
- Not a general-purpose ML framework
- Not a replacement for PyTorch/TensorFlow
- Not "AI at the edge" (marketing term for compressed large models)
- Not BitNet (that's 1.58-bit approximation; this is true 2-bit with exhaustive verification)

### What this IS:
- A specific architecture (CfC + enrollment) for a specific problem (temporal anomaly detection) at a specific scale (L1-cache-resident, single-channel to 8-channel)
- Verified from transistor-equivalent (2-bit trit operations) up through application (live ISS/seismic/keystroke)
- Designed to be read, not just run

## The Decision Gate

After Experiments 1-3, there is a binary decision:

### If ternary quantization PASSES (Experiment 1 success):
- Compose the full ternary stack (Experiment 2)
- The product is: **verified ternary CfC enrollment-based anomaly detection**
- The ternary verification story and the CfC detection story become one story
- The "regulator can read every weight" claim is proven end-to-end
- Next: embedded deployment proof, then find a customer

### If ternary quantization FAILS (Experiment 1 kill criteria hit):
- Document honestly. Archive the probe.
- The product is: **float CfC enrollment-based anomaly detection**
- The ternary primitives remain valid for other uses (future larger networks, batch processing via NEON/Metal kernels)
- The CfC demos ship as-is with float weights
- The "regulator can read every weight" claim requires a different approach (weight visualization, not ternary)
- Next: embedded deployment with float CfC, then find a customer

Either path leads forward. Neither is a dead end. The experiment determines which story is true.

## Execution Order

1. **Experiment 1** (Ternary Quantization) — do first, gates everything else
2. **Decision Gate** — evaluate results honestly
3. **Experiment 2** (Composition) — only if gate passes
4. **Experiment 3** (Cross-compile) — independent of gate, can run in parallel with 1
5. **Customer discovery** — after at least one deployment proof exists

## What the Previous Roadmap Got Wrong

The throughline_synth roadmap was: evolve → export → deploy → verify. Linear. Feature-driven.

The actual path was: build primitives → build cell → discover enrollment → connect live data → discover tau principle → discover composition gap. Nonlinear. Emergence-driven.

This SYNTH doesn't propose a roadmap. It proposes three falsifiable experiments with kill criteria and a decision gate. If the experiments pass, the path forward is clear. If they fail, the failures tell us what to do instead.

The LMM worked. The wood split where it wanted to split. Now we test whether the grain holds all the way through.

## Remaining Questions Not Addressed Here

- **Who is the customer?** (Cannot be answered from inside the repo. Requires going outside. But going outside requires having something to show — which these experiments produce.)
- **Is "enrollment-based temporal anomaly detection" a recognized market category?** (Probably not. Closest: "online anomaly detection" or "streaming anomaly detection." The enrollment framing is ours.)
- **What about EntroMorph?** (Evolution-based training was falsified for this application. Enrollment replaced it. EntroMorph may still be valid for other tasks — classification, control — but that's a different LMM.)
- **What about the NEON/Metal kernels?** (They serve batch ternary computation for larger networks. Irrelevant to the current CfC demos but strategically important if ternary networks scale up. Not addressed here because they're a different product at a different timescale.)
