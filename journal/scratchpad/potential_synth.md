# Synthesis: Yinsen's Potential

## The One-Liner

**Yinsen: Auditable neural computation for regulated industries.**

## The Thesis

Neural networks are black boxes. This is acceptable when the stakes are low (ad recommendations, photo filters). It's unacceptable when stakes are high (medical diagnosis, autonomous systems, financial decisions).

Yinsen takes a different approach: constrain the network so severely that every computation becomes traceable. Ternary weights mean "this input was added, subtracted, or ignored" - nothing else. A trained Yinsen network isn't just explainable post-hoc; it's auditable by construction.

This isn't for everyone. It's for industries where "trust me, it works" isn't good enough.

---

## Architecture: The Clean Cut

```
┌─────────────────────────────────────────────────────────────────┐
│                        YINSEN STACK                              │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │ AUDIT LAYER                                              │    │
│  │ • Decision trace for each inference                      │    │
│  │ • Weight-to-outcome mapping                              │    │
│  │ • Formal bounds certificate                              │    │
│  └─────────────────────────────────────────────────────────┘    │
│                            ▲                                     │
│                            │                                     │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │ EVOLUTION LAYER (EntroMorph)                             │    │
│  │ • Tournament selection                                   │    │
│  │ • Ternary mutation (flip weights between -1, 0, +1)      │    │
│  │ • No gradients required                                  │    │
│  └─────────────────────────────────────────────────────────┘    │
│                            ▲                                     │
│                            │                                     │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │ NETWORK LAYER (Ternary CfC)                              │    │
│  │ • Ternary weights {-1, 0, +1}                            │    │
│  │ • CfC temporal dynamics                                  │    │
│  │ • Handles irregular time series                          │    │
│  └─────────────────────────────────────────────────────────┘    │
│                            ▲                                     │
│                            │                                     │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │ PRIMITIVE LAYER (Verified)                               │    │
│  │ • Logic gates (PROVEN)                                   │    │
│  │ • Arithmetic (PROVEN)                                    │    │
│  │ • Activations (TESTED)                                   │    │
│  │ • Ternary ops (TESTED)                                   │    │
│  └─────────────────────────────────────────────────────────┘    │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## Minimum Viable Demonstration: "Pulse Anomaly Detector"

### Task Definition

Detect anomalies in a synthetic pulse signal:
- Normal: regular sine wave, period ~1 second
- Anomaly: skipped beat, double beat, or amplitude deviation
- Irregular sampling: inputs arrive at 10-50ms intervals (not fixed)

### Why This Task

1. **Temporal:** Requires memory across time steps (can't solve with feedforward)
2. **Irregular:** CfC's dt handling is exercised
3. **Binary output:** Anomaly or not (simple evaluation)
4. **Auditable:** Can trace "why did it flag this?"
5. **Medical-adjacent:** Resembles heart rate variability detection
6. **Achievable:** Should be solvable with ~100 weights

### Network Specification

```c
// Ternary CfC network for pulse anomaly detection
// Input: 1 (normalized pulse amplitude)
// Hidden: 8 units
// Output: 2 (normal, anomaly) via softmax

TernaryCfCCell cell = {
    .input_size = 1,
    .hidden_size = 8,
    .output_size = 2,
    // Total ternary weights:
    // W_gate: (1+8) * 8 = 72
    // W_cand: (1+8) * 8 = 72
    // W_out:  8 * 2 = 16
    // Total: 160 weights = 40 bytes
};
```

### Evolution Specification

```c
// EntroMorph configuration for pulse task
EntroConfig config = {
    .population_size = 50,
    .generations = 500,
    .mutation_rate = 0.05,     // 5% of weights mutate per generation
    .tournament_size = 5,
    .elitism = 2,              // Top 2 always survive
    .fitness_fn = pulse_anomaly_fitness
};

// Fitness: accuracy on 1000 synthetic sequences
// Sequence: 100 timesteps, 0-5 anomalies randomly placed
```

### Success Criteria

| Criterion | Target | Verification |
|-----------|--------|--------------|
| Evolution converges | Fitness improves over generations | Plot fitness curve |
| Accuracy | >90% on held-out test set | Run 1000 test sequences |
| Determinism | Same network, same input → same output | Test across 3 runs |
| Audit trace | Can explain each anomaly detection | Generate trace document |
| Memory | <100 bytes for network | Measure actual size |
| Export | Standalone C file compiles and runs | gcc + execute |

### Deliverables

1. **`examples/pulse_demo.c`** - Complete evolution + inference demo
2. **`docs/PULSE_AUDIT.md`** - Audit trail for trained network
3. **`build/pulse_network.h`** - Exported trained network (generated)
4. **Test results** - Accuracy, timing, memory measurements

---

## Roadmap

### Phase 1: Prove Viability (2-4 weeks)

| Task | Description | Success |
|------|-------------|---------|
| Test EntroMorph | Run evolution on toy problem | Fitness converges |
| Ternary evolution | Evolve ternary weights specifically | Network learns |
| Pulse demo | Full pipeline on pulse task | >90% accuracy |
| Audit trace | Generate explanation for decisions | Document produced |

### Phase 2: Build Credibility (1-2 months)

| Task | Description | Success |
|------|-------------|---------|
| Document everything | API, theory, examples | Complete docs |
| Open source | Push to GitHub, clean README | Public repo |
| Write paper | "Auditable Neural Networks via Ternary Constraints" | Preprint on arXiv |
| Find prospect | One company with auditability pain | Meeting scheduled |

### Phase 3: Prove Utility (3-6 months)

| Task | Description | Success |
|------|-------------|---------|
| Real dataset | Partner provides actual time series data | Data received |
| Solve their problem | Train network, generate audit | Network deployed |
| Regulatory alignment | Document compliance pathway | Written opinion |
| Case study | Publish results with partner | Public reference |

---

## Key Decisions

### Decision 1: Ternary-First (not Float-then-Quantize)

**Rationale:** Quantization after training loses the audit trail. If we train in ternary, the weights ARE the explanation. No "this weight was 0.73, we rounded to 1" confusion.

### Decision 2: Evolution-Only Training

**Rationale:** Gradient descent on ternary weights requires tricks (straight-through estimator, etc.). Evolution searches the discrete space directly. Simpler, more honest, and the search process itself is auditable.

### Decision 3: CfC over LSTM/GRU

**Rationale:** CfC handles irregular dt natively. Medical/industrial data is irregularly sampled. Also, CfC is less well-known, which helps differentiation.

### Decision 4: Medical-Adjacent Demo (not Medical)

**Rationale:** Actual medical data requires IRB, compliance, etc. Synthetic "pulse-like" signal demonstrates capability without regulatory burden. Real medical application is Phase 3, after proving viability.

### Decision 5: Document First, Then Build

**Rationale:** The audit trail is the product. If we can't document what the network does, we've failed even if it works. Documentation isn't after-the-fact; it's intrinsic.

---

## Anti-Goals

Things we are NOT trying to do:

1. **Beat SOTA on any benchmark** - We're not competing on accuracy
2. **Support arbitrary architectures** - Ternary CfC only (for now)
3. **Be a general ML framework** - This is a specialized tool
4. **Work on huge datasets** - Edge deployment implies small
5. **Appeal to everyone** - Regulated industries are the target

---

## Metrics That Matter

| Metric | Why It Matters |
|--------|----------------|
| **Audit completeness** | Can we trace every decision? |
| **Network size (bytes)** | Can it fit on target hardware? |
| **Evolution generations to converge** | Is training feasible? |
| **Accuracy on task** | Does it actually work? |
| **Time to explain one decision** | Is auditing practical? |

Metrics that DON'T matter (for us):
- Inference latency (it's fast enough)
- FLOPS (not competing on compute)
- Accuracy vs. SOTA (not the game we're playing)

---

## The Headline

When this works, the headline is:

**"The neural network a regulator can read."**

Every weight is +1, -1, or 0. Every decision traces to specific inputs. The network fits in 100 bytes. A nurse, a pilot, or a judge can follow the logic.

That's not a feature. That's a category.

---

## Next Immediate Actions

1. [ ] **Test EntroMorph** - Does evolution converge on ANY task?
2. [ ] **Ternary evolution harness** - Adapt EntroMorph for ternary weights
3. [ ] **Synthetic pulse generator** - Create the demo dataset
4. [ ] **Pulse demo** - Full pipeline: evolve → train → audit → export
5. [ ] **Write PULSE_AUDIT.md** - The audit document for the trained network

When these five items are done, we have a proof of concept. Everything else builds on this foundation.
