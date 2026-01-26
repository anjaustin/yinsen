# Nodes of Interest: Yinsen's Potential

Extracted from RAW phase. These are observations and tensions, not solutions.

---

## Node 1: Ternary as Computational Model (not compression)

Ternary weights aren't quantization - they're a fundamentally different computational model. Forward pass becomes conditional accumulation, not multiplication.

**Why it matters:** This reframes what we're building. Not "compressed neural nets" but "a different kind of computation."

**Tension:** Different doesn't mean better. Need to prove utility.

---

## Node 2: Human-Executable Networks

A ternary network can be executed with pencil and paper. Weights are +1, -1, 0. You're just adding and subtracting.

**Why it matters:** This is radical for interpretability. Not "explain the model" but "trace the computation by hand."

**Tension:** Human-executable at small scale. Does it scale to useful sizes?

---

## Node 3: The Regulatory Gap

Regulated industries (medical, avionics, finance, automotive) face a dilemma: ML is powerful but unauditable. Current solutions are either "don't use ML" or "use ML and hope regulators don't ask hard questions."

**Why it matters:** There's a market gap for auditable ML. Not "explainable AI" post-hoc, but "verifiable by construction."

**Tension:** Is the gap real or theoretical? Who's actually blocked by this?

---

## Node 4: The Ternary + CfC + Evolution Triad

Three components that reinforce each other:
- **Ternary:** Discrete weights, auditable, no FPU needed
- **CfC:** Handles irregular time series, has time constant
- **Evolution:** Doesn't need gradients, works on discrete search spaces

**Why it matters:** The combination addresses problems none of the pieces solve alone.

**Tension:** We have no proof the triad works. EntroMorph is untested. End-to-end pipeline doesn't exist.

---

## Node 5: Edge Deployment Sweet Spot

Ternary networks fit in kilobytes. No FPU required. Deterministic execution. This is perfect for tiny MCUs, battery-powered devices, hard real-time systems.

**Why it matters:** Opens deployment scenarios closed to conventional ML.

**Tension:** Edge deployment is competitive. What's the unfair advantage beyond size?

---

## Node 6: Medical Wearables as Beachhead

Medical wearables tick all boxes:
- Irregular sampling (battery saving)
- Interpretability required (FDA)
- Resource constrained (battery, MCU)
- High stakes (health decisions)

**Why it matters:** Concrete vertical with real pain points.

**Tension:** Medical is heavily regulated. Long sales cycles. Need clinical validation.

---

## Node 7: The "Nothing Works Yet" Problem

We have primitives. We have no trained network. No benchmark. No proof anything useful can be learned with ternary weights.

**Why it matters:** All the potential is theoretical until we demonstrate learning.

**Tension:** What's the minimum viable demonstration?

---

## Node 8: Formal Verification Possibility

Ternary networks have finite state spaces. A small network could potentially be formally verified (not just tested, but proven). This is impossible for float32 networks.

**Why it matters:** "Formally verified neural network" would be unprecedented for certification.

**Tension:** State space explodes combinatorially. Only viable for tiny networks?

---

## Node 9: Orthogonal Positioning

The AI world is obsessed with scale: bigger transformers, more parameters, more compute. We're going the opposite direction: smaller, simpler, auditable.

**Why it matters:** Not competing with PyTorch. Playing a different game entirely.

**Tension:** Is "orthogonal" actually "irrelevant"? The market might not want what we're building.

---

## Node 10: The Pipeline Gap

We have: primitives, tests, ternary weights, CfC cells.
We're missing: training loop, evolution harness, export to deployable C, benchmark tasks.

**Why it matters:** The gap between "research code" and "useful tool" is the pipeline.

**Tension:** Building the pipeline is significant work. Is it the right next step?

---

## Node 11: Research vs. Commercial Value

Two paths:
- **Research:** Publish papers, prove the triad works, academic credibility
- **Commercial:** Solve a real problem, generate revenue, build a company

**Why it matters:** The paths diverge. Research wants novelty. Commercial wants utility.

**Tension:** Which path first? Can they be parallel?

---

## Node 12: The Simplicity Hypothesis

The core bet: simplicity is a feature, not a bug. Ternary's limitations force parsimony. The network that emerges might be more robust than a complex one.

**Why it matters:** If true, this flips the narrative. Not "limited" but "constrained to essentials."

**Tension:** This is a hypothesis. No evidence yet.

---

## Connections Map

```
[Node 1: Computational Model] ──────┐
         │                          │
         ▼                          ▼
[Node 2: Human Executable] ◄──► [Node 8: Formal Verification]
         │                          │
         ▼                          │
[Node 3: Regulatory Gap] ◄──────────┘
         │
         ▼
[Node 6: Medical Wearables] ◄──► [Node 5: Edge Deployment]
         │                          │
         ▼                          │
[Node 4: The Triad] ◄───────────────┘
         │
         ▼
[Node 7: Nothing Works Yet] ──► [Node 10: Pipeline Gap]
         │
         ▼
[Node 11: Research vs Commercial]
         │
         ▼
[Node 12: Simplicity Hypothesis] ◄──► [Node 9: Orthogonal Positioning]
```

---

## The Delta (Boundary Items)

Items at the boundaries between categories - need extra scrutiny:

1. **"Auditable" vs "Interpretable"** - Are these the same? Auditable might mean "can trace execution." Interpretable might mean "can understand why." Different claims.

2. **"Edge" vs "Embedded"** - Edge could mean smartphone. Embedded means bare-metal MCU. Different markets, different requirements.

3. **"Evolution" vs "Training"** - Evolution is search. Training implies gradients. Using "training" for evolution might confuse people.

4. **"Verified" vs "Certified"** - We say verified (tests pass). Certification is a regulatory process. Don't conflate them.
