# Insights from Fungible Computation Research

## The Ecosystem

Three related repos under `anjaustin/`:

1. **fungible-computation** - The paper arguing neural/classical computation are interchangeable
2. **flynnconceivable** - Neural network that IS a 6502 CPU (460,928 combinations, 100% accuracy)
3. **trix** - 2-bit ternary neural architecture with emergent routing (different from our yinsen/trix.research)

## The Core Thesis

> "Neural and classical computation are fungible—interchangeable representations of the same underlying computational primitives."

**Evidence:**
- Neural → Classical: FLYNNCONCEIVABLE achieves 100% accuracy on 6502 ops
- Classical → Neural: Spline-6502 compresses 3.7MB neural params to 3,088 bytes
- Routing = Lookup: TriX content-addressable routing is equivalent to spline interval selection

## Key Insights for Yinsen

### 1. Exhaustive Verification is Possible AND Valuable

FLYNNCONCEIVABLE tested **all 460,928 combinations** of 6502 operations. Zero errors.

This validates our approach:
- We exhaustively test logic gates (4/4)
- We exhaustively test full adder (8/8)
- We exhaustively test 8-bit adder (65,536/65,536)
- We exhaustively test 2x2 ternary matvec (81/81)

**Takeaway:** Exhaustive verification at small scale is the gold standard. We're on the right path.

### 2. Neural Networks CAN Be Exact (Not Approximate)

The "neural networks are approximate function approximators" trope is false for digital semantics.

FLYNNCONCEIVABLE proves:
- Neural networks can compute EXACTLY
- The trick is training on ALL possible inputs
- 100% accuracy is achievable, not aspirational

**Takeaway:** Our goal of "auditable neural computation" is not just possible but demonstrated.

### 3. Compression Via Lookup Tables

Spline-6502 achieved **1199× compression** by converting neural weights to lookup tables.

| Organ | Neural | Spline | Compression |
|-------|--------|--------|-------------|
| ALU | 1.7MB | 512B | 3,389× |
| SHIFT | 418KB | 1,536B | 272× |
| LOGIC | 425KB | 256B | 1,660× |
| TOTAL | 3.7MB | 3,088B | 1,199× |

**Takeaway:** Ternary networks at small scale might compile to even simpler representations. A trained ternary CfC could potentially become a state machine or lookup table.

### 4. Routing = Lookup = Computation

From the trix repo:
```
signature = weights.sum(dim=0).sign()
scores = input @ signatures.T
winner = scores.argmax()
```

Content-addressable routing is mathematically equivalent to:
- Looking up a spline interval
- Selecting a case in a switch statement
- Indexing a lookup table

**Takeaway:** Our ternary weights {-1, 0, +1} already enable this. The signature IS the address. The tile IS the function.

### 5. The "Don't Learn What You Can Read" Principle

From trix README:
> Core idea: **Don't learn what you can read.**

Neural networks waste capacity learning things that could be looked up. If a function is enumerable, don't approximate it—just store it.

**Takeaway:** For small-scale Yinsen networks, we might:
- Pre-compute common sub-functions
- Store exact results where feasible
- Use the network for interpolation/generalization only

### 6. Soroban (Thermometer) Encoding for Arithmetic

FLYNNCONCEIVABLE uses "Soroban encoding" for the ALU:
```
Value 37:
Rod 0 (1s):   ●●●●●●●○  = 7
Rod 1 (10s):  ●●●○○○○○  = 3
```

This makes carry propagation visible to the network.

**Takeaway:** Input encoding matters. Ternary weights might work better with certain encodings. Worth experimenting with thermometer encoding for arithmetic tasks.

### 7. Tiles Specialize Without Supervision

From trix:
> Tiles specialize without supervision (92% purity on 6502 ops)

The routing mechanism causes tiles to naturally specialize on different input classes.

**Takeaway:** If we build a ternary routing mechanism (like trix but smaller), tiles might specialize into:
- "Normal state" tile
- "Anomaly detected" tile
- Different temporal pattern tiles

This is emergent behavior, not designed.

### 8. The Closed Loop

The fungible-computation paper demonstrates:
> A spline-based function executes correctly on the neural 6502.

This is turtles all the way down:
- Neural network computes CPU ops
- CPU runs programs that use splines
- Splines are compressed neural networks

**Takeaway:** The boundary between "neural" and "classical" is arbitrary. We can choose whichever representation serves our goals (auditability, size, speed).

## Implications for Yinsen

### Strategic Alignment

Our repos share DNA:
- Both use ternary weights {-1, 0, +1}
- Both aim for exact computation (not approximation)
- Both pursue exhaustive verification
- Both seek compression (ternary = 2 bits)

But different focus:
- trix: transformer FFN replacement, routing mechanisms
- yinsen: CfC for time series, auditability for regulated industries

### Technical Opportunities

1. **Routing via ternary signatures**
   - Our ternary CfC weights already have signatures
   - Could add content-addressable routing like trix
   - "Which hidden state pattern activates which output path?"

2. **Compilation to lookup tables**
   - A trained ternary CfC might compile to state machine + LUT
   - This would be even more auditable than weights
   - "The network is literally this lookup table"

3. **Thermometer encoding for inputs**
   - Might improve learnability for arithmetic tasks
   - Worth testing on pulse anomaly detection

4. **Emergent specialization**
   - If we have multiple "tiles" or pathways
   - They might naturally specialize on different input classes
   - Could make auditability even cleaner

### What We Can Borrow

| From fungible-computation | Apply to Yinsen |
|---------------------------|-----------------|
| Exhaustive verification methodology | Already doing ✓ |
| Claim of 100% accuracy | Goal for trained networks |
| Spline compilation | Future: compile CfC to LUT |
| Soroban encoding | Experiment with input encoding |
| Routing = computation insight | Consider content-addressable outputs |

### What's Different

| fungible-computation / trix | yinsen |
|----------------------------|--------|
| PyTorch, GPU-focused | Pure C, MCU-focused |
| Transformer FFN | CfC (recurrent) |
| Gradient descent | Evolution (planned) |
| Large scale (512+ d_model) | Small scale (8-64 hidden) |
| General ML | Regulated industries |

## The Bigger Picture

Both projects are exploring the same fundamental question:

> **What is the simplest, most transparent representation of computation?**

- trix answers: "Ternary weights with emergent routing"
- yinsen answers: "Ternary CfC with exhaustive verification"

They're complementary, not competing:
- trix for feedforward / spatial
- yinsen for recurrent / temporal

## Next Steps Informed by This Research

1. **Test if EntroMorph converges** - fungible-computation shows 100% is achievable
2. **Consider routing in CfC** - could add tile specialization
3. **Explore compilation** - trained network → lookup table
4. **Try thermometer encoding** - for arithmetic-heavy tasks
5. **Document the connection** - cite fungible-computation in future papers

## The Quote That Matters

From FLYNNCONCEIVABLE:
> "The neural network IS the CPU. Not simulating. Computing."

For Yinsen:
> "The ternary network IS the decision. Not approximating. Auditing."
