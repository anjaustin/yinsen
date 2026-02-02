# Theory: Mathematical Foundations

This document explains the mathematical foundations of Yinsen.

## 1. Logic as Polynomials

Boolean logic gates can be expressed as polynomials that are **exact** for binary inputs {0, 1}.

### The Seven Gates

| Gate | Boolean | Polynomial | Verification |
|------|---------|------------|--------------|
| AND | a ∧ b | `a * b` | 0*0=0, 0*1=0, 1*0=0, 1*1=1 |
| OR | a ∨ b | `a + b - a*b` | 0+0-0=0, 0+1-0=1, 1+0-0=1, 1+1-1=1 |
| NOT | ¬a | `1 - a` | 1-0=1, 1-1=0 |
| XOR | a ⊕ b | `a + b - 2*a*b` | 0+0-0=0, 0+1-0=1, 1+0-0=1, 1+1-2=0 |
| NAND | ¬(a ∧ b) | `1 - a*b` | Derived from NOT(AND) |
| NOR | ¬(a ∨ b) | `1 - a - b + a*b` | Derived from NOT(OR) |
| XNOR | ¬(a ⊕ b) | `1 - a - b + 2*a*b` | Derived from NOT(XOR) |

**Status: PROVEN** - Exhaustively tested, all truth tables verified.

### Why This Works

For binary inputs, these polynomials collapse to exact truth tables because:
- `0 * x = 0` for any x
- `1 * x = x` for any x
- Addition and subtraction in {0, 1} with the polynomial coefficients produce exactly {0, 1} outputs

### Proof: XOR

```
XOR(a, b) = a + b - 2ab

XOR(0, 0) = 0 + 0 - 2(0)(0) = 0 - 0 = 0  ✓
XOR(0, 1) = 0 + 1 - 2(0)(1) = 1 - 0 = 1  ✓
XOR(1, 0) = 1 + 0 - 2(1)(0) = 1 - 0 = 1  ✓
XOR(1, 1) = 1 + 1 - 2(1)(1) = 2 - 2 = 0  ✓
```

## 2. Arithmetic from Logic

### Full Adder

A full adder computes `sum` and `carry` from inputs `a`, `b`, and `carry_in`:

```
sum   = a ⊕ b ⊕ c
carry = (a ∧ b) ∨ ((a ⊕ b) ∧ c)
```

In polynomial form:
```
sum   = (a + b - 2ab) + c - 2(a + b - 2ab)c
carry = ab + (a + b - 2ab)c - ab(a + b - 2ab)c
```

**Status: PROVEN** - All 8 input combinations verified.

### Ripple Carry Adder

Chain 8 full adders to get an 8-bit adder:

```c
for (int i = 0; i < 8; i++) {
    full_adder(a[i], b[i], carry, &sum[i], &carry);
}
```

**Status: PROVEN** - All 65,536 combinations (256 × 256) verified.

## 3. Primitive Operations (Hypothesis)

> **Note:** This section describes an organizational framework, not a proven theorem.

The operations in Yinsen can be categorized into primitives:

| Primitive | Symbol | Used For |
|-----------|--------|----------|
| ADD | + | Accumulation |
| MUL | × | Scaling, gating |
| EXP | e^x | Sigmoid, softmax |
| MAX | max(a,b) | ReLU |
| DIV | ÷ | Normalization (softmax, layer norm) |
| CONST | k | Thresholds, coefficients |

### Observations (Not Proofs)

- Logic gates use only: ADD, MUL, CONST
- Activations use: ADD, MUL, EXP, MAX, CONST
- Normalization requires: DIV

### Open Questions

- Is this set minimal? (Unknown - not proven)
- Is this set complete for neural computation? (Unknown - not proven)
- Is there a formal sense in which these are "primitive"? (No - this is organizational, not mathematical)

## 4. Ternary Weights: The Core Architectural Choice

### Why Ternary?

The name "TriX" derives from **ternary** weights: constraining all network parameters to {-1, 0, +1}. This isn't a quantization trick—it's a fundamental architectural decision that changes the computational model.

### How Ternary Computation Works

A standard neural network computes dot products:
```
y = w₁x₁ + w₂x₂ + w₃x₃ + ... + wₙxₙ
```

With float weights, this requires n multiplications. With ternary weights:
```
y = Σ(xᵢ where wᵢ = +1) - Σ(xᵢ where wᵢ = -1)
```

The operation becomes **conditional accumulation**: add x if weight is +1, subtract if -1, skip if 0. No multiplication required.

### Memory Representation

A ternary value ("trit") needs only 2 bits:
- `00` → 0
- `01` → +1
- `10` → -1
- `11` → reserved (decoded as 0)

This gives 4x compression vs int8 and 16x vs float32:

| Format | Bits/weight | 1000 weights |
|--------|-------------|--------------|
| float32 | 32 | 4000 bytes |
| int8 | 8 | 1000 bytes |
| ternary | 2 | 250 bytes |

### Implications for Verification

Ternary weights enable exhaustive testing at small scales:
- 2x2 weight matrix: 3⁴ = 81 configurations (all tested)
- 3x3 weight matrix: 3⁹ = 19,683 configurations (feasible)
- 4x4 weight matrix: 3¹⁶ ≈ 43M configurations (overnight job)

For large networks, exhaustive testing is impossible. But the discrete nature of ternary weights means:
- Every weight can be audited by inspection
- The set of possible networks is finite (though large)
- Edge cases are enumerable

### Limitations

Ternary weights reduce expressivity:
- Fine gradients cannot be captured
- Some functions may require more neurons to approximate
- Training typically uses gradient estimation techniques

**Status: TESTED** - Ternary primitives verified including exhaustive 2x2 and 4x4 matvec. Ternary CfC cell tested for determinism and stability. Ternary CfC trained on sine (MSE 0.000362) and Lorenz (MSE 0.001490) tasks via STE + Adam.

## 5. CfC: A Gated Recurrence with Time Constants

### What CfC Is

CfC (Closed-form Continuous-time) is a recurrent cell architecture from Hasani et al. (2022). Our implementation computes:

```
h(t) = (1 - gate) * h_prev * decay + gate * candidate
```

Where:
- `gate = sigmoid(W_gate @ [x; h_prev] + b_gate)`
- `candidate = tanh(W_cand @ [x; h_prev] + b_cand)`
- `decay = exp(-dt / tau)`

### Relationship to Other Architectures

CfC resembles a GRU with an explicit time constant:

| Feature | GRU | CfC |
|---------|-----|-----|
| Gating | Yes | Yes |
| Reset gate | Yes | No |
| Time constant | No | Yes (tau) |
| Variable dt | No | Yes |

The key difference is that CfC incorporates `dt` explicitly, making it suitable for irregularly-sampled time series.

### The "Closed-Form" Claim

The literature describes CfC as a "closed-form solution" to a continuous-time ODE. In practice:

- Our implementation is a discrete update rule
- It doesn't require an ODE solver
- Whether this constitutes "solving an ODE in closed form" depends on interpretation

**Status: TESTED** - Determinism and stability verified on single platform. No comparison to ODE solvers. No benchmark against GRU.

## 6. Ternary CfC: Combining Time and Discretization

The ternary CfC cell applies ternary weight constraints to the CfC architecture:

```
h(t) = (1 - gate) * h_prev * decay + gate * candidate
```

Where:
- `gate = sigmoid(ternary_matvec(W_gate, [x; h_prev]) + b_gate)`
- `candidate = tanh(ternary_matvec(W_cand, [x; h_prev]) + b_cand)`
- `decay = exp(-dt / tau)`

### Why This Combination?

1. **Temporal modeling**: CfC handles irregular time series via explicit dt
2. **Auditability**: Ternary weights are inspectable
3. **Compression**: 4.4x measured memory reduction vs float CfC
4. **Determinism**: Integer ops in forward pass (except for activations)

### Activation Tiers (DONE — activation_chip.h)

Activations were the last floating-point dependency. Now resolved with three tiers:

| Tier | Sigmoid Error | Tanh Error | Speed | Dependency |
|---|---|---|---|---|
| PRECISE | exact | exact | ~4.2 ns | libm (expf, tanhf) |
| FAST3 | 8.7e-2 | 2.4e-2 | ~3.6 ns | None (polynomial) |
| LUT+lerp | 4.7e-5 | 3.8e-4 | ~3.4 ns | 2KB read-only tables |

**LUT+lerp is the practical winner.** 200x more accurate than FAST3 for negligible cost. 256 entries per function, linear interpolation between entries. Tables initialized once at startup (`ACTIVATION_LUT_INIT()`), shared read-only across all channels, hot in L1.

Key finding from Probe 2: Cubic splines are SLOWER than precise libm on Apple Silicon due to hardware transcendentals. LUT+lerp avoids this by trading compute for memory.

After 1000 CfC steps, L2 divergence from precise path:
- FAST3: 0.137 (11% drift)
- LUT+lerp: 0.000684 (0.05% drift)

For long-running sensors (ISS telemetry: hours; seismic: continuous), LUT+lerp prevents error accumulation that would shift the discriminant baseline.

### Sparse Ternary Computation

At threshold 0.10, ternary quantization zeroes 81% of weights. Dense GEMM wastes cycles multiplying by zero. The sparse variant stores only nonzero indices:

```
Dense:  pre[j] = bias[j] + Σ(W[k,j] * concat[k])      // 160 MACs
Sparse: pre[j] = bias[j] + Σ(concat[pos]) - Σ(concat[neg])  // 31 adds
```

Key insight: The FPU doesn't care what it multiplies by. `1.0 * x` takes the same cycles as `0.3 * x`. The ternary constraint is on VALUES, not INSTRUCTIONS. But when 81% of values are zero, skipping them entirely is the real win.

Performance ladder (Apple M-series, hidden_dim=8):
- CFC_CELL_GENERIC: 54 ns (160 MACs + libm activations)
- CFC_CELL_LUT: 35 ns (160 MACs + LUT activations + precomputed decay)
- CFC_CELL_SPARSE: 20 ns (31 adds + LUT activations + precomputed decay)

**Status: TESTED** - Determinism, stability (1000 iterations), bounded outputs verified. 4.4x compression measured. Ternary CfC trained on sine (MSE 0.000362, nearly matches float) and Lorenz attractor (MSE 0.001490, 12.69x degradation from float -- the "quantization wall"). Ternary quantization validated in Probe 1 (99% quality at threshold 0.10). CFC_CELL_SPARSE bit-identical to CFC_CELL_LUT.

## 6b. Enrollment-Based Detection

### The Pattern

For temporal anomaly detection, no training is needed. The pipeline:

1. **Calibrate**: Run CfC on sensor data to learn per-channel input scales
2. **Enroll**: Process normal-behavior data, collect hidden state trajectories
3. **Build discriminant**: PCA on hidden state covariance, extract principal components
4. **Detect**: Project new hidden states onto discriminant, measure deviation

CfC acts as a temporal feature extractor. The PCA discriminant acts as a decision layer. The discriminant is 268 bytes, human-readable. No gradient descent, no loss function, no backpropagation.

### Why This Works

CfC's gated recurrence with temporal decay creates a rich hidden state representation that captures temporal dynamics. Normal-behavior data creates a characteristic trajectory in hidden state space. Anomalies push the trajectory into unfamiliar regions. PCA identifies the directions of maximum variance in the normal trajectory, providing a compact discriminant.

### The Tau Principle

Tau differentiation emerges when the decay dynamic range R = max(decay)/min(decay) is commensurate with the signal's temporal structure T = max(timescale)/min(timescale).

- Seismic data: R = 2700x, T = 3000x. R ~ T, so matched tau gives 2.2-2.4x faster detection.
- ISS telemetry: R = 2700x, T ~ 1x (slow sensors, similar timescales). R >> T, so tau doesn't differentiate.

Rule of thumb: if your signal has multi-scale temporal structure, match tau dynamic range to signal dynamic range.

### Discriminant Convergence

Across 3 domains (keystroke, ISS, seismic), the PCA discriminant converges to a Mahalanobis distance of 0.84-0.89. Rule: N_PCS ~ ceil(HIDDEN_DIM * 0.6). For hidden_dim=8, PCA(5) is the sweet spot.

## 7. What's Verified vs. What's Claimed

| Claim | Status | Evidence |
|-------|--------|----------|
| Logic polynomials are exact for {0,1} | **PROVEN** | Exhaustive truth tables |
| Full adder correct | **PROVEN** | 8/8 combinations |
| 8-bit adder correct | **PROVEN** | 65,536/65,536 combinations |
| 2x2 ternary matvec correct | **PROVEN** | 81/81 weight configurations |
| Ternary pack/unpack lossless | **TESTED** | Roundtrip tests |
| Ternary dot product correct | **TESTED** | Property tests |
| 4x memory compression | **TESTED** | 8 trits = 2 bytes (vs 8 bytes int8) |
| Activations correct | **TESTED** | Property tests, single platform |
| CfC is deterministic | **TESTED** | Same-machine repeatability |
| CfC is stable | **TESTED** | 10K iterations without divergence |
| Ternary CfC deterministic | **TESTED** | Same-input same-output |
| Ternary CfC stable | **TESTED** | 1K iterations without divergence |
| Ternary CfC 4.4x compression | **TESTED** | 52 bytes vs 228 bytes (measured) |
| Cross-platform determinism | **UNTESTED** | Claimed, not verified |
| CfC equivalent to ODE solution | **UNTESTED** | No comparison performed |
| Ternary sufficient for useful tasks | **TESTED** | Sine MSE 0.000362, Lorenz MSE 0.001490 |
| LUT+lerp activations accurate to 4.7e-5 | **TESTED** | 256-entry table + linear interpolation |
| Sparse ternary: zero multiplies in GEMM | **TESTED** | CFC_CELL_SPARSE at 20 ns/step |
| Enrollment IS the product for anomaly detection | **VALIDATED** | 3 domains, no training needed |
| Tau principle: R ~ T enables differentiation | **VALIDATED** | Seismic R=2700x, T=3000x |
| Primitives are complete/minimal | **HYPOTHESIS** | Organizational, not mathematical |

## 8. Verification Approach

### Exhaustive Testing

Where feasible, we test every possible input:
- Logic gates: 4 combinations each (2 × 2)
- Full adder: 8 combinations (2 × 2 × 2)
- 8-bit adder: 65,536 combinations (256 × 256)
- 2x2 ternary matvec: 81 weight configurations (3⁴)

### Property Testing

For continuous functions, we verify properties:
- Sigmoid: output in (0, 1), sigmoid(0) = 0.5
- Softmax: outputs sum to 1, all positive
- CfC: determinism, bounded outputs, proper decay
- Ternary CfC: determinism, bounded outputs, stability over time
- Ternary pack/unpack: roundtrip losslessness

### What We Don't Test

- Cross-platform reproducibility (different compilers, architectures)
- Numerical equivalence to reference implementations
- Behavior under different floating-point modes (`-ffast-math`)

## References

1. Hasani et al., "Closed-form Continuous-time Neural Networks" (2022)
2. Liquid.ai research on CfC networks
3. Boolean algebra and polynomial representations
