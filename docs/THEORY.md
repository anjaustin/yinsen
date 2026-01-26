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

## 4. CfC: A Gated Recurrence with Time Constants

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

## 5. What's Verified vs. What's Claimed

| Claim | Status | Evidence |
|-------|--------|----------|
| Logic polynomials are exact for {0,1} | **PROVEN** | Exhaustive truth tables |
| Full adder correct | **PROVEN** | 8/8 combinations |
| 8-bit adder correct | **PROVEN** | 65,536/65,536 combinations |
| Activations correct | **TESTED** | Property tests, single platform |
| CfC is deterministic | **TESTED** | Same-machine repeatability |
| CfC is stable | **TESTED** | 10K iterations without divergence |
| Cross-platform determinism | **UNTESTED** | Claimed, not verified |
| CfC equivalent to ODE solution | **UNTESTED** | No comparison performed |
| Primitives are complete/minimal | **HYPOTHESIS** | Organizational, not mathematical |

## 6. Verification Approach

### Exhaustive Testing

Where feasible, we test every possible input:
- Logic gates: 4 combinations each (2 × 2)
- Full adder: 8 combinations (2 × 2 × 2)
- 8-bit adder: 65,536 combinations (256 × 256)

### Property Testing

For continuous functions, we verify properties:
- Sigmoid: output in (0, 1), sigmoid(0) = 0.5
- Softmax: outputs sum to 1, all positive
- CfC: determinism, bounded outputs, proper decay

### What We Don't Test

- Cross-platform reproducibility (different compilers, architectures)
- Numerical equivalence to reference implementations
- Behavior under different floating-point modes (`-ffast-math`)

## References

1. Hasani et al., "Closed-form Continuous-time Neural Networks" (2022)
2. Liquid.ai research on CfC networks
3. Boolean algebra and polynomial representations
