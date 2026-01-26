# Theory: Frozen Computation

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

This is exact for binary inputs and produces the correct truth table for all 8 input combinations.

### Ripple Carry Adder

Chain 8 full adders to get an 8-bit adder:

```c
for (int i = 0; i < 8; i++) {
    full_adder(a[i], b[i], carry, &sum[i], &carry);
}
```

**Verified:** All 65,536 combinations of 8-bit addition (256 × 256) produce correct results.

## 3. The 5 Primes

All operations in Yinsen derive from five primitive operations:

| Prime | Symbol | Description |
|-------|--------|-------------|
| ADD | + | Addition |
| MUL | × | Multiplication |
| EXP | e^x | Exponential |
| MAX | max(a,b) | Maximum (for ReLU) |
| CONST | k | Constants (0, 1, -1, 2, etc.) |

### Derivations

- **Logic gates**: ADD, MUL, CONST only
- **Sigmoid**: `1 / (1 + exp(-x))` = ADD, MUL, EXP, CONST
- **Tanh**: `(exp(x) - exp(-x)) / (exp(x) + exp(-x))` = ADD, MUL, EXP, CONST
- **ReLU**: `max(0, x)` = MAX, CONST
- **Softmax**: `exp(x_i) / sum(exp(x_j))` = ADD, MUL, EXP

## 4. CfC: Closed-form Continuous-time Networks

### The Problem with ODEs

Traditional continuous-time neural networks require solving:
```
dh/dt = f(h, x, t)
```

This requires numerical integration (Euler, RK4, etc.), which is:
- Iterative (many steps)
- Approximate
- Non-deterministic across platforms

### The CfC Solution

CfC networks have a closed-form solution:

```
h(t) = (1 - gate) * h_prev * decay + gate * candidate
```

Where:
- `gate = sigmoid(W_gate @ [x; h_prev] + b_gate)`
- `candidate = tanh(W_cand @ [x; h_prev] + b_cand)`
- `decay = exp(-dt / tau)`

### Why This Works

The key insight is that with gated updates and exponential decay, the ODE:
```
dh/dt = -h/tau + gate * (candidate - h)
```

Has an analytical solution that can be computed in one step regardless of `dt`.

### Properties

1. **Deterministic**: Same inputs always produce same outputs
2. **Arbitrary dt**: Works for any time step, not just small ones
3. **No iteration**: Single forward pass, no solver loops
4. **Composable**: Built from the 5 Primes

## 5. Frozen vs. Learned

### What is Frozen

- Mathematical operations (XOR = a + b - 2ab)
- Activation functions (sigmoid, tanh, etc.)
- The CfC update equation structure

### What is Learned

- Weight matrices (W_gate, W_cand, W_out)
- Biases (b_gate, b_cand, b_out)
- Time constants (tau)

### The Philosophy

> "The shapes are frozen. The routing is learned."

The mathematical structure is immutable truth. Only the parameters (weights) that route signals through this structure are learned or evolved.

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

### Numerical Stability

We test edge cases:
- Large inputs to softmax (should not overflow)
- Many iterations of CfC (should not diverge)
- Zero inputs (should produce reasonable outputs)

## References

1. Hasani et al., "Closed-form Continuous-time Neural Networks" (2022)
2. Liquid.ai research on CfC networks
3. Boolean algebra and polynomial representations
