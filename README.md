# Yinsen

Verified frozen computation primitives for deterministic neural networks.

## What is this?

Yinsen is a minimal C library implementing:

1. **Frozen Logic Shapes** - Boolean operations as polynomials (XOR, AND, OR, etc.)
2. **Arithmetic Shapes** - Full adders and ripple carry adders from logic primitives
3. **Neural Network Shapes** - Activations, matrix operations, normalization
4. **CfC Networks** - Closed-form Continuous-time neural networks
5. **EntroMorph** - Evolutionary engine for CfC networks

Everything is header-only, dependency-free (except libc), and verified by tests.

## Quick Start

```bash
# Build and run tests
make test

# Build and run examples
make examples
./build/hello_xor
```

## The Core Idea

Logic gates can be expressed as polynomials that are exact for binary inputs:

```c
// XOR: a + b - 2ab
float xor_result = yinsen_xor(1.0f, 0.0f);  // Returns 1.0

// AND: a * b
float and_result = yinsen_and(1.0f, 1.0f);  // Returns 1.0

// OR: a + b - ab
float or_result = yinsen_or(0.0f, 1.0f);    // Returns 1.0
```

From these primitives, we build arithmetic (full adders, ripple carry adders) and eventually neural networks.

## Project Structure

```
yinsen/
├── include/
│   ├── apu.h          # Logic shapes + arithmetic
│   ├── onnx_shapes.h  # Activations, matmul, softmax
│   ├── cfc.h          # CfC neural network
│   └── entromorph.h   # Evolution engine
├── test/
│   ├── test_shapes.c  # 44 tests for shapes
│   └── test_cfc.c     # 6 tests for CfC
├── examples/
│   └── hello_xor.c    # Simplest example
└── docs/
    ├── THEORY.md      # Mathematical foundations
    ├── API.md         # Function reference
    └── EXAMPLES.md    # Usage guide
```

## Verification Status

| Component | Tests | Status |
|-----------|-------|--------|
| Logic gates (XOR, AND, OR, NOT, NAND, NOR, XNOR) | Truth tables | PASS |
| Full adder | 8 combinations | PASS |
| 8-bit ripple adder | 65,536 combinations | PASS |
| Activations (ReLU, Sigmoid, Tanh, GELU, SiLU) | Numerical | PASS |
| Softmax | Sum=1, stability | PASS |
| MatMul | Correctness | PASS |
| CfC determinism | Same input = same output | PASS |
| CfC stability | 10,000 iterations | PASS |

## Usage

### Include the headers

```c
#include "yinsen/apu.h"          // Logic and arithmetic
#include "yinsen/onnx_shapes.h"  // Neural network ops
#include "yinsen/cfc.h"          // CfC networks
#include "yinsen/entromorph.h"   // Evolution
```

### Compile

```bash
gcc -I./include -O2 your_code.c -lm
```

## Documentation

- [THEORY.md](docs/THEORY.md) - Mathematical foundations
- [API.md](docs/API.md) - Function reference
- [EXAMPLES.md](docs/EXAMPLES.md) - Usage examples

## Philosophy

1. **Falsifiable** - Every claim is backed by tests that can fail
2. **Minimal** - No dependencies beyond libc
3. **Deterministic** - Same input always produces same output
4. **Verified** - Exhaustive testing where feasible

## License

MIT
