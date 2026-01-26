# Yinsen

**Verified primitives for neural computation.**

A research library exploring exhaustively-tested, dependency-free neural network building blocks in pure C.

## Current Status: Foundation

Yinsen has verified primitives. It does not yet have:
- A trained/evolved network solving a real task
- Cross-platform determinism testing
- Performance benchmarks (WCET, stack usage)
- Certification artifacts

This is research code exploring whether neural computation can be made more auditable. It is not production-ready.

## What's Actually Here

**Verified (exhaustively tested):**
- Logic gates as polynomials (XOR, AND, OR, NOT, NAND, NOR, XNOR)
- Full adder (8/8 input combinations)
- 8-bit ripple adder (65,536/65,536 combinations)

**Tested (property tests, single platform):**
- Activation functions (ReLU, Sigmoid, Tanh, GELU, SiLU)
- Softmax (sum=1, numerical stability)
- MatMul (correctness)
- CfC cell (determinism on this machine, stability over 10K iterations)

**Present but untested:**
- EntroMorph evolution engine (no convergence tests)
- Cross-platform determinism (only tested on darwin/arm64)

## Quick Start

```bash
make test    # Run 50 tests
make examples
./build/hello_xor
```

## Verification Status

| Component | Coverage | Status | Notes |
|-----------|----------|--------|-------|
| Logic gates | Truth tables | **PROVEN** | Exact for binary inputs |
| Full adder | 8/8 | **PROVEN** | Exhaustive |
| 8-bit adder | 65,536/65,536 | **PROVEN** | Exhaustive |
| Activations | Properties | **TESTED** | Single platform |
| Softmax | Properties | **TESTED** | Single platform |
| MatMul | Spot checks | **TESTED** | Not exhaustive |
| CfC cell | Properties | **TESTED** | Single platform only |
| EntroMorph | None | **UNTESTED** | No convergence proof |
| Cross-platform | None | **UNTESTED** | Claimed, not verified |

## Project Structure

```
yinsen/
├── include/
│   ├── apu.h           # Logic + arithmetic (verified)
│   ├── onnx_shapes.h   # Activations, matmul, softmax (tested)
│   ├── cfc.h           # CfC cell (tested, single platform)
│   └── entromorph.h    # Evolution (present, untested)
├── test/
│   ├── test_shapes.c   # 44 tests
│   └── test_cfc.c      # 6 tests
├── examples/
│   └── hello_xor.c
├── docs/
│   ├── THEORY.md       # Mathematical foundations
│   ├── API.md          # Function reference
│   └── EXAMPLES.md     # Usage guide
└── journal/
    ├── LMM.md          # Research methodology
    ├── scratchpad/     # Active work
    └── archive/        # Completed explorations
```

## Usage

```c
#include "apu.h"          // Logic and arithmetic
#include "onnx_shapes.h"  // Neural network ops
#include "cfc.h"          // CfC networks
#include "entromorph.h"   // Evolution (untested)
```

```bash
gcc -I./include -O2 -std=c11 your_code.c -lm
```

## Research Questions

This repo explores:

1. **Can neural primitives be exhaustively verified?** (Partial: yes for logic/arithmetic, property tests for continuous functions)

2. **Can we achieve cross-platform determinism?** (Unknown: untested)

3. **Can evolution replace backprop with full provenance?** (Unknown: evolution engine untested)

4. **Is CfC meaningfully different from GRU?** (Unknown: no comparative benchmarks)

## Known Limitations

- **No end-to-end demo**: Primitives exist but haven't produced a working network
- **Determinism untested across platforms**: `expf()` may vary between implementations
- **CfC is a gated recurrence**: The "continuous-time" framing comes from literature; our implementation is a discrete update rule
- **Header-only tradeoffs**: Good for embedding, bad for build times at scale
- **No WCET/stack analysis**: Can't deploy to hard real-time without this

## Documentation

- [THEORY.md](docs/THEORY.md) - Mathematical foundations
- [API.md](docs/API.md) - Function reference  
- [EXAMPLES.md](docs/EXAMPLES.md) - Usage examples

## Roadmap (What Would Make This Real)

- [ ] **Convergence test**: Prove EntroMorph produces fit networks
- [ ] **One benchmark task**: Solve something, report accuracy
- [ ] **Cross-platform CI**: Test determinism on Linux/macOS/Windows, ARM/x86
- [ ] **WCET analysis**: For one platform, one network size
- [ ] **End-to-end demo**: Evolve → export → deploy → verify identical output

## License

MIT
