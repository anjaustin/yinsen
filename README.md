# Yinsen

**Ternary neural computation with verified primitives.**

A research library exploring exhaustively-tested, dependency-free neural network building blocks in pure C. The "Tri" in the original TriX name stands for **ternary weights**: all network parameters are constrained to {-1, 0, +1}.

## Why Ternary?

| Property | Float32 | Int8 | Ternary |
|----------|---------|------|---------|
| Multiply op | `a * b` | `a * b` | `switch(w)` |
| Memory/weight | 32 bits | 8 bits | 2 bits |
| Determinism | Platform-dependent | Exact | Exact |
| Auditability | "This weight is 0.7324..." | "This weight is 83..." | "This weight is -1, 0, or +1" |

Ternary weights eliminate floating-point multiplication in the forward pass. A dot product becomes counting and subtraction. This isn't just an optimization—it's a different computational model that enables exhaustive verification.

## Current Status: Evolution Works

Yinsen has verified primitives AND proven evolution convergence. It does not yet have:
- A trained ternary network solving a real task (evolution uses float CfC)
- Cross-platform determinism testing (only darwin/arm64)
- Certification artifacts

**NEW:** EntroMorph evolution converges on XOR - 5/5 runs succeed in 10-30 generations.

This is research code exploring whether neural computation can be made auditable. It is not production-ready.

## What's Actually Here

### Verified (exhaustively tested)
- Logic gates as polynomials (XOR, AND, OR, NOT, NAND, NOR, XNOR)
- Full adder (8/8 input combinations)
- 8-bit ripple adder (65,536/65,536 combinations)
- 2x2 ternary matrix-vector multiply (81/81 weight configurations)

### Tested (property tests, single platform)
- Activation functions (ReLU, Sigmoid, Tanh, GELU, SiLU)
- Softmax (sum=1, numerical stability)
- MatMul (correctness)
- CfC cell (determinism, stability over 10K iterations)
- Ternary CfC cell (determinism, stability, 4.4x memory compression)

### Evolution (newly tested!)
- EntroMorph evolution converges on XOR (5/5 runs, 10-30 generations)
- Tournament selection with elitism
- Genome export to C header

### Present but untested
- Cross-platform determinism (only tested on darwin/arm64)
- Ternary evolution (current evolution uses float CfC)

## Quick Start

```bash
make test      # Run 111 core tests
make falsify   # Run 38 adversarial tests
make evolve    # Run 11 evolution tests (proves XOR convergence)
make examples
./build/hello_ternary
```

## Verification Status

| Component | Coverage | Status | Notes |
|-----------|----------|--------|-------|
| Logic gates | Truth tables | **PROVEN** | Exact for binary inputs |
| Full adder | 8/8 | **PROVEN** | Exhaustive |
| 8-bit adder | 65,536/65,536 | **PROVEN** | Exhaustive |
| Ternary 2x2 matvec | 81/81 | **PROVEN** | All 3^4 weight configs |
| Activations | Properties | **TESTED** | Single platform |
| Softmax | Properties | **TESTED** | Single platform |
| MatMul | Spot checks | **TESTED** | Not exhaustive |
| Ternary ops | Properties | **TESTED** | Pack/unpack, dot product |
| CfC cell | Properties | **TESTED** | Single platform only |
| Ternary CfC | Properties | **TESTED** | 4.4x compression measured |
| EntroMorph | XOR convergence | **TESTED** | 5/5 runs converge |
| Cross-platform | None | **UNTESTED** | Claimed, not verified |

## Project Structure

```
yinsen/
├── include/
│   ├── apu.h           # Logic + arithmetic (verified)
│   ├── onnx_shapes.h   # Activations, matmul, softmax (tested)
│   ├── ternary.h       # Ternary weights {-1,0,+1} (tested)
│   ├── cfc.h           # CfC cell, float weights (tested)
│   ├── cfc_ternary.h   # CfC cell, ternary weights (tested)
│   └── entromorph.h    # Evolution (TESTED - XOR converges)
├── test/
│   ├── test_shapes.c      # 44 tests
│   ├── test_cfc.c         # 6 tests
│   ├── test_ternary.c     # 55 tests
│   ├── test_cfc_ternary.c # 6 tests
│   ├── test_entromorph.c  # 11 tests (XOR convergence!)
│   └── test_falsify.c     # 38 adversarial tests
├── examples/
│   ├── hello_xor.c
│   └── hello_ternary.c
├── docs/
│   ├── THEORY.md       # Mathematical foundations
│   ├── API.md          # Function reference
│   ├── EXAMPLES.md     # Usage guide
│   └── CLAIMS.md       # Verification claims register
└── journal/
    ├── LMM.md          # Research methodology
    ├── scratchpad/     # Active work
    └── archive/        # Completed explorations
```

## Usage

```c
#include "ternary.h"      // Ternary weight system
#include "cfc_ternary.h"  // Ternary CfC networks
#include "apu.h"          // Logic and arithmetic
#include "onnx_shapes.h"  // Neural network ops
#include "entromorph.h"   // Evolution (TESTED)
```

```bash
gcc -I./include -O2 -std=c11 your_code.c -lm
```

## Architecture: The Ternary Stack

```
┌─────────────────────────────────────────┐
│  EntroMorph (Evolution)     [TESTED]    │
│  - Evolves CfC networks (XOR proven)    │
├─────────────────────────────────────────┤
│  CfC Ternary Cell           [TESTED]    │
│  - Temporal dynamics with ternary W     │
│  - 4.4x memory vs float CfC             │
├─────────────────────────────────────────┤
│  Ternary Primitives         [TESTED]    │
│  - 2-bit trit encoding                  │
│  - Multiplication-free dot product      │
│  - Pack/unpack for compression          │
├─────────────────────────────────────────┤
│  APU (Logic/Arithmetic)     [PROVEN]    │
│  - Gates as polynomials                 │
│  - Exhaustive adder verification        │
└─────────────────────────────────────────┘
```

## Research Questions

This repo explores:

1. **Can ternary neural networks be exhaustively verified?** (Partial: 2x2 proven, larger sizes tested)

2. **Can we achieve cross-platform determinism?** (Unknown: untested)

3. **Can evolution produce useful networks?** (YES for float CfC on XOR; ternary evolution untested)

4. **What tasks can ternary CfC solve?** (Unknown: no end-to-end demo yet)

## Known Limitations

- **No end-to-end demo**: Primitives exist but haven't produced a working network
- **Determinism untested across platforms**: `expf()` in activations may vary
- **Ternary limits expressivity**: Some functions may need more precision
- **CfC is a gated recurrence**: The "continuous-time" framing comes from literature
- **Header-only tradeoffs**: Good for embedding, bad for build times at scale
- **No WCET/stack analysis**: Can't deploy to hard real-time without this

## Documentation

- [THEORY.md](docs/THEORY.md) - Mathematical foundations
- [API.md](docs/API.md) - Function reference  
- [EXAMPLES.md](docs/EXAMPLES.md) - Usage examples
- [CLAIMS.md](docs/CLAIMS.md) - Verification claims register

## Roadmap

- [x] **Test EntroMorph**: Evolution converges on XOR (float CfC)
- [ ] **Ternary evolution**: Evolve ternary CfC networks
- [ ] **Harder tasks**: Sequence prediction, anomaly detection
- [ ] **Cross-platform CI**: Test determinism on Linux/macOS/Windows, ARM/x86
- [ ] **Benchmark ternary vs float**: Speed, accuracy, memory on same task
- [ ] **WCET analysis**: For one platform, one network size

## License

MIT
