# Yinsen

**Certifiable neural computation.**

Neural networks are black boxes. You can't certify a black box.

Yinsen is different. Every primitive is exhaustively tested. Every network is evolved with full provenance. Every deployment is deterministic across platforms.

If you need neural computation you can audit, certify, and defend - that's what Yinsen is for.

## Who is this for?

Organizations deploying neural computation where:

- **Certification is required** - aerospace (DO-178C), medical (IEC 62304), automotive (ISO 26262)
- **Determinism is mandatory** - safety-critical systems, financial applications
- **Resources are constrained** - embedded, edge, IoT
- **Audit trails are non-negotiable** - regulated industries, legal defensibility

## What Yinsen provides

| Need | Solution |
|------|----------|
| Prove correctness | Exhaustive testing (65,536 combinations for 8-bit adder) |
| Explain the model | Evolution with full provenance, not black-box backprop |
| Deploy anywhere | Header-only C, no dependencies, no runtime |
| Guarantee determinism | Same input → same output, verified |
| Defend to auditors | Complete retention of all research artifacts |

## Quick Start

```bash
make test    # Run verification suite (50 tests)
make examples
./build/hello_xor
```

## The Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         YINSEN                                   │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   VERIFY              BUILD               DEPLOY                 │
│   ──────              ─────               ──────                 │
│   Exhaustive    ───▶  CfC networks  ───▶  Header-only C         │
│   tests for           from verified       export with            │
│   all primitives      primitives          zero dependencies      │
│                                                                  │
│   EVOLVE              AUDIT               CERTIFY                │
│   ──────              ─────               ───────                │
│   EntroMorph          Full lineage        Determinism            │
│   with complete       retained in         across platforms       │
│   provenance          journal/                                   │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

## Verification Status

| Component | Test Coverage | Status |
|-----------|---------------|--------|
| Logic gates (XOR, AND, OR, NOT, NAND, NOR, XNOR) | Complete truth tables | **PROVEN** |
| Full adder | 8/8 combinations | **PROVEN** |
| 8-bit ripple adder | 65,536/65,536 combinations | **PROVEN** |
| Activations (ReLU, Sigmoid, Tanh, GELU, SiLU) | Numerical properties | **TESTED** |
| Softmax | Sum=1, numerical stability | **TESTED** |
| MatMul | Correctness vs. known values | **TESTED** |
| CfC determinism | Identical calls → identical results | **TESTED** |
| CfC stability | 10,000 iterations without divergence | **TESTED** |

## Project Structure

```
yinsen/
├── include/
│   ├── apu.h           # Logic shapes + arithmetic (verified)
│   ├── onnx_shapes.h   # Activations, matmul, softmax
│   ├── cfc.h           # Closed-form Continuous-time networks
│   └── entromorph.h    # Evolution engine with provenance
├── test/
│   ├── test_shapes.c   # 44 verification tests
│   └── test_cfc.c      # 6 CfC tests
├── examples/
│   └── hello_xor.c     # Simplest demonstration
├── docs/
│   ├── THEORY.md       # Mathematical foundations
│   ├── API.md          # Function reference
│   └── EXAMPLES.md     # Usage guide
└── journal/
    ├── LMM.md          # Lincoln Manifold Method (research process)
    ├── scratchpad/     # Active explorations
    └── archive/        # Completed research (never deleted)
```

## Usage

```c
#include "yinsen/apu.h"          // Logic and arithmetic
#include "yinsen/onnx_shapes.h"  // Neural network ops
#include "yinsen/cfc.h"          // CfC networks
#include "yinsen/entromorph.h"   // Evolution
```

```bash
gcc -I./include -O2 -std=c11 your_code.c -lm
```

## Core Principles

1. **Certifiable** - Built for environments where you must prove correctness
2. **Auditable** - Full provenance, complete retention, no black boxes
3. **Minimal** - Header-only C, zero dependencies beyond libc
4. **Deterministic** - Same input always produces same output
5. **Verified** - Exhaustive testing where feasible, property testing elsewhere

## Documentation

- [THEORY.md](docs/THEORY.md) - Mathematical foundations (why polynomials, why CfC)
- [API.md](docs/API.md) - Complete function reference
- [EXAMPLES.md](docs/EXAMPLES.md) - Practical usage examples

## Roadmap

- [ ] End-to-end demo: evolve → export → deploy → verify
- [ ] Cross-platform determinism CI (Linux, macOS, Windows, ARM)
- [ ] Compliance mapping (DO-178C, ISO 26262, IEC 62304)
- [ ] MISRA C compliance analysis
- [ ] Formal verification of critical primitives

## License

MIT
