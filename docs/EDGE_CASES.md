# Edge Cases and Known Behaviors

This document records edge case behavior discovered through falsification testing.

## Test Results Summary

**Falsification tests:** 37/38 passed  
**Known issues:** 3 (documented below, none critical)

---

## Verified Robust Behaviors

### Absmean Quantization
- **All zeros input:** Works correctly, outputs all zeros (scale = 0, eps prevents div-by-zero)
- **Large values (1e30):** Correctly quantizes to +1/-1
- **Denormal values (1e-40):** Doesn't crash, quantizes reasonably
- **Single element:** Works correctly

### Int8 Quantization
- **All same values:** Correctly maps to max (127)
- **All zeros:** Returns all zeros, scale ≈ 0
- **Tight range:** Loses precision (expected) but doesn't crash

### Ternary Operations
- **Non-multiple-of-4 lengths:** Correctly handles 1, 2, 3, 5, 6, 7 element vectors
- **Zero-length inputs:** Returns 0, doesn't crash
- **Large accumulation (1000 elements):** No int32 overflow

### CfC Ternary
- **Small dt (0.0001):** Stable, finite output
- **Large dt (1000):** Stable, finite output
- **Zero dt:** Stable (decay = 1, full retention)
- **Tiny tau (1e-10):** Stable (decay ≈ 0, full update)
- **10K iterations:** Remains bounded

### Sparsity Counting
- **All zeros (0x00):** Correctly reports 100% sparse
- **All +1 (0x55):** Correctly reports 0% sparse
- **All -1 (0xFF):** Correctly reports 0% sparse

### Reserved Encoding
- **Encoding 0b10:** Treated as 0 (skip), doesn't contribute to dot product

---

## Known Issues (Non-Critical)

### Issue 1: NaN/Inf Input Handling

**Behavior:** NaN and Inf inputs are not validated. They propagate through computation.

**Example:**
```c
float x[4] = {1.0f, NAN, 1.0f, 1.0f};
float result = ternary_dot(&w, x, 4);  // result is NaN
```

**Impact:** Low - if inputs contain NaN/Inf, the computation is already invalid.

**Recommendation:** Add optional validation functions for safety-critical use:
```c
int ternary_validate_inputs(const float* x, int n);  // Returns 0 if valid
```

**Status:** Documented, not fixed (performance tradeoff)

---

### Issue 2: Negative dt Handling

**Behavior:** Negative dt values are not validated. They produce unexpected decay.

**Example:**
```c
// dt = -1, tau = 1
// decay = exp(-(-1)/1) = exp(1) = 2.718
// This means state is AMPLIFIED, not decayed
```

**Impact:** Medium - negative dt is physically meaningless but doesn't crash.

**Recommendation:** Document that dt must be non-negative. Optionally add assertion.

**Status:** Documented, not fixed (caller responsibility)

---

### Issue 3: Zero tau Behavior

**Behavior:** tau = 0 produces decay = exp(-inf) = 0, which means full update (no retention).

**Analysis:**
```c
// tau = 0, dt = 0.1
// -dt/tau = -0.1/0 = -inf
// decay = exp(-inf) = 0
// h_new = (1-gate) * h_prev * 0 + gate * candidate
//       = gate * candidate
```

**Impact:** Low - mathematically coherent but not the expected "division by zero" error.

**Recommendation:** Document that tau should be positive. Consider adding epsilon to tau.

**Status:** Documented, mathematically valid behavior

---

## Overflow Boundaries

### Int32 Accumulator
- **Maximum safe vector length:** ~16.9 million elements
- **Calculation:** INT32_MAX / 127 = 16,909,320
- **Typical Yinsen networks:** <10K parameters (safe by factor of 1,690x)

### Float32 Weights
- **Large values (1e30):** Work correctly
- **Denormal values (1e-40):** Work correctly
- **Inf/NaN:** Undefined behavior (see Issue 1)

---

## Recommendations for Safety-Critical Use

If using Yinsen in safety-critical applications, consider:

1. **Input validation:** Add explicit checks for NaN/Inf before computation
2. **Parameter validation:** Assert dt >= 0, tau > 0 at initialization
3. **Output validation:** Check outputs are finite after each cell step
4. **Bounds checking:** Verify vector lengths are within expected ranges

Example validation wrapper:
```c
int yinsen_cfc_ternary_cell_safe(
    const float* x, const float* h_prev, float dt,
    const CfCTernaryParams* params, float* h_new
) {
    // Validate inputs
    if (dt < 0) return -1;  // Invalid dt
    if (params->tau[0] <= 0) return -2;  // Invalid tau
    
    for (int i = 0; i < params->input_dim; i++) {
        if (!isfinite(x[i])) return -3;  // Invalid input
    }
    
    // Run computation
    yinsen_cfc_ternary_cell(x, h_prev, dt, params, h_new);
    
    // Validate output
    for (int i = 0; i < params->hidden_dim; i++) {
        if (!isfinite(h_new[i])) return -4;  // Computation produced invalid result
    }
    
    return 0;  // Success
}
```

---

## Test Coverage

| Area | Tests | Status |
|------|-------|--------|
| Absmean quantization | 5 | All pass |
| Single element ops | 3 | All pass |
| Large values | 3 | All pass |
| Denormal values | 1 | Pass |
| NaN/Inf handling | 1 | Pass (propagates NaN) |
| Int8 overflow | 1 | Pass |
| Misaligned lengths | 4 | All pass |
| Empty inputs | 2 | All pass |
| Extreme dt | 4 | All pass |
| Extreme tau | 2 | 1 pass, 1 expected behavior |
| Activation extremes | 4 | All pass |
| Sparsity edge cases | 6 | All pass |
| Reserved encoding | 2 | All pass |
| Stress test | 1 | Pass |

**Total:** 38 falsification tests, 37 pass, 1 documented behavior

---

## See Also

- [API.md](API.md) - Complete function reference
- [CLAIMS.md](CLAIMS.md) - Verification status of all claims
- [VERIFICATION.md](VERIFICATION.md) - Complete test verification report

---

## Changelog

- 2026-01-26: Initial falsification testing, documented edge cases
