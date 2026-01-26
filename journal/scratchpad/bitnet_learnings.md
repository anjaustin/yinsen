# Actionable Learnings from BitNet b1.58

What can we actually apply to Yinsen right now?

---

## 1. Absmean Quantization Function

**What BitNet does:**
```
W̃ = RoundClip(W / γ, -1, 1)
γ = mean(|W|)  # average absolute value
```

**What we do:**
```c
// Current: threshold-based
if (val > threshold) trit = +1;
else if (val < -threshold) trit = -1;
else trit = 0;
```

**The problem:** Our threshold is arbitrary. BitNet's approach adapts to the weight distribution.

**Action:** Add `ternary_quantize_absmean()` function.

```c
static inline void ternary_quantize_absmean(
    const float* weights,
    uint8_t* packed,
    int n
) {
    // Compute absmean (γ)
    float sum = 0.0f;
    for (int i = 0; i < n; i++) {
        sum += fabsf(weights[i]);
    }
    float gamma = sum / n;
    
    // Quantize using γ as scale
    int byte_idx = 0;
    uint8_t current_byte = 0;
    
    for (int i = 0; i < n; i++) {
        float scaled = weights[i] / (gamma + 1e-8f);
        int8_t trit;
        if (scaled > 0.5f) trit = 1;
        else if (scaled < -0.5f) trit = -1;
        else trit = 0;
        
        int bit_pos = i % 4;
        current_byte |= (trit_encode(trit) << (bit_pos * 2));
        
        if (bit_pos == 3 || i == n - 1) {
            packed[byte_idx++] = current_byte;
            current_byte = 0;
        }
    }
}
```

**Benefit:** Better quantization when converting pre-trained float weights.

---

## 2. Activation Quantization (8-bit)

**What BitNet does:**
- Activations are quantized to 8-bit before matrix multiply
- Range: [-Qb, Qb] per token
- This enables integer-only matmul kernels

**What we do:**
- Activations stay float32 throughout

**The opportunity:** Quantize activations for fully integer forward pass (except final nonlinearities).

**Action:** Add optional activation quantization.

```c
typedef struct {
    float scale;
    int8_t zero_point;  // 0 for symmetric
} QuantParams;

// Quantize activations to int8
static inline void quantize_activations_symmetric(
    const float* x,
    int8_t* x_q,
    int n,
    QuantParams* params
) {
    // Find absmax
    float absmax = 0.0f;
    for (int i = 0; i < n; i++) {
        float abs_val = fabsf(x[i]);
        if (abs_val > absmax) absmax = abs_val;
    }
    
    params->scale = absmax / 127.0f;
    params->zero_point = 0;
    
    for (int i = 0; i < n; i++) {
        float scaled = x[i] / params->scale;
        x_q[i] = (int8_t)fmaxf(-127.0f, fminf(127.0f, roundf(scaled)));
    }
}

// Integer ternary dot product
static inline int32_t ternary_dot_int8(
    const uint8_t* w_packed,
    const int8_t* x_q,
    int n
) {
    int32_t sum = 0;
    int byte_idx = 0;
    int bit_pos = 0;
    
    for (int i = 0; i < n; i++) {
        int8_t trit = trit_unpack(w_packed[byte_idx], bit_pos);
        
        if (trit > 0) {
            sum += x_q[i];
        } else if (trit < 0) {
            sum -= x_q[i];
        }
        
        bit_pos++;
        if (bit_pos == 4) {
            bit_pos = 0;
            byte_idx++;
        }
    }
    
    return sum;  // Dequantize later: result * scale
}
```

**Benefit:** 
- True integer matmul (no float in inner loop)
- Better for MCUs without FPU
- Closer to BitNet's full pipeline

---

## 3. Remove Biases (like LLaMA)

**What BitNet does:**
- Follows LLaMA: removes all biases from linear layers
- Only uses RMSNorm (not LayerNorm)

**What we do:**
- Keep biases on everything

**The question:** Do we need biases?

**Observation:** With ternary weights, the bias is the ONLY float parameter in linear layers. Removing it makes the whole layer integer.

**Action:** Add bias-free variants.

```c
// Bias-free ternary matvec
static inline void ternary_matvec_nobias(
    const uint8_t* W_packed,
    const float* x,
    float* y,
    int M,
    int N
) {
    ternary_matvec(W_packed, x, y, M, N);
    // No bias addition
}
```

**Consideration:** Biases might matter more at small scale. Test both.

---

## 4. Proper "1.58 bits" Terminology

**What BitNet does:**
- Uses "1.58 bits" (log₂(3) ≈ 1.585)
- More accurate than "2 bits"

**What we do:**
- Say "2 bits per weight"

**Action:** Update documentation to use "1.58-bit" terminology.

**Why it matters:** 
- More precise
- Aligns with literature
- Sounds more sophisticated than "2-bit"

---

## 5. RMSNorm Instead of LayerNorm

**What BitNet does:**
- Uses RMSNorm (from LLaMA):
  ```
  RMSNorm(x) = x / RMS(x) * γ
  RMS(x) = sqrt(mean(x²))
  ```
- Cheaper than LayerNorm (no mean subtraction)

**What we do:**
- No normalization in CfC cell currently

**Action:** Add RMSNorm for potential use.

```c
static inline void yinsen_rmsnorm(
    const float* x,
    const float* gamma,  // Learnable scale
    float* y,
    int n,
    float eps
) {
    // Compute RMS
    float sum_sq = 0.0f;
    for (int i = 0; i < n; i++) {
        sum_sq += x[i] * x[i];
    }
    float rms = sqrtf(sum_sq / n + eps);
    
    // Normalize and scale
    for (int i = 0; i < n; i++) {
        y[i] = (x[i] / rms) * gamma[i];
    }
}
```

**Benefit:** If we add normalization, use the cheaper one.

---

## 6. The Zero Value Insight

**What BitNet says:**
> "Its modeling capability is stronger due to its explicit support for feature filtering, made possible by the inclusion of 0 in the model weights."

**What this means:**
- Zero isn't just "missing information"
- Zero is an explicit "ignore this input" signal
- This is a FEATURE, not a limitation

**Action:** 
- Emphasize this in our documentation
- Track sparsity (zero count) as a metric
- Consider sparse ternary evolution (favor zeros)

---

## 7. Scaling Law Observation

**What BitNet shows:**
- Performance improves with scale even for ternary
- 3B ternary ≈ 3B float performance
- The scaling law still applies

**Implication for Yinsen:**
- Even at small scale, "bigger is better" within constraints
- If we have budget for 1000 weights, use all 1000
- Don't artificially constrain network size

---

## 8. Energy Model Reference

**What BitNet cites:**
- Horowitz (2014): "Computing's Energy Problem"
- On 7nm: INT8 ADD is 0.03 pJ, FP16 MUL is 0.9 pJ
- That's 30x energy difference per operation

**Action:** Add energy estimation to our metrics.

```c
// Rough energy model (7nm estimates from Horowitz)
#define ENERGY_INT8_ADD_PJ   0.03f
#define ENERGY_FP16_MUL_PJ   0.9f
#define ENERGY_FP16_ADD_PJ   0.4f

static inline float estimate_ternary_matvec_energy_pj(int M, int N) {
    // Ternary: mostly int adds, some float adds for accumulation
    // Approximation: N int-adds per row, M rows
    return M * N * ENERGY_INT8_ADD_PJ;
}

static inline float estimate_float_matvec_energy_pj(int M, int N) {
    // Float: N muls + N adds per row, M rows
    return M * N * (ENERGY_FP16_MUL_PJ + ENERGY_FP16_ADD_PJ);
}
```

**Benefit:** Can quote energy savings in documentation.

---

## Implementation Priority

| Learning | Priority | Effort | Impact |
|----------|----------|--------|--------|
| Absmean quantization | HIGH | Low | Better weight conversion |
| Int8 activations | MEDIUM | Medium | True integer forward pass |
| Remove biases | LOW | Low | Experiment needed |
| 1.58-bit terminology | HIGH | Trivial | Documentation accuracy |
| RMSNorm | LOW | Low | Only if we add normalization |
| Zero as feature | MEDIUM | Trivial | Documentation + metrics |
| Energy model | MEDIUM | Low | Nice for marketing |

---

## Immediate Next Steps

1. **Add `ternary_quantize_absmean()`** to ternary.h
2. **Add `ternary_dot_int8()`** for integer forward pass
3. **Update docs** to use "1.58-bit" terminology
4. **Add sparsity tracking** (count zeros as metric)
5. **Add energy estimation** functions

These are all additive - they don't break existing code.
