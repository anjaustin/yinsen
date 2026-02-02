# API Reference

Complete function reference for Yinsen. Two layers: **Chip Forge** (the product) and **Base Primitives** (foundations).

**Last Updated:** 2026-02-01

---

## Chip Forge (`include/chips/`)

Eight frozen primitives. All `static inline`. Header-only, zero dependencies beyond `<math.h>`.

---

### cfc_cell_chip.h — CfC Cell (4 Variants)

The atomic unit of liquid computation. Single function call replaces ODE integration with closed-form math.

#### CFC_CELL_GENERIC — Variable time step

```c
static inline void CFC_CELL_GENERIC(
    const float* x,            // Input signal [input_dim]
    const float* h_prev,       // Previous hidden state [hidden_dim]
    float dt,                  // Time delta
    const float* W_gate,       // Gate weights [hidden_dim, input_dim + hidden_dim]
    const float* b_gate,       // Gate biases [hidden_dim]
    const float* W_cand,       // Candidate weights [hidden_dim, input_dim + hidden_dim]
    const float* b_cand,       // Candidate biases [hidden_dim]
    const float* tau,          // Time constants [hidden_dim] or [1]
    int tau_shared,            // If true, single tau for all neurons
    int input_dim,
    int hidden_dim,
    float* h_new               // Output: new hidden state [hidden_dim]
)
```

Computes: `h_new = (1 - gate) * h_prev * decay + gate * candidate`
- gate = sigmoid(W_gate @ [x; h_prev] + b_gate)
- candidate = tanh(W_cand @ [x; h_prev] + b_cand)
- decay = exp(-dt / tau)

Performance: 54 ns/step (Apple M-series, hidden_dim=8). Uses libm sigmoid/tanh/exp.

#### CFC_CELL_FIXED — Precomputed decay

```c
static inline void CFC_CELL_FIXED(
    const float* x,
    const float* h_prev,
    const float* W_gate,
    const float* b_gate,
    const float* W_cand,
    const float* b_cand,
    const float* decay_precomputed,  // Precomputed exp(-dt/tau) [hidden_dim]
    int input_dim,
    int hidden_dim,
    float* h_new
)
```

Same as GENERIC but with precomputed decay. No exp() at runtime. 51 ns/step.

#### CFC_CELL_LUT — LUT+lerp activations

```c
static inline void CFC_CELL_LUT(
    const float* x,
    const float* h_prev,
    const float* W_gate,
    const float* b_gate,
    const float* W_cand,
    const float* b_cand,
    const float* decay_precomputed,
    int input_dim,
    int hidden_dim,
    float* h_new
)
```

Uses LUT+lerp for sigmoid/tanh (200x more accurate than FAST3, only 4ns slower). Requires `ACTIVATION_LUT_INIT()` once at startup. 35 ns/step.

#### CFC_CELL_SPARSE — Zero-multiply

```c
static inline void CFC_CELL_SPARSE(
    const float* x,
    const float* h_prev,
    const CfcSparseWeights* sw,          // Sparse weights (built by cfc_build_sparse)
    const float* b_gate,
    const float* b_cand,
    const float* decay_precomputed,
    int input_dim,
    int hidden_dim,
    float* h_new
)
```

Replaces dense GEMM with sparse index-list traversal. For each neuron: `pre = bias + sum(x[pos]) - sum(x[neg])`. Zero multiplies. Bit-identical to CFC_CELL_LUT with same ternary weights. **20 ns/step** (2.73x faster than GENERIC).

#### Sparse Types

```c
#define CFC_SPARSE_MAX_CONCAT  32
#define CFC_SPARSE_MAX_HIDDEN  32

typedef struct {
    int8_t pos_idx[CFC_SPARSE_MAX_CONCAT + 1];  // indices where w=+1, -1 terminated
    int8_t neg_idx[CFC_SPARSE_MAX_CONCAT + 1];  // indices where w=-1, -1 terminated
} CfcSparseRow;

typedef struct {
    CfcSparseRow gate[CFC_SPARSE_MAX_HIDDEN];
    CfcSparseRow cand[CFC_SPARSE_MAX_HIDDEN];
    int hidden_dim;
    int concat_dim;
} CfcSparseWeights;
```

#### Builders

```c
// Build sparse from float weights
static inline void cfc_build_sparse(
    const float* W_gate,
    const float* W_cand,
    float threshold,       // Quantization threshold (0 for pre-quantized {-1,0,+1})
    int hidden_dim,
    int concat_dim,
    int transposed,        // 1 if W is [hidden x concat], 0 if [concat x hidden]
    CfcSparseWeights* out
)

// Precompute decay for fixed dt
static inline void cfc_precompute_decay(
    const float* tau,
    int tau_shared,
    float dt,
    int hidden_dim,
    float* decay_out       // Output: precomputed decay [hidden_dim]
)
```

**GEMM layout note:** `yinsen_gemm` reads `W[k * hidden_dim + j]`. Demo weights are declared as `W[hidden_dim * concat_dim]`. When calling `cfc_build_sparse`, use `transposed=0` for GEMM-native layout.

---

### activation_chip.h — Activations (3 Tiers)

#### Sigmoid

```c
static inline float SIGMOID_CHIP(float x)        // Precise: 1/(1+exp(-x))
static inline float SIGMOID_CHIP_FAST(float x)    // Rational: 0.5 + 0.5*x/(1+|x|), ~0.07 error
static inline float SIGMOID_CHIP_FAST3(float x)   // Degree-3: ~2e-3 error, clamps outside [-4,4]
static inline float SIGMOID_CHIP_LUT(float x)     // LUT+lerp: 4.7e-5 max error over [-8,8]
```

#### Tanh

```c
static inline float TANH_CHIP(float x)            // Precise: tanhf(x)
static inline float TANH_CHIP_FAST(float x)       // x/(1+|x|), ~0.14 error
static inline float TANH_CHIP_FAST3(float x)      // Pade-like: ~5e-3 error, clamps outside [-3,3]
static inline float TANH_CHIP_LUT(float x)        // LUT+lerp: 3.8e-4 max error over [-8,8]
```

#### Exp

```c
static inline float EXP_CHIP(float x)             // Precise: expf(x)
static inline float EXP_CHIP_FAST(float x)        // Schraudolph bit-trick, ~4% error
```

#### LUT Initialization

```c
static inline void ACTIVATION_LUT_INIT(void)      // Fill tables. Call once. Idempotent.
```

2KB total (1KB sigmoid + 1KB tanh). 256 entries each. Shared read-only, hot in L1.

#### Other Activations

```c
static inline float RELU_CHIP(float x)            // max(0, x)
static inline float GELU_CHIP(float x)            // x * sigmoid(1.702*x)
static inline float GELU_CHIP_FAST(float x)
static inline float SILU_CHIP(float x)            // x * sigmoid(x)
static inline float SILU_CHIP_FAST(float x)
```

#### Vectorized Variants

```c
static inline void SIGMOID_VEC_CHIP(const float* x, float* y, int n)
static inline void SIGMOID_VEC_CHIP_FAST(const float* x, float* y, int n)
static inline void SIGMOID_VEC_CHIP_LUT(const float* x, float* y, int n)
static inline void TANH_VEC_CHIP(const float* x, float* y, int n)
static inline void TANH_VEC_CHIP_FAST(const float* x, float* y, int n)
static inline void TANH_VEC_CHIP_LUT(const float* x, float* y, int n)
static inline void RELU_VEC_CHIP(const float* x, float* y, int n)
static inline void EXP_VEC_CHIP(const float* x, float* y, int n)
static inline void EXP_VEC_CHIP_FAST(const float* x, float* y, int n)
```

---

### gemm_chip.h — General Matrix Multiply

```c
// C = A @ B (no bias, no scaling)
static inline void GEMM_CHIP_BARE(
    const float* a, const float* b, float* c,
    int M, int N, int K
)

// C = A @ B + bias
static inline void GEMM_CHIP_BIASED(
    const float* a, const float* b, const float* bias, float* c,
    int M, int N, int K
)

// C = alpha * A @ B + beta * bias (full GEMM)
static inline void GEMM_CHIP(
    const float* a, const float* b, const float* bias, float* c,
    int M, int N, int K, float alpha, float beta
)
```

For M=1 (the CfC case), the inner loop is a dot product. The compiler vectorizes with FMA.

---

### decay_chip.h — Temporal Decay

```c
static inline float DECAY_CHIP_SCALAR(float dt, float tau)       // exp(-dt/tau)
static inline float DECAY_CHIP_SCALAR_FAST(float dt, float tau)  // Schraudolph, ~4% error

// Fill vector with single decay (tau_shared mode)
static inline void DECAY_CHIP_SHARED(float dt, float tau, float* decay_out, int hidden_dim)

// Fill vector with per-neuron decay
static inline void DECAY_CHIP_VECTOR(float dt, const float* tau, float* decay_out, int hidden_dim)
```

---

### ternary_dot_chip.h — Multiplication-Free Dot Product

```c
// Float activations, 2-bit weights
static inline float TERNARY_DOT_CHIP(const uint8_t* w_packed, const float* x, int n)

// Int8 activations, 2-bit weights -> int32 accumulator
static inline int32_t TERNARY_DOT_CHIP_INT8(const uint8_t* w_packed, const int8_t* x, int n)

// Int16 activations, 2-bit weights -> int32 accumulator
static inline int32_t TERNARY_DOT_CHIP_INT16(const uint8_t* w_packed, const int16_t* x, int n)
```

Encoding: `00`=0 (skip), `01`=+1 (add), `10`=-1 (subtract), `11`=reserved (skip).

---

### fft_chip.h — Fast Fourier Transform

```c
// Forward FFT, in-place, radix-2 Cooley-Tukey
static inline void FFT_CHIP(float* real, float* imag, int N)

// Inverse FFT
static inline void IFFT_CHIP(float* real, float* imag, int N)

// Magnitude spectrum: out[i] = sqrt(real[i]^2 + imag[i]^2)
static inline void FFT_MAGNITUDE(const float* real, const float* imag, float* out, int N)

// Power spectrum: out[i] = real[i]^2 + imag[i]^2
static inline void FFT_POWER(const float* real, const float* imag, float* out, int N)
```

N must be a power of 2. For real-only input, pass imag initialized to zeros.

---

### softmax_chip.h — Softmax

```c
static inline void SOFTMAX_CHIP(const float* x, float* out, int n)      // Numerically stable
static inline void SOFTMAX_CHIP_FAST(const float* x, float* out, int n) // Schraudolph exp
static inline int  ARGMAX_CHIP(const float* x, int n)                   // Returns index of max
```

---

### norm_chip.h — Normalization

```c
// Layer Normalization: out = gamma * (x - mean) / sqrt(var + eps) + beta
static inline void LAYERNORM_CHIP(
    const float* x, const float* gamma, const float* beta,
    float* out, int n, float eps
)

// RMS Normalization: out = gamma * x / sqrt(mean(x^2) + eps)
static inline void RMSNORM_CHIP(
    const float* x, const float* gamma,
    float* out, int n, float eps
)

// Batch Normalization (inference): out = gamma * (x - mean) / sqrt(var + eps) + beta
static inline void BATCHNORM_CHIP(
    const float* x, const float* mean, const float* var,
    const float* gamma, const float* beta,
    float* out, int n, float eps
)
```

---

## Base Primitives (`include/`)

### apu.h — Logic and Arithmetic

```c
// Logic gates — exact for {0.0, 1.0} inputs (PROVEN)
float yinsen_xor(float a, float b)    // a + b - 2ab
float yinsen_and(float a, float b)    // a * b
float yinsen_or(float a, float b)     // a + b - ab
float yinsen_not(float a)             // 1 - a
float yinsen_nand(float a, float b)   // 1 - ab
float yinsen_nor(float a, float b)    // 1 - a - b + ab
float yinsen_xnor(float a, float b)   // 1 - a - b + 2ab

// Arithmetic
void yinsen_full_adder(float a, float b, float c, float* sum, float* carry)
void yinsen_ripple_add_8bit(const float* a, const float* b, float c_in, float* result, float* c_out)
float yinsen_hamming(const float* a, const float* b, int len)
```

### onnx_shapes.h — Neural Network Operations

```c
// Activations
float yinsen_relu(float x)
float yinsen_sigmoid(float x)
float yinsen_tanh(float x)
float yinsen_gelu(float x)
float yinsen_silu(float x)

// Softmax
void yinsen_softmax(const float* x, float* out, int n)

// Matrix operations
void yinsen_matmul(const float* a, const float* b, float* c, int M, int N, int K)
void yinsen_gemm(const float* a, const float* b, const float* bias, float* c,
                 int M, int N, int K, float alpha, float beta)

// Reductions
float yinsen_reduce_sum(const float* x, int n)
float yinsen_reduce_mean(const float* x, int n)
float yinsen_reduce_max(const float* x, int n)

// Layer normalization
void yinsen_layer_norm(const float* x, const float* gamma, const float* beta,
                       float* out, int n, float eps)
```

### ternary.h — 2-bit Weight Operations

```c
// Encoding (canonical: 00=0, 01=+1, 10=-1, 11=reserved)
int8_t trit_unpack(uint8_t packed, int pos)
uint8_t trit_encode(int8_t val)
uint8_t trit_pack4(int8_t t0, int8_t t1, int8_t t2, int8_t t3)

// Dot product and matvec (multiplication-free)
float ternary_dot(const uint8_t* w_packed, const float* x, int n)
void ternary_matvec(const uint8_t* W_packed, const float* x, float* y, int M, int N)
void ternary_matvec_bias(const uint8_t* W_packed, const float* x, const float* bias, float* y, int M, int N)

// Quantization
void ternary_quantize(const float* weights, uint8_t* packed, int n, float threshold)
void ternary_quantize_absmean(const float* weights, uint8_t* packed, int n)
float ternary_absmean_scale(const float* weights, int n)

// Unpacking
void ternary_unpack_to_float(const uint8_t* packed, float* weights, int n)

// Int8 operations
int32_t ternary_dot_int8(const uint8_t* w_packed, const int8_t* x_q, int n)
void ternary_matvec_int8(const uint8_t* W_packed, const int8_t* x_q, int32_t* y_int, int M, int N)

// Memory utilities
size_t ternary_bytes(int n)
size_t ternary_matrix_bytes(int M, int N)

// Statistics
typedef struct { int total, positive, negative, zeros; float sparsity; } TernaryStats;
void ternary_stats(const uint8_t* packed, int n, TernaryStats* stats)
float ternary_sparsity(const uint8_t* packed, int n)

// Energy estimation (Horowitz 2014, 7nm)
float ternary_matvec_energy_pj(int M, int N)
float float_matvec_energy_pj(int M, int N)
float ternary_energy_savings_ratio(int M, int N)
```

### cfc.h / cfc_ternary.h — Original CfC (struct-based)

These are the original struct-based CfC APIs, still valid but superseded by the chip forge for new code.

```c
// Float CfC
void yinsen_cfc_cell(const float* x, const float* h_prev, float dt, const CfCParams* params, float* h_new)
void yinsen_cfc_cell_fixed(const float* x, const float* h_prev, const CfCParamsFixed* params, float* h_new)
void yinsen_cfc_forward(const float* inputs, int seq_len, float dt, const CfCParams* params,
                        const float* h_init, float* outputs, float* h_final)
void yinsen_cfc_output(const float* h, const CfCOutputParams* params, float* output)
void yinsen_cfc_output_softmax(const float* h, const CfCOutputParams* params, float* probs)

// Ternary CfC
void yinsen_cfc_ternary_cell(const float* x, const float* h_prev, float dt,
                              const CfCTernaryParams* params, float* h_new)
void yinsen_cfc_ternary_output(const float* h, const CfCTernaryOutputParams* params, float* output)
void yinsen_cfc_ternary_output_softmax(const float* h, const CfCTernaryOutputParams* params, float* probs)
```

### entromorph.h — Evolution Engine (FALSIFIED)

> **WARNING: FALSIFIED.** Does not produce learned solutions. See [FALSIFICATION_ENTROMORPH.md](FALSIFICATION_ENTROMORPH.md).

Component functions (RNG, genesis, mutation, export) work correctly. The evolution process does not.

---

## See Also

- [EDGE_CASES.md](EDGE_CASES.md) - Edge case behavior
- [CLAIMS.md](CLAIMS.md) - Verification status of all claims
- [VERIFICATION.md](VERIFICATION.md) - Complete test verification report
- [EXAMPLES.md](EXAMPLES.md) - Usage examples
