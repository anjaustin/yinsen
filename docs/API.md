# API Reference

Complete function reference for Yinsen.

## apu.h - Logic and Arithmetic

### Logic Shapes

All logic functions take `float` inputs in {0.0, 1.0} and return exact binary results.

```c
float yinsen_xor(float a, float b)   // a + b - 2ab
float yinsen_and(float a, float b)   // a * b
float yinsen_or(float a, float b)    // a + b - ab
float yinsen_not(float a)            // 1 - a
float yinsen_nand(float a, float b)  // 1 - ab
float yinsen_nor(float a, float b)   // 1 - a - b + ab
float yinsen_xnor(float a, float b)  // 1 - a - b + 2ab
```

### Full Adder

```c
void yinsen_full_adder(
    float a,        // Input bit a
    float b,        // Input bit b
    float c,        // Carry in
    float* sum,     // Output: sum bit
    float* carry    // Output: carry out
)
```

Computes `sum = a XOR b XOR c` and `carry = majority(a, b, c)`.

### 8-bit Ripple Adder

```c
void yinsen_ripple_add_8bit(
    const float* a,     // 8 input bits, LSB first
    const float* b,     // 8 input bits, LSB first
    float c_in,         // Carry in
    float* result,      // 8 output bits, LSB first
    float* c_out        // Carry out
)
```

Adds two 8-bit numbers represented as float arrays.

### Hamming Distance

```c
float yinsen_hamming(const float* a, const float* b, int len)
```

Computes the Hamming distance (number of differing bits) between two bit vectors.

---

## onnx_shapes.h - Neural Network Operations

### Activation Functions

```c
float yinsen_relu(float x)      // max(0, x)
float yinsen_sigmoid(float x)   // 1 / (1 + exp(-x))
float yinsen_tanh(float x)      // tanh(x)
float yinsen_gelu(float x)      // x * sigmoid(1.702 * x)
float yinsen_silu(float x)      // x * sigmoid(x)
```

### Softmax

```c
void yinsen_softmax(const float* x, float* out, int n)
```

Computes softmax with numerical stability (subtracts max before exp).

- `x`: Input logits [n]
- `out`: Output probabilities [n], sum to 1.0
- `n`: Vector length

### Matrix Operations

```c
void yinsen_matmul(
    const float* a,   // Matrix A [M x K]
    const float* b,   // Matrix B [K x N]
    float* c,         // Output C [M x N]
    int M, int N, int K
)
```

Standard matrix multiplication: C = A @ B

```c
void yinsen_gemm(
    const float* a,     // Matrix A [M x K]
    const float* b,     // Matrix B [K x N]
    const float* bias,  // Bias vector [N]
    float* c,           // Output C [M x N]
    int M, int N, int K,
    float alpha,        // Scale for matmul
    float beta          // Scale for bias
)
```

General matrix multiply: C = alpha * (A @ B) + beta * bias

### Reductions

```c
float yinsen_reduce_sum(const float* x, int n)   // Sum of elements
float yinsen_reduce_mean(const float* x, int n)  // Mean of elements
float yinsen_reduce_max(const float* x, int n)   // Maximum element
```

### Layer Normalization

```c
void yinsen_layer_norm(
    const float* x,       // Input [n]
    const float* gamma,   // Scale [n]
    const float* beta,    // Shift [n]
    float* out,           // Output [n]
    int n,
    float eps             // Epsilon for numerical stability
)
```

Computes: `out = gamma * (x - mean) / sqrt(var + eps) + beta`

---

## cfc.h - Closed-form Continuous-time Networks

### Types

```c
typedef struct {
    int input_dim;
    int hidden_dim;
    const float* W_gate;    // [hidden_dim, input_dim + hidden_dim]
    const float* b_gate;    // [hidden_dim]
    const float* W_cand;    // [hidden_dim, input_dim + hidden_dim]
    const float* b_cand;    // [hidden_dim]
    const float* tau;       // [hidden_dim] or [1]
    int tau_shared;         // 1 if single tau for all neurons
} CfCParams;

typedef struct {
    int input_dim;
    int hidden_dim;
    const float* W_gate;
    const float* b_gate;
    const float* W_cand;
    const float* b_cand;
    const float* decay;     // Precomputed exp(-dt/tau) [hidden_dim]
} CfCParamsFixed;

typedef struct {
    int hidden_dim;
    int output_dim;
    const float* W_out;     // [hidden_dim, output_dim]
    const float* b_out;     // [output_dim]
} CfCOutputParams;
```

### CfC Cell

```c
void yinsen_cfc_cell(
    const float* x,         // Input [input_dim]
    const float* h_prev,    // Previous hidden state [hidden_dim]
    float dt,               // Time step
    const CfCParams* params,
    float* h_new            // Output: new hidden state [hidden_dim]
)
```

Single CfC cell forward pass. Computes:
```
gate = sigmoid(W_gate @ [x; h_prev] + b_gate)
candidate = tanh(W_cand @ [x; h_prev] + b_cand)
decay = exp(-dt / tau)
h_new = (1 - gate) * h_prev * decay + gate * candidate
```

### CfC Cell (Fixed dt)

```c
void yinsen_cfc_cell_fixed(
    const float* x,
    const float* h_prev,
    const CfCParamsFixed* params,
    float* h_new
)
```

Faster version with precomputed decay (for fixed sample rates).

### Sequence Processing

```c
void yinsen_cfc_forward(
    const float* inputs,    // [seq_len, input_dim]
    int seq_len,
    float dt,
    const CfCParams* params,
    const float* h_init,    // Initial state [hidden_dim] or NULL
    float* outputs,         // [seq_len, hidden_dim]
    float* h_final          // Final state [hidden_dim] or NULL
)
```

Process a sequence through the CfC cell.

### Output Projection

```c
void yinsen_cfc_output(
    const float* h,                 // Hidden state [hidden_dim]
    const CfCOutputParams* params,
    float* output                   // Output [output_dim]
)
```

Linear projection: `output = h @ W_out + b_out`

```c
void yinsen_cfc_output_softmax(
    const float* h,
    const CfCOutputParams* params,
    float* probs                    // Probabilities [output_dim]
)
```

Linear projection followed by softmax.

### Utilities

```c
size_t yinsen_cfc_memory_footprint(const CfCParams* params)
```

Returns the memory size in bytes for the CfC parameters.

---

## ternary.h - 1.58-bit Weight Operations

The core of Yinsen: ternary weights {-1, 0, +1} eliminate multiplication in forward pass.

### Trit Encoding

```c
#define TRIT_ZERO  0x0  /* 00 - explicit "ignore this input" */
#define TRIT_POS   0x1  /* 01 - add */
#define TRIT_NEG   0x3  /* 11 - subtract */
```

4 trits pack into 1 byte (2 bits each).

### Low-Level Functions

```c
int8_t trit_unpack(uint8_t packed, int pos)  // Extract trit at position 0-3
uint8_t trit_encode(int8_t val)               // Encode -1/0/+1 to 2 bits
uint8_t trit_pack4(int8_t t0, int8_t t1, int8_t t2, int8_t t3)  // Pack 4 trits
```

### Ternary Dot Product

```c
float ternary_dot(
    const uint8_t* w_packed,  // Packed trit weights
    const float* x,           // Input vector
    int n                     // Vector length
)
```

Computes `y = Σ(x[i] where w[i]=+1) - Σ(x[i] where w[i]=-1)`. No multiplication.

### Ternary Matrix-Vector Multiply

```c
void ternary_matvec(
    const uint8_t* W_packed,  // Packed trit matrix [M x N]
    const float* x,           // Input [N]
    float* y,                 // Output [M]
    int M, int N
)

void ternary_matvec_bias(
    const uint8_t* W_packed,
    const float* x,
    const float* bias,        // Bias [M]
    float* y,
    int M, int N
)
```

### Quantization Functions

```c
// Threshold-based quantization
void ternary_quantize(
    const float* weights,     // Input float weights
    uint8_t* packed,          // Output packed trits
    int n,
    float threshold           // |w| > threshold -> +/-1
)

// BitNet b1.58 absmean quantization (recommended)
void ternary_quantize_absmean(
    const float* weights,
    uint8_t* packed,
    int n
)
// Automatically adapts to weight distribution

float ternary_absmean_scale(const float* weights, int n)
// Returns the absmean scale factor (γ)
```

### Unpacking

```c
void ternary_unpack_to_float(
    const uint8_t* packed,
    float* weights,           // Output: -1.0, 0.0, or +1.0
    int n
)
```

### Int8 Activation Quantization

For fully integer forward pass (except nonlinearities).

```c
typedef struct {
    float scale;
    int8_t zero_point;  // Always 0 for symmetric
} TernaryQuantParams;

void ternary_quantize_activations(
    const float* x,           // Float activations
    int8_t* x_q,              // Quantized int8 output
    int n,
    TernaryQuantParams* params
)

void ternary_dequantize_activations(
    const int8_t* x_q,
    float* x,
    int n,
    const TernaryQuantParams* params
)
```

### Integer Dot Product

```c
int32_t ternary_dot_int8(
    const uint8_t* w_packed,
    const int8_t* x_q,        // Quantized activations
    int n
)
// Returns int32 accumulator; multiply by scale to get float

void ternary_matvec_int8(
    const uint8_t* W_packed,
    const int8_t* x_q,
    int32_t* y_int,           // Output: int32 accumulators
    int M, int N
)
```

### Memory Utilities

```c
size_t ternary_bytes(int n)                    // Bytes for n trits
size_t ternary_matrix_bytes(int M, int N)      // Bytes for M×N matrix

void ternary_memory_stats(
    int n,
    size_t* ternary_bytes_out,
    size_t* float_bytes_out,
    float* compression_ratio
)
```

### Statistics

```c
typedef struct {
    int total;
    int positive;     // Count of +1
    int negative;     // Count of -1
    int zeros;        // Count of 0
    float sparsity;   // zeros / total
} TernaryStats;

void ternary_stats(const uint8_t* packed, int n, TernaryStats* stats)

int ternary_count_nonzero(const uint8_t* packed, int n)
int ternary_count_zeros(const uint8_t* packed, int n)
int ternary_count_positive(const uint8_t* packed, int n)
int ternary_count_negative(const uint8_t* packed, int n)
float ternary_sparsity(const uint8_t* packed, int n)
```

### Energy Estimation

Based on Horowitz 2014 (7nm process).

```c
float ternary_matvec_energy_pj(int M, int N)   // Ternary path energy
float float_matvec_energy_pj(int M, int N)     // Float path energy
float ternary_energy_savings_ratio(int M, int N)  // ~43x typical
```

---

## cfc_ternary.h - Ternary CfC Networks

CfC with ternary weights. Same update rule, 4-5x less memory.

### Types

```c
typedef struct {
    int input_dim;
    int hidden_dim;
    const uint8_t* W_gate;    // Packed ternary [hidden_dim, input+hidden]
    const float* b_gate;      // Bias (still float) [hidden_dim]
    const uint8_t* W_cand;    // Packed ternary [hidden_dim, input+hidden]
    const float* b_cand;
    const float* tau;         // Time constant [hidden_dim] or [1]
    int tau_shared;
} CfCTernaryParams;

typedef struct {
    int hidden_dim;
    int output_dim;
    const uint8_t* W_out;     // Packed ternary [output_dim, hidden_dim]
    const float* b_out;
} CfCTernaryOutputParams;
```

### CfC Ternary Cell

```c
void yinsen_cfc_ternary_cell(
    const float* x,           // Input [input_dim]
    const float* h_prev,      // Previous hidden state [hidden_dim]
    float dt,                 // Time step (>= 0)
    const CfCTernaryParams* params,
    float* h_new              // Output [hidden_dim]
)
```

Update rule:
```
gate = sigmoid(ternary_dot(W_gate, [x; h_prev]) + b_gate)
candidate = tanh(ternary_dot(W_cand, [x; h_prev]) + b_cand)
decay = exp(-dt / tau)
h_new = (1 - gate) * h_prev * decay + gate * candidate
```

### Output Projection

```c
void yinsen_cfc_ternary_output(
    const float* h,
    const CfCTernaryOutputParams* params,
    float* output
)

void yinsen_cfc_ternary_output_softmax(
    const float* h,
    const CfCTernaryOutputParams* params,
    float* probs
)
```

### Sequence Processing

```c
void yinsen_cfc_ternary_forward(
    const float* inputs,      // [seq_len, input_dim]
    int seq_len,
    float dt,
    const CfCTernaryParams* params,
    const float* h_init,      // [hidden_dim] or NULL
    float* outputs,           // [seq_len, hidden_dim]
    float* h_final            // [hidden_dim] or NULL
)
```

### Memory Comparison

```c
size_t yinsen_cfc_ternary_weight_bytes(const CfCTernaryParams* params)

void yinsen_cfc_ternary_memory_comparison(
    const CfCTernaryParams* params,
    size_t* ternary_bytes,
    size_t* float_bytes,
    float* ratio              // Typically 4-5x
)
```

---

## entromorph.h - Evolution Engine

> **WARNING: FALSIFIED**
>
> EntroMorph evolution does NOT work. Falsification testing revealed:
> - 100/100 runs "converge" but 0/100 have >10% confidence margin
> - Solutions predict ~0.5 for all inputs (random chance)
> - Fragile to 1% noise (88% accuracy vs expected 100%)
>
> **Do not use for learning tasks.** The component functions (RNG, genesis,
> mutation, export) work correctly, but the evolution process does not
> produce learned solutions.
>
> See [FALSIFICATION_ENTROMORPH.md](FALSIFICATION_ENTROMORPH.md) for details.

### Types

```c
typedef struct {
    uint8_t input_dim;
    uint8_t hidden_dim;
    uint8_t output_dim;
    float tau[ENTROMORPH_MAX_HIDDEN];
    float W_gate[ENTROMORPH_MAX_HIDDEN * ENTROMORPH_MAX_CONCAT];
    float b_gate[ENTROMORPH_MAX_HIDDEN];
    float W_cand[ENTROMORPH_MAX_HIDDEN * ENTROMORPH_MAX_CONCAT];
    float b_cand[ENTROMORPH_MAX_HIDDEN];
    float W_out[ENTROMORPH_MAX_HIDDEN * ENTROMORPH_MAX_OUTPUT];
    float b_out[ENTROMORPH_MAX_OUTPUT];
    float fitness;
    uint32_t id;
    uint32_t generation;
} LiquidGenome;

typedef struct {
    uint64_t state;
} EntroRNG;

typedef struct {
    float weight_mutation_rate;
    float weight_mutation_std;
    float tau_mutation_rate;
    float tau_mutation_std;
} MutationParams;
```

### Random Number Generator

```c
void entro_rng_seed(EntroRNG* rng, uint64_t seed)
uint64_t entro_rng_next(EntroRNG* rng)
float entro_rng_float(EntroRNG* rng)           // [0, 1)
float entro_rng_range(EntroRNG* rng, float min, float max)
float entro_rng_gaussian(EntroRNG* rng, float mean, float std)
uint32_t entro_rng_int(EntroRNG* rng, uint32_t max)
```

Fast Xorshift64 PRNG for evolution.

### Genome Operations

```c
void entro_genesis(
    LiquidGenome* genome,
    int in_dim, int hid_dim, int out_dim,
    EntroRNG* rng,
    uint32_t id
)
```

Initialize a genome with random weights (Xavier initialization).

```c
void entro_mutate(
    LiquidGenome* genome,
    const MutationParams* params,
    EntroRNG* rng
)
```

Mutate a genome's weights and time constants.

### Conversion

```c
void entro_genome_to_params(
    const LiquidGenome* genome,
    CfCParams* cell_params,
    CfCOutputParams* out_params
)
```

Convert a genome to CfC parameters for evaluation.

### Export

```c
void entro_export_header(
    const LiquidGenome* genome,
    const char* name,
    FILE* out
)
```

Export a genome as a C header file with frozen weights.

### Constants

```c
#define ENTROMORPH_MAX_INPUT   16
#define ENTROMORPH_MAX_HIDDEN  32
#define ENTROMORPH_MAX_OUTPUT  8

static const MutationParams MUTATION_DEFAULT = {
    .weight_mutation_rate = 0.1f,
    .weight_mutation_std = 0.1f,
    .tau_mutation_rate = 0.05f,
    .tau_mutation_std = 0.2f,
};
```

---

## See Also

- [EDGE_CASES.md](EDGE_CASES.md) - Edge case behavior for ternary ops, CfC, and quantization
- [CLAIMS.md](CLAIMS.md) - Verification status of all claims
- [VERIFICATION.md](VERIFICATION.md) - Complete test verification report
- [FALSIFICATION_ENTROMORPH.md](FALSIFICATION_ENTROMORPH.md) - Why EntroMorph evolution is broken
- [EXAMPLES.md](EXAMPLES.md) - Usage examples for each module
