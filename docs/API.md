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

## entromorph.h - Evolution Engine

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
