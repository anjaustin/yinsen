# Raw Thoughts: Can Yinsen Compete with LNNs/LFNs?

## What are LNNs/LFNs?

Liquid Neural Networks (LNNs) / Liquid Foundation Networks (LFNs) from Liquid.ai:
- Based on CfC (Closed-form Continuous-time) architecture - same as us
- Trained with gradient descent on massive datasets
- Float32/16 weights with full precision
- State-of-the-art results on time series, robotics, autonomous driving
- Recently raised $250M+ funding

## What Yinsen Has

- CfC architecture (same underlying math)
- Ternary weights {-1, 0, +1} (severe constraint)
- Evolution-based training (EntroMorph) - UNTESTED
- No gradient descent
- 100-10K parameters (tiny)
- 111 tests passing, but zero trained networks

## The Honest Answer

**No. Not yet. Probably not ever on raw accuracy.**

Here's why:

### 1. We Can't Even Train Yet

EntroMorph is untested. We have zero evidence that evolution converges on ternary CfC networks. We're comparing a concept to a shipping product.

### 2. Ternary is a Severe Constraint

BitNet b1.58 shows ternary can match float at 3B+ parameters. But:
- They use gradient descent with straight-through estimator
- They train on 100B+ tokens
- They're competing at scale where the constraint "averages out"

At 1000 parameters, ternary is MUCH more limiting. You can't represent fine gradients. Some functions may be impossible to approximate well.

### 3. Evolution vs Gradients

Gradient descent is remarkably efficient at finding good solutions. Evolution:
- Needs more evaluations
- May get stuck in local optima
- Has no gradient signal to guide search

For the same compute budget, gradients almost certainly win.

### 4. We're Playing a Different Game

LFNs optimize: accuracy, latency, capability
Yinsen optimizes: auditability, verifiability, trust

These are orthogonal objectives. Asking "can Yinsen compete on accuracy" is like asking "can a bicycle compete with a car on speed." Wrong question.

## What Yinsen COULD Compete On

### 1. Auditability
- LFNs: "Trust the model, it works"
- Yinsen: "Here's every weight, trace any decision"

No LFN can offer this. The weights are float32 - meaningless to humans.

### 2. Formal Verification
- LFNs: Impossible to formally verify (continuous state space)
- Yinsen: Feasible at small scale (finite ternary weights)

### 3. Memory Footprint
- LFN: Megabytes to gigabytes
- Yinsen: Kilobytes (ternary compression)

### 4. Determinism
- LFNs: Platform-dependent float ops
- Yinsen: Integer ops in forward pass (more deterministic)

### 5. Regulatory Acceptance
- LFNs: "Explain this to the FDA"
- Yinsen: "The network is 100 weights, each is +1, -1, or 0"

## The Real Question

Not "can Yinsen compete with LFNs" but:

**"Are there tasks where a tiny, auditable, ternary network is GOOD ENOUGH?"**

If the task is:
- Simple enough that 1000 weights can learn it
- In a regulated domain where auditability matters
- On hardware where memory/power is constrained

Then Yinsen might be the RIGHT choice, even if it's less accurate.

## What We Need To Prove

1. **Evolution converges** - Does EntroMorph work at all?
2. **Ternary learns something** - Can we solve ANY non-trivial task?
3. **The accuracy tradeoff** - How much accuracy do we lose for auditability?
4. **The sweet spot** - What tasks fit Yinsen's constraints?

## Concrete Comparison We Could Do

Take a simple benchmark that LNNs have published results on:
- Walker2D (robotics control)
- Sequential MNIST
- Some time series from their papers

Train Yinsen on the same task. Compare:
- Accuracy (we'll probably lose)
- Memory (we'll win by 10-100x)
- Auditability (we'll win completely)
- Training compute (unclear)

This would be honest science. Not "we beat LFNs" but "here's the tradeoff curve."

## The Uncomfortable Truth

If someone needs state-of-the-art accuracy on a complex task, they should use LFNs (or transformers, or whatever SOTA is).

Yinsen is for people who need:
- To explain their model to regulators
- To run on a $2 microcontroller
- To formally verify behavior
- To audit every decision

That's a real market. It's just not the same market LFNs serve.
