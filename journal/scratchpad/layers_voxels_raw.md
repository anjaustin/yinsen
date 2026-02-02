# Raw Thoughts: Layers, Voxels, and What Ternary CfC Actually Needs Next

## Stream of Consciousness

The v2 experiment just told us something loud: width is king, clever gradients don't matter, trajectory distillation actively hurts. On sine. But sine is trivially easy — a single-frequency smooth function. The ternary h=32 student hit MSE 0.000362, nearly matching the float teacher at 0.000257. We're at the noise floor. There's nothing left to improve.

So the question "would layers or voxels help" is really two questions: (a) would they help on sine, and (b) would they help on something harder where we actually need them.

Answer to (a) is almost certainly no. Sine is solved. Adding machinery to a solved problem just adds parameters to trip over.

Answer to (b) is the interesting one and we don't know yet because we haven't found what breaks ternary CfC.

Let me think about what layers actually ARE in a recurrent context. A stacked CfC is: x → CfC_1 → CfC_2 → ... → CfC_L → y. At each timestep, x feeds into layer 1, layer 1's hidden state feeds into layer 2 as "input", etc. Each layer has its own hidden state, its own gate, its own timescale. This is how deep RNNs work (stacked LSTMs, etc.).

Why would this help for ternary? Each ternary weight is a sign decision: add, subtract, or ignore. A single layer has to make all sign decisions simultaneously. With multiple layers, the INPUTS to layer 2 are already nonlinearly transformed — they're the hidden state of layer 1, which has been through sigmoid gating and tanh activation. So layer 2 is making sign decisions about features that are already processed. This is function composition. f(g(x)) can represent things that neither f nor g can alone.

But here's the thing: CfC already HAS a recurrence. The hidden state at time t depends on the hidden state at time t-1. So a single-layer CfC is already implicitly deep through time. Unrolling T timesteps gives you a T-layer deep network. Adding spatial layers gives you depth in a different axis — representation depth at each timestep, vs temporal depth across timesteps.

When does representation depth matter? When the mapping from input features to useful representations is complex. If x is raw audio, you need many layers to go from waveform → spectral features → phonemes. If x is already sin(t), you don't need any feature extraction.

So layers help when: the input is complex, the desired representation is far from the raw input, and the transformation is hierarchical.

Now VOXELS. This is more unusual. What does it mean to give a hidden state spatial structure?

Option 1: 3D grid of neurons. h is not a flat vector but h[x][y][z]. Each neuron connects only to its neighbors. This is a convolutional prior applied to the hidden state. Connectivity is sparse and local.

Option 2: Multi-resolution. h has coarse and fine scales. Coarse neurons capture slow/global dynamics, fine neurons capture fast/local dynamics. Like wavelet decomposition of the hidden state.

Option 3: Geometric hidden state. h lives on a manifold (sphere, torus). The ternary weights define discrete flows on this manifold.

For ternary specifically, voxels could be interesting because: LOCAL CONNECTIVITY = SPARSE WEIGHT MATRICES. If each neuron only connects to 6 neighbors (3D grid), then W_gate is extremely sparse — most entries are 0. And ternary is already about sparsity (0 means "ignore"). A voxel structure would make the sparsity structured rather than learned. Could be faster (fewer ops) and might regularize well.

But there's a tension: CfC with time constants + voxels gives you a spatiotemporal system. The decay exp(-dt/tau) varies per neuron, so different regions of the voxel grid evolve at different speeds. This is literally a reaction-diffusion system if you squint. Wave-like dynamics could emerge.

Is that useful? For sine prediction, absolutely not. For modeling physical systems (fluid dynamics, brain activity, cardiac rhythms), maybe very useful.

What actually scares me: we might build layers and voxels and get great results on a harder task, but not know whether the improvement came from the architecture or just from having more parameters. The v2 experiment's biggest lesson was that width (= more parameters) is the dominant effect. We need to control for parameter count.

What scares me more: we still haven't found a task that BREAKS ternary CfC. Without a failure, we can't measure whether architectural changes help. We need a task where h=32 flat ternary clearly fails, so we have room to improve.

Candidates for harder tasks:
- Multi-frequency: predict sin(t) + sin(sqrt(2)*t) + sin(pi*t). Three incommensurate frequencies. The hidden state needs to track multiple phases.
- Mackey-Glass: dx/dt = beta*x(t-tau)/(1+x(t-tau)^n) - gamma*x(t). Chaotic for certain parameters. Sensitive to initial conditions. Quantization noise might compound catastrophically.
- Lorenz: three coupled ODEs. Classic chaotic system.
- Sequential MNIST: classify a 28x28 image by reading it pixel by pixel (784 timesteps). This is the standard RNN benchmark.
- Copy task: memorize a sequence and reproduce it after a delay. Tests long-term memory.

Actually, the copy task is interesting because it directly tests what quantization destroys. To memorize a sequence, the hidden state must losslessly encode past inputs. Ternary gating might not have the precision to route information through the recurrence without corruption.

Wait — there's something Delta Observer-related here. The transient clustering phenomenon: during training, semantic clusters form and dissolve. If we're watching ternary CfC train on a harder task, we might see the same thing in the hidden state trajectories. The sign patterns (which inputs get +1, which get -1) might reorganize during training in a way that mirrors the transient clustering. The voxel structure could make these dynamics visible — you could literally watch wave-like patterns of sign changes propagate through the hidden state grid during training.

That's speculative. But it connects back to the user's Delta Observer work.

## Questions Arising

- What task actually BREAKS h=32 flat ternary? We need a failure to have something to fix.
- Is the benefit of layers (representation depth) orthogonal to the benefit of width (capacity)?
- Could voxel-structured sparsity compensate for narrower width by being more parameter-efficient?
- Should we control experiments by parameter count or by hidden dimension?
- Is there a task where temporal dynamics (CfC's strength) matter more than representation capacity?
- Could the copy task reveal a fundamental limit of ternary recurrence?
- Does structured connectivity (voxels) give any advantage over random sparse connectivity?

## First Instincts

1. Layers won't help until the task requires hierarchical feature extraction. Sine doesn't. Multi-frequency might.
2. Voxels are more interesting than layers because they change the STRUCTURE of computation, not just the depth. Local connectivity + ternary sparsity is a natural fit.
3. We need to find what breaks ternary CfC before building anything. The experiment should be: try harder tasks with the current architecture, find the failure mode, THEN ask whether layers/voxels address that failure mode.
4. The right next experiment is NOT "add layers" — it's "find the wall."
5. Copy task is the most information-theoretic test. It asks: can ternary CfC maintain fidelity of stored information through recurrent dynamics?
