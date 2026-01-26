# Raw Thoughts: Yinsen's Potential

## Stream of Consciousness

What do we actually have here? A ternary neural computation library in pure C. Header-only. 88 tests. No dependencies. The "Tri" means something now - weights are {-1, 0, +1}.

The obvious thing is "make it do something" - train a network, solve a task. But that's the naive path. Everyone does that. What's the *actual* potential?

Ternary weights are weird. They're not a compression trick - they're a different computational model. You don't multiply, you accumulate conditionally. This is closer to how you'd build a neural network in hardware. Or in a spreadsheet. Or by hand.

Wait - by hand. Could someone literally execute a ternary network with pencil and paper? The weights are just +1, -1, 0. You're adding and subtracting. A child could do it. That's... that's actually radical for interpretability.

What industries care about interpretability? Medical devices. Avionics. Finance (explainability regulations). Autonomous vehicles. Anywhere a regulator might ask "why did it do that?" and you need an answer better than "the weights are 0.7324 and -0.1892 and..."

The CfC angle is interesting too. Continuous-time means you can handle irregular sampling. Medical data is irregular. Sensor data is irregular. Financial tick data is irregular.

What about the evolution engine? EntroMorph is sitting there untested. Evolution doesn't need gradients. Ternary weights are discrete - you can't do gradient descent naturally. But you CAN evolve them. Mutation is just "flip this weight from 0 to +1" or whatever. The search space is finite (huge, but finite).

Hmm. The combination: ternary + CfC + evolution. You evolve networks that:
1. Handle irregular time series (CfC)
2. Are fully inspectable (ternary)
3. Don't require backprop (evolution)

What's the killer app? Where do all three matter?

Edge devices. Tiny MCUs. No floating point unit, limited memory. Ternary is 2 bits per weight. A meaningful network could fit in kilobytes, not megabytes.

Medical wearables? Heart monitors, glucose monitors, seizure prediction. Irregular sampling (you don't sample continuously), need for interpretability (FDA), resource constrained (battery life).

Industrial IoT? Predictive maintenance on vibration data. Irregular sampling, edge deployment, need to explain why you're shutting down a $10M machine.

But wait - we have NOTHING working end-to-end. No trained network. No benchmark. No proof that ternary CfC can learn anything useful. That's a big gap.

What's the fastest path to "this actually works"? 
- XOR sequence learning (trivial but proves the pipeline)
- Sine wave prediction (shows temporal modeling)
- ECG anomaly detection (shows real-world relevance)

The research value vs. commercial value split:
- Research: prove ternary+CfC+evolution is viable
- Commercial: solve a real problem in a regulated industry

The unfair advantage: everyone else is doing float32 transformers. We're doing something orthogonal. If it works, we're not competing - we're in a different category.

What scares me? That ternary is too limited. That you need fine-grained weights for real tasks. That this is an intellectual curiosity, not a practical tool.

What excites me? That the simplicity might be the feature. That auditability becomes possible. That a nurse could literally trace through a decision if needed.

## Questions Arising

- Can ternary CfC learn anything non-trivial?
- What's the minimum network size for useful tasks?
- How does evolution perform on ternary search spaces?
- Is there a market that desperately needs interpretable time series models?
- What would FDA/CE certification look like for an ML model?
- Could we formally verify a ternary network? (The state space is finite...)

## First Instincts

1. Prove the pipeline works: evolve → train → verify → export
2. Pick one vertical (medical wearables?)
3. Build the smallest possible demo that matters
4. The story is "auditable AI for regulated industries"
5. Don't compete with PyTorch - be orthogonal to it
