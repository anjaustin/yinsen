# Raw Thoughts: Chip Stack Applications

## Stream of Consciousness

We have 8 frozen chips. They total 49KB of source, compile to almost nothing. They need no OS, no runtime, no framework. The CfC cell IS the neural network — not a layer in a bigger thing. It's the whole brain.

What scares me: we're sitting on something that could be genuinely useful for problems that don't currently have good solutions, but we could also mistake novelty for utility. The fact that you CAN run a liquid neural network on a $1.50 MCU doesn't mean anyone SHOULD unless the temporal dynamics actually matter.

Where do temporal dynamics matter? Anywhere the signal is a process, not a snapshot. A photograph is a snapshot — CNN territory. A vibration pattern unfolding over time is a process. A heartbeat. A chemical reaction curve. A degradation trajectory. These are CfC territory.

What's different about CfC vs an LSTM on TFLite? Two things: (1) variable dt is first-class, and (2) the closed-form solution means we don't iterate. We jump. This matters when your sensor doesn't sample regularly, or when you need to express "time since last event" as a feature.

The FFT chip opens something I haven't thought about enough. FFT → CfC is a spectral-temporal pipeline. The FFT extracts frequency content at one moment. The CfC tracks how that frequency content evolves. This is literally what a spectrogram does, but incrementally, on device, in real-time, with memory.

What about non-obvious applications? What if the "sensor" isn't physical?

- Financial tick data: irregular timestamps, need to track regime changes
- Network packet timing: detect anomalies in traffic patterns
- Keystroke dynamics: biometric authentication from typing rhythm
- CAN bus messages: vehicle diagnostics from message timing patterns
- Power grid frequency: detect load events from mains frequency drift

The variable-dt thing is a genuine differentiator. Most edge ML assumes fixed sample rates. Real sensors don't work that way. GPS gives you a fix every 1-15 seconds depending on signal. CAN bus messages arrive at irregular intervals. Event cameras fire on brightness changes. BLE beacon RSSI arrives when you're in range.

What about the ternary angle? Every weight is -1, 0, or +1. That means the weights are INTERPRETABLE. You can literally read them. Weight[i][j] = +1 means "this input contributes positively to this hidden unit." This is auditable AI. Where does auditability matter?

- Medical devices (FDA wants to understand the model)
- Automotive safety (ISO 26262 requires deterministic behavior)
- Aviation (DO-178C certification)
- Nuclear/industrial safety systems
- Financial compliance (algorithmic trading oversight)

The determinism chip property — bit-identical outputs — is not just nice to have. It's a regulatory requirement in safety-critical domains.

What about the norm chip and its online normalization? Welford's streaming mean/variance is useful for drift detection. If your sensor gradually drifts (temperature compensation, aging, environmental change), the running stats track the drift and the CfC sees normalized input. But the STATS THEMSELVES are a feature. If mean is drifting, that's information. If variance is growing, something's changing.

The softmax top-K chip — when would you need top-K on edge? Multi-label classification. A vibration pattern might indicate BOTH bearing wear AND misalignment. You want the top 2-3 most likely conditions, not just argmax.

Thinking wilder now. What if you chain CfC cells with different tau values? Fast cell (tau=0.01s) for transient detection, slow cell (tau=60s) for trend tracking, both feeding into a decision cell. This is a multi-timescale architecture. Our chip stack supports it — just call CFC_CELL_GENERIC twice with different tau. The decay_chip with precomputed values makes this almost free.

What about CfC as a CONTROLLER, not just a classifier? Output isn't a class — it's a continuous control signal. PID replacement. The CfC learns the plant dynamics and outputs a control action. Where? Drone stabilization. Robotic gripper force control. HVAC damper positioning. Motor speed regulation. Anywhere you have a PID loop that someone hand-tuned.

## Questions Arising
- How small can hidden_dim be and still be useful? 4? 2?
- Is there a market for "auditable AI" specifically?
- Can we quantify the value of variable-dt vs fixed-dt in real applications?
- What's the smallest useful FFT → CfC pipeline?
- Where are PID loops being replaced by ML right now?
- What safety certifications could we realistically pursue?

## First Instincts
- Predictive maintenance is the most mature market
- Medical wearables have the highest stakes but hardest regulatory path
- Industrial control (PID replacement) is unexplored and could be huge
- The auditability angle could be a moat, not just a feature
- Multi-timescale CfC is an architecture paper waiting to happen
