# Raw Thoughts: Keystroke Biometric Prototype

## Stream of Consciousness

The prototype works. It compiles, runs, and the simulation shows separation between two simulated users. Score separation is 0.346 (User A avg 0.377, User B avg 0.031). That's with UNTRAINED hand-initialized weights. The pipeline discriminates based on structure alone. That's either a very good sign or a misleading one.

Let me be honest about what's actually happening. The two simulated users differ dramatically: 0.12s mean dt vs 0.35s mean dt. That's a 3x difference. A threshold on raw dt would separate them. The CfC is overkill for this test. The real question is: would it separate two users who type at the SAME speed but with different rhythmic patterns? Different digraph timings? I don't know. The simulation doesn't test that.

The User A scores worry me. They're in the 0.29-0.47 range — never above 0.5. That means even the ENROLLED user doesn't score as "MATCH" by our own threshold (0.7). The pipeline discriminates (A > B) but doesn't authenticate (A is not high). The absolute scores are too low. Why?

Looking at the scoring mechanism: it combines a MATVEC projection (logit) with a Mahalanobis-like distance penalty. The distance penalty dominates. The enrolled hidden state mean values are tiny: [-0.007, 0.033, 0.068, 0.070, 0.007, 0.014, 0.006, 0.008]. These are near-zero. The std values from enrollment are probably also small. So any deviation from near-zero produces a large normalized distance. The distance penalty crushes the score even for the enrolled user.

This is a calibration problem, not a structural one. The distance_penalty coefficient (0.5) and the output projection weights were hand-picked. With training, the network learns to produce hidden states where the enrolled user's trajectory lives in a region that maps to high scores. Right now the projection doesn't know where "good" is.

The Welford normalization on the INPUT side is doing something I didn't think about: it's normalizing dt relative to all dt's seen so far. During enrollment, the running mean/variance of dt stabilizes. During authentication of User A, the stats continue updating, but since User A's distribution matches, the normalization stays stable. When User B arrives, the normalization stats start drifting (different dt distribution), which the drift detector catches. But it also means User B's inputs get normalized differently than enrollment inputs were. The normalization itself becomes a discriminator. Is that a feature or a bug? It's both — it helps separate users but it also means the system is partially discriminating on input statistics, not just on temporal dynamics.

The key_code feature is random in simulation. Both users draw from the same uniform distribution over printable ASCII. In a real scenario, different users type different text with different character frequencies. That's additional signal. But also noise — you don't want the biometric to depend on WHAT you type, only on your rhythm. Should key_code even be an input? Or should it just be dt? If it's just dt, input_dim drops to 1, and the system is purely a rhythm recognizer. That might be cleaner.

Wait — there's a subtlety. The dt between specific digraphs (letter pairs) is characteristic. The dt between 't' and 'h' is different from the dt between 'q' and 'z' for the same person. And different between people for the same digraph. So key_code + dt together carry more signal than dt alone. But the current implementation normalizes key_code to [0,1] linearly by ASCII value, which throws away the structure. 'a' and 'b' are adjacent in ASCII but not on the keyboard. Better: normalize by keyboard position, or one-hot encode, or just use the raw byte.

Actually for a small CfC (hidden_dim=8), one-hot encoding 95 printable characters would make input_dim=95, which bloats the weight matrices to 95*8*2 = 1520 weights just for gate+candidate. That's too much for a prototype. The linear ASCII normalization is a reasonable compromise for now. A digraph-aware encoding would be the upgrade.

The simulation's random number generator is crude. The "normal distribution" is (u+v-1) which is triangular, not Gaussian. Real keystroke timing distributions are approximately log-normal, not normal. The dt is always positive with a right skew. I should be generating log-normal dt, not normal dt. This matters because the Welford normalization and the CfC dynamics respond differently to symmetric vs skewed inputs.

Something I didn't do: I didn't measure execution time. The synthesis doc estimated <200ns per keystroke. I should clock it. If it's actually 200ns, that's a real number for the demo narrative. If it's 50ns or 2000ns, I should know.

The interactive mode exists but I haven't tested it. That's the actual demo. The simulation proves the pipeline separates users; the interactive mode IS the product experience. I should test it and see how it feels.

The binary is 34KB. On a Cortex-M0 with 32KB flash, this wouldn't fit because it includes printf, simulation code, and terminal handling. The actual chip pipeline code — the part that would deploy on an MCU — is a fraction of that. I should measure what the chip pipeline alone compiles to, without the demo harness.

The drift detector works — it caught User B's different distribution. But in interactive mode, the drift detector's running stats never reset. Over a long session, the stats will converge toward the combined distribution of all users who typed. The drift detection becomes less sensitive over time. Should the drift stats reset periodically? Or use an exponential moving average instead of Welford's full-history accumulator?

The enrollment process is all-or-nothing: type 80 keys, freeze the profile. What if the user's first 10 keys are warmup (finding home row, adjusting posture) and the last 70 are their real rhythm? The warmup pollutes the enrollment profile. A better approach: discard the first N keys as warmup, or weight later keys more heavily.

The state reset between enrollment and authentication (memset h_state to 0) is necessary for a fair test — you don't want the auth phase to inherit state from enrollment. But it also means the first few auth keystrokes are from a cold start. The CfC hidden state takes a few steps to warm up. That's visible in the simulation: User A's scores start low (0.374) and generally trend upward (0.467 by key 20). There's a warmup transient. The system should probably discount or ignore the first 5-10 auth keystrokes.

## Questions Arising
- Would the pipeline separate two users with the SAME average typing speed but different rhythmic patterns?
- Is the distance penalty coefficient (0.5) too aggressive? What if it were 0.1 or 0.0?
- Should key_code be an input at all, or should we go pure rhythm (dt only)?
- What is the actual execution time per keystroke on this machine?
- How does the interactive mode feel? Is 80 keys enough enrollment?
- What does the hidden state trajectory look like over a typing session? Can we visualize it?
- Does the Welford normalization help or hurt? What if we skip normalization entirely?
- How do scores behave if User A returns after a break (cold restart)?
- What's the minimum enrollment length where separation still holds?

## First Instincts
- The absolute scores being low is the biggest problem — fix the scoring calibration
- The simulation is too easy (3x speed difference). Need a harder test.
- Execution time measurement is missing and matters for the narrative.
- The warmup transient in auth phase is a real UX issue.
- The digraph timing signal (key_code + dt together) is valuable but the encoding is naive.
- Drift detection needs a forgetting mechanism for long sessions.
- The prototype proves the pipeline works. Now it needs to prove the DIFFERENTIATION works.
