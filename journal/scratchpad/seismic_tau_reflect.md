# Seismic Tau Ablation — REFLECT

## Date: February 2026

## What surprised me
The gradient of tau's importance. It's not binary (matters/doesn't matter) — it's a function of signal-to-noise ratio. Above M3, tau is irrelevant. At M2, it gives 2x speedup. At M1.5, it gives 2.4x. This suggests that at even lower magnitudes, the gap would widen further. Tau is a **sensitivity amplifier at the noise floor**, not a general performance booster.

## What I got wrong
I expected ISS tau to be "broken" on seismic data — producing degenerate hidden states like it did on ISS without pre-scaling. Instead, ISS tau actually has HIGHER H-Std (0.186 vs 0.116). The issue isn't that ISS tau kills dynamics — it's that it creates the WRONG dynamics. All ISS-tau neurons retain >99.8% per step, creating a slowly-drifting hidden state that's noisy but not informative about transients.

I also expected CfC to beat STA/LTA on all tests. It didn't. STA/LTA won on M1.5 because its energy integration over 30 seconds accumulates weak signal that CfC's instantaneous scoring misses. This is honest and useful.

## What do I believe now that I didn't before?
1. **Tau is a frequency-selective filter, not just a memory parameter.** With proper values, different neurons respond to different temporal scales. This is the CfC "time shortcut" actually working as designed — not just as a math trick, but as a genuine multi-timescale decomposition.

2. **The decay dynamic range is the key metric.** ISS failed not because tau values were "wrong" but because exp(-10/5) to exp(-10/600) is only a 7x range in decay rate. Seismic tau creates exp(-0.01/0.01)=0.368 to exp(-0.01/30)=0.9997, which is a 2700x range. More dynamic range = more temporal discrimination.

3. **CfC and STA/LTA should be combined, not compared.** CfC catches sharp onsets. STA/LTA catches gradual energy accumulation. A compound detector would get the best of both in ~2KB total memory.

## What changed vs ISS conclusions?
The ISS falsification note said: "The tau tuning story is aspirational, not proven." This needs to be updated:

**New conclusion: Tau tuning is validated at low SNR when decay dynamic range spans the signal's temporal structure. It does not matter at high SNR. The advantage is ~2-2.5x detection speed at the noise floor.**

This is a qualified, bounded result. Not "tau is magic" but "tau helps when it should, at the margin where it matters."

## Open questions
1. Would CfC with seismic tau + STA/LTA combined beat either alone? (Likely yes — complementary strengths.)
2. Does the advantage hold on REAL seismic data from SeedLink? (Synthetic P/S/surface wave model may not capture real complexity.)
3. What happens with tau values even more extreme — e.g., tau=0.001s for microseismic?
4. Can we now go back to ISS and test with tau matched to ISS timescales (minutes to hours) instead of the current 5-600s? The orbit is 92 min. Maybe tau=[60, 180, 600, 1800, 120, 360, 900, 5520] would show the same effect.
