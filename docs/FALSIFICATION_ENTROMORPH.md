# EntroMorph Falsification Report

**Date:** 2026-01-26
**Status:** CRITICAL ISSUES FOUND

## Executive Summary

The EntroMorph evolution engine appears to "solve" XOR, but falsification testing reveals it does NOT actually learn the XOR function. The "solutions" are numerical coincidences where predictions near 0.5 happen to fall on the correct side of the decision boundary.

## Test Results

### Initial Claims (Before Falsification)

| Claim | Status |
|-------|--------|
| Evolution converges on XOR | ✓ 100/100 seeds converge |
| Solutions are deterministic | ✓ Same seed = same result |
| Evolution produces useful networks | **FALSIFIED** |

### Falsification Findings

#### 1. Solution Confidence Analysis

**Test:** Check confidence margins of 100 "successful" XOR solutions

| Metric | Result |
|--------|--------|
| Solutions with >10% confidence margin | 0/100 |
| Solutions with >20% confidence margin | 0/100 |
| Solutions with >30% confidence margin | 0/100 |
| Solutions with >40% confidence margin | 0/100 |

**Conclusion:** Every solution predicts values within 10% of 0.5. These are not learned solutions.

#### 2. Noise Robustness

**Test:** Add small noise to inputs and check accuracy

| Noise Level | Accuracy |
|-------------|----------|
| 0% (baseline) | 100% |
| 1% | 88.2% |
| 5% | 86.2% |
| 10% | 87.0% |
| 20% | 86.8% |

**Conclusion:** Solutions are fragile. A truly learned XOR should be robust to small input noise.

#### 3. Root Cause Analysis

**Finding 1: Initialization produces neutral networks**
- 10,000 random genomes average prediction: 0.5000 for ALL inputs
- Only 0.2% of random genomes get 4/4 correct (by chance)

**Finding 2: Cross-entropy fitness is WRONG for this task**
- Manual confident XOR solution: fitness = -19.6
- Random near-0.5 solutions: fitness = -2.77
- **All 10,000 random solutions have BETTER fitness than the confident one!**

This is because cross-entropy heavily penalizes confident wrong predictions. Staying near 0.5 is the "safe" strategy.

**Finding 3: Exhaustive search confirms**
- Searched 1,000,000 random genomes
- 1,995 achieved 4/4 correct (0.2%)
- **ZERO had >20% confidence margin**
- Genesis fundamentally cannot produce confident XOR solutions

#### 4. Why Does Evolution "Converge"?

The evolution loop finds genomes where:
1. All 4 predictions are very close to 0.5
2. Due to tiny numerical variations, they happen to fall on the correct side
3. This counts as "solved" but has no learning

Example "solution":
```
[0,0] -> 0.500 (target: 0, margin: 0.0%)
[0,1] -> 0.606 (target: 1, margin: 21.3%)
[1,0] -> 0.516 (target: 1, margin: 3.2%)
[1,1] -> 0.449 (target: 0, margin: 10.2%)
```

The [0,0] case has 0% margin - it's literally on the decision boundary.

## Implications

### For Yinsen

1. **The "TESTED" status for EntroMorph convergence is MISLEADING**
   - Evolution runs complete, but solutions are meaningless
   
2. **Current architecture cannot learn from evolution alone**
   - Genesis initialization is fundamentally flawed
   - Fitness function doesn't guide toward good solutions
   
3. **Claims must be downgraded**

### What Would Actually Work

1. **Different fitness function**
   - Add confidence margin penalty
   - Use hinge loss instead of cross-entropy
   
2. **Different initialization**
   - Scale weights to produce varied predictions
   - Use layer-wise initialization schemes
   
3. **Different mutation**
   - Larger mutations to escape local optima
   - Adaptive mutation rates

## Updated Claims

| Original Claim | Updated Status | Evidence |
|----------------|----------------|----------|
| Evolution converges on XOR | **MISLEADING** | Finds boundary artifacts, not solutions |
| XOR convergence: 5/5 runs | **MISLEADING** | 0/100 solutions have >10% confidence |
| Evolution produces useful networks | **FALSE** | Solutions fragile to 1% noise |
| RNG determinism verified | TRUE | Same seed = same result |
| Genesis produces valid genomes | TRUE | Genomes are valid, just not useful |

## Recommendations

1. **Do NOT claim EntroMorph "works"** until solutions have meaningful confidence margins
2. **Implement confidence-aware fitness** before further testing
3. **Add minimum margin to success criteria** (e.g., >20% from decision boundary)
4. **Test on tasks where 0.5 is not a viable strategy** (e.g., regression)

## Files

- `test/test_entromorph_falsify.c` - Initial falsification tests
- `test/test_entromorph_deep.c` - Confidence analysis
- `test/test_entromorph_diagnosis.c` - Root cause analysis

## Conclusion

EntroMorph evolution is **not functional** for learning tasks. It passes trivial correctness tests by exploiting a loophole (predictions near decision boundary). The architecture, initialization, and fitness function need significant changes before it can be considered a working evolution engine.
