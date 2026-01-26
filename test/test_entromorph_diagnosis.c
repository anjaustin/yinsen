/*
 * YINSEN EntroMorph Diagnosis
 *
 * ROOT CAUSE ANALYSIS: Why can't evolution find confident XOR solutions?
 *
 * Hypotheses:
 * 1. CfC architecture can't represent XOR with these weights
 * 2. Fitness landscape has no gradient toward confident solutions
 * 3. Mutation is destroying good solutions
 * 4. Initial weights are all in a bad basin
 */

#include "../include/entromorph.h"
#include "../include/cfc.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

static const float XOR_INPUTS[4][2] = {
    {0.0f, 0.0f}, {0.0f, 1.0f}, {1.0f, 0.0f}, {1.0f, 1.0f},
};
static const float XOR_TARGETS[4] = {0.0f, 1.0f, 1.0f, 0.0f};

/* Get predictions */
static void get_predictions(LiquidGenome* genome, float preds[4]) {
    CfCParams cell_params;
    CfCOutputParams out_params;
    entro_genome_to_params(genome, &cell_params, &out_params);
    
    for (int i = 0; i < 4; i++) {
        float h[ENTROMORPH_MAX_HIDDEN] = {0};
        float h_new[ENTROMORPH_MAX_HIDDEN];
        float output[1];
        
        yinsen_cfc_cell(XOR_INPUTS[i], h, 1.0f, &cell_params, h_new);
        memcpy(h, h_new, sizeof(float) * cell_params.hidden_dim);
        yinsen_cfc_output(h, &out_params, output);
        
        preds[i] = 1.0f / (1.0f + expf(-output[0]));
    }
}

/* ============================================================================
 * HYPOTHESIS 1: Can CfC even represent XOR?
 * Try to manually construct a solution
 * ============================================================================ */
static void test_manual_xor_construction(void) {
    printf("=== Hypothesis 1: Can CfC represent XOR? ===\n\n");
    
    /* XOR can be computed as: XOR(a,b) = OR(AND(a, NOT(b)), AND(NOT(a), b))
     * Or equivalently: (a + b - 2*a*b) for binary inputs
     * 
     * For a neural network with sigmoid output:
     * We need output > 0.5 when exactly one input is 1
     * We need output < 0.5 when both are same
     * 
     * Let's try a simple 2-hidden-unit network:
     * h1 = sigmoid(w1*a + w2*b + b1)  -- detects "a AND b"
     * h2 = sigmoid(w3*a + w4*b + b2)  -- detects "a OR b"  
     * out = sigmoid(v1*h1 + v2*h2 + bout)
     * 
     * XOR = OR AND NOT(AND) = h2 AND NOT(h1)
     */
    
    LiquidGenome genome;
    memset(&genome, 0, sizeof(genome));
    
    genome.input_dim = 2;
    genome.hidden_dim = 4;  /* Need capacity for XOR pattern */
    genome.output_dim = 1;
    
    /* Initialize time constants (won't matter much for dt=1) */
    for (int i = 0; i < 4; i++) {
        genome.tau[i] = 1.0f;
    }
    
    /* Manual weight construction attempt
     * CfC: h_new = decay*h + (1-decay)*gate*candidate
     * With dt=1, tau=1: decay = exp(-1) ≈ 0.37
     * 
     * For a feedforward-like behavior, we want gate≈1 always,
     * then candidate drives the output.
     */
    
    const int in_dim = 2;
    const int hid_dim = 4;
    const int concat_dim = in_dim + hid_dim;
    
    /* Gate weights: want gate ≈ 1 (sigmoid(large positive)) */
    for (int i = 0; i < hid_dim; i++) {
        genome.b_gate[i] = 5.0f;  /* Bias to keep gate open */
    }
    
    /* Candidate weights: this is where XOR computation happens
     * Unit 0: detect (a AND b) → output high when both inputs high
     * Unit 1: detect (a XOR b) → output high when exactly one input high
     */
    
    /* Unit 0: AND detector: tanh(10*a + 10*b - 15) 
     * a=0,b=0: tanh(-15) ≈ -1
     * a=1,b=0: tanh(-5) ≈ -1  
     * a=0,b=1: tanh(-5) ≈ -1
     * a=1,b=1: tanh(5) ≈ 1
     */
    genome.W_cand[0 * concat_dim + 0] = 10.0f;  /* w for input a */
    genome.W_cand[0 * concat_dim + 1] = 10.0f;  /* w for input b */
    genome.b_cand[0] = -15.0f;
    
    /* Unit 1: OR detector: tanh(10*a + 10*b - 5)
     * a=0,b=0: tanh(-5) ≈ -1
     * a=1,b=0: tanh(5) ≈ 1
     * a=0,b=1: tanh(5) ≈ 1
     * a=1,b=1: tanh(15) ≈ 1
     */
    genome.W_cand[1 * concat_dim + 0] = 10.0f;
    genome.W_cand[1 * concat_dim + 1] = 10.0f;
    genome.b_cand[1] = -5.0f;
    
    /* Units 2,3: unused, keep small */
    
    /* Output: XOR = OR AND NOT(AND) = h1 XOR-like pattern
     * We want: out = sigmoid(v1*h1 + v2*h2 + b)
     * where h1 is AND, h2 is OR
     * XOR = OR - AND in binary logic
     * So: out = sigmoid(-large*AND + large*OR)
     */
    genome.W_out[0] = -10.0f;  /* Subtract AND */
    genome.W_out[1] = 10.0f;   /* Add OR */
    genome.b_out[0] = -5.0f;   /* Bias for good range */
    
    printf("Testing manually constructed XOR network:\n");
    float preds[4];
    get_predictions(&genome, preds);
    
    for (int i = 0; i < 4; i++) {
        printf("  Input [%.0f, %.0f]: pred=%.4f, target=%.0f, %s\n",
               XOR_INPUTS[i][0], XOR_INPUTS[i][1],
               preds[i], XOR_TARGETS[i],
               ((preds[i] > 0.5f) == (XOR_TARGETS[i] > 0.5f)) ? "CORRECT" : "WRONG");
    }
    
    /* Check confidence */
    float min_margin = 1.0f;
    for (int i = 0; i < 4; i++) {
        float margin = fabsf(preds[i] - 0.5f);
        if (margin < min_margin) min_margin = margin;
    }
    printf("\n  Min confidence margin: %.1f%%\n", min_margin * 100 * 2);
    
    if (min_margin > 0.1f) {
        printf("  SUCCESS: Manual construction CAN achieve >20%% margins!\n");
    } else {
        printf("  FAIL: Even manual construction has low confidence\n");
    }
}

/* ============================================================================
 * HYPOTHESIS 2: Random initialization landscape
 * ============================================================================ */
static void test_initialization_landscape(void) {
    printf("\n=== Hypothesis 2: Initial Population Landscape ===\n\n");
    
    EntroRNG rng;
    entro_rng_seed(&rng, 12345);
    
    int correct_counts[5] = {0, 0, 0, 0, 0};  /* 0-4 correct predictions */
    float avg_preds[4] = {0, 0, 0, 0};
    int total = 10000;
    
    for (int i = 0; i < total; i++) {
        LiquidGenome genome;
        entro_genesis(&genome, 2, 8, 1, &rng, i);
        
        float preds[4];
        get_predictions(&genome, preds);
        
        int correct = 0;
        for (int j = 0; j < 4; j++) {
            avg_preds[j] += preds[j];
            if ((preds[j] > 0.5f) == (XOR_TARGETS[j] > 0.5f)) correct++;
        }
        correct_counts[correct]++;
    }
    
    printf("Correct predictions in random initial population (%d samples):\n", total);
    for (int i = 0; i <= 4; i++) {
        printf("  %d/4 correct: %d (%.1f%%)\n", i, correct_counts[i], 100.0f * correct_counts[i] / total);
    }
    
    printf("\nAverage predictions across random genomes:\n");
    for (int i = 0; i < 4; i++) {
        printf("  Input [%.0f, %.0f]: avg=%.4f (target=%.0f)\n",
               XOR_INPUTS[i][0], XOR_INPUTS[i][1],
               avg_preds[i] / total, XOR_TARGETS[i]);
    }
    
    /* For random networks, if avg ≈ 0.5 for all inputs, 
     * then evolution is just doing random search near decision boundary */
    
    float total_avg = 0;
    for (int i = 0; i < 4; i++) total_avg += avg_preds[i] / total;
    total_avg /= 4;
    
    printf("\n  Grand average: %.4f\n", total_avg);
    if (fabsf(total_avg - 0.5f) < 0.1f) {
        printf("  INSIGHT: Random networks average near 0.5 - initialization is neutral\n");
    }
}

/* ============================================================================
 * HYPOTHESIS 3: Fitness landscape analysis
 * ============================================================================ */
static void test_fitness_landscape(void) {
    printf("\n=== Hypothesis 3: Fitness Landscape ===\n\n");
    
    /* Compare fitness of random solutions vs our manual good solution */
    
    /* Create manual good solution */
    LiquidGenome good;
    memset(&good, 0, sizeof(good));
    good.input_dim = 2;
    good.hidden_dim = 4;
    good.output_dim = 1;
    for (int i = 0; i < 4; i++) good.tau[i] = 1.0f;
    
    const int concat_dim = 6;
    for (int i = 0; i < 4; i++) good.b_gate[i] = 5.0f;
    
    good.W_cand[0 * concat_dim + 0] = 10.0f;
    good.W_cand[0 * concat_dim + 1] = 10.0f;
    good.b_cand[0] = -15.0f;
    good.W_cand[1 * concat_dim + 0] = 10.0f;
    good.W_cand[1 * concat_dim + 1] = 10.0f;
    good.b_cand[1] = -5.0f;
    
    good.W_out[0] = -10.0f;
    good.W_out[1] = 10.0f;
    good.b_out[0] = -5.0f;
    
    /* Compute fitness of good solution */
    float good_preds[4];
    get_predictions(&good, good_preds);
    
    float good_fitness = 0;
    for (int i = 0; i < 4; i++) {
        float target = XOR_TARGETS[i];
        float pred = good_preds[i];
        float eps = 1e-7f;
        good_fitness -= target * logf(pred + eps) + (1.0f - target) * logf(1.0f - pred + eps);
    }
    good_fitness = -good_fitness;
    
    printf("Manual good solution:\n");
    printf("  Predictions: [%.3f, %.3f, %.3f, %.3f]\n", 
           good_preds[0], good_preds[1], good_preds[2], good_preds[3]);
    printf("  Fitness: %.4f\n", good_fitness);
    
    /* Compare to random solutions */
    EntroRNG rng;
    entro_rng_seed(&rng, 42);
    
    float sum_random_fitness = 0;
    float max_random_fitness = -INFINITY;
    int random_better = 0;
    
    for (int i = 0; i < 10000; i++) {
        LiquidGenome genome;
        entro_genesis(&genome, 2, 8, 1, &rng, i);
        
        float preds[4];
        get_predictions(&genome, preds);
        
        float fitness = 0;
        for (int j = 0; j < 4; j++) {
            float target = XOR_TARGETS[j];
            float pred = preds[j];
            float eps = 1e-7f;
            fitness -= target * logf(pred + eps) + (1.0f - target) * logf(1.0f - pred + eps);
        }
        fitness = -fitness;
        
        sum_random_fitness += fitness;
        if (fitness > max_random_fitness) max_random_fitness = fitness;
        if (fitness > good_fitness) random_better++;
    }
    
    printf("\nRandom population (10000 samples):\n");
    printf("  Average fitness: %.4f\n", sum_random_fitness / 10000);
    printf("  Best fitness: %.4f\n", max_random_fitness);
    printf("  Random solutions better than manual: %d\n", random_better);
    
    if (random_better > 0) {
        printf("\n  INSIGHT: Random solutions can have better FITNESS than confident ones!\n");
        printf("  This means cross-entropy fitness doesn't reward confidence!\n");
    }
}

/* ============================================================================
 * HYPOTHESIS 4: Can we find the good solution via exhaustive search?
 * ============================================================================ */
static void test_search_for_confident_solution(void) {
    printf("\n=== Hypothesis 4: Exhaustive Search for Confident Solutions ===\n\n");
    
    /* Generate millions of random genomes, find the most confident one */
    
    EntroRNG rng;
    entro_rng_seed(&rng, 999);
    
    LiquidGenome best;
    float best_min_margin = 0;
    int solutions_checked = 0;
    int correct_found = 0;
    int confident_found = 0;
    
    printf("Searching 1,000,000 random genomes...\n");
    
    for (int i = 0; i < 1000000; i++) {
        LiquidGenome genome;
        entro_genesis(&genome, 2, 8, 1, &rng, i);
        solutions_checked++;
        
        float preds[4];
        get_predictions(&genome, preds);
        
        /* Check if correct */
        int correct = 0;
        for (int j = 0; j < 4; j++) {
            if ((preds[j] > 0.5f) == (XOR_TARGETS[j] > 0.5f)) correct++;
        }
        
        if (correct == 4) {
            correct_found++;
            
            /* Check confidence */
            float min_margin = 1.0f;
            for (int j = 0; j < 4; j++) {
                float margin = fabsf(preds[j] - 0.5f);
                if (margin < min_margin) min_margin = margin;
            }
            
            if (min_margin > 0.1f) confident_found++;
            
            if (min_margin > best_min_margin) {
                best_min_margin = min_margin;
                memcpy(&best, &genome, sizeof(genome));
            }
        }
    }
    
    printf("\nResults:\n");
    printf("  Checked: %d genomes\n", solutions_checked);
    printf("  Correct (4/4): %d (%.2f%%)\n", correct_found, 100.0f * correct_found / solutions_checked);
    printf("  Confident (>20%% margin): %d\n", confident_found);
    printf("  Best min margin found: %.1f%%\n", best_min_margin * 100 * 2);
    
    if (confident_found == 0) {
        printf("\n  CONCLUSION: Random genesis CANNOT produce confident XOR solutions!\n");
        printf("  The initialization scheme is fundamentally flawed for this task.\n");
    }
    
    /* Show best found */
    if (correct_found > 0) {
        printf("\nBest solution found:\n");
        float preds[4];
        get_predictions(&best, preds);
        for (int i = 0; i < 4; i++) {
            printf("  [%.0f, %.0f] -> %.4f (target: %.0f)\n",
                   XOR_INPUTS[i][0], XOR_INPUTS[i][1], preds[i], XOR_TARGETS[i]);
        }
    }
}

int main(void) {
    printf("\n=== YINSEN EntroMorph ROOT CAUSE ANALYSIS ===\n\n");
    
    test_manual_xor_construction();
    test_initialization_landscape();
    test_fitness_landscape();
    test_search_for_confident_solution();
    
    printf("\n=== DIAGNOSIS COMPLETE ===\n\n");
    
    return 0;
}
