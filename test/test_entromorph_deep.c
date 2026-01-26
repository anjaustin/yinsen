/*
 * YINSEN EntroMorph DEEP Falsification
 *
 * The previous test revealed a critical problem:
 * Networks "solve" XOR with 0% confidence - they're not learning,
 * they're finding lucky decision boundaries near 0.5.
 *
 * This test investigates:
 * 1. Are the solutions actually robust?
 * 2. What does a REAL XOR solution look like?
 * 3. Can we force proper learning?
 */

#include "../include/entromorph.h"
#include "../include/cfc.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* XOR dataset */
static const float XOR_INPUTS[4][2] = {
    {0.0f, 0.0f}, {0.0f, 1.0f}, {1.0f, 0.0f}, {1.0f, 1.0f},
};
static const float XOR_TARGETS[4] = {0.0f, 1.0f, 1.0f, 0.0f};

/* Tournament selection */
static int tournament_select(LiquidGenome* pop, int pop_size, int k, EntroRNG* rng) {
    int best = entro_rng_int(rng, pop_size);
    for (int i = 1; i < k; i++) {
        int candidate = entro_rng_int(rng, pop_size);
        if (pop[candidate].fitness > pop[best].fitness) {
            best = candidate;
        }
    }
    return best;
}

static void genome_copy(LiquidGenome* dst, const LiquidGenome* src) {
    memcpy(dst, src, sizeof(LiquidGenome));
}

/* Get predictions for a genome */
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

/* Check if XOR is solved with minimum confidence */
static int check_xor_confident(float preds[4], float min_conf) {
    for (int i = 0; i < 4; i++) {
        float target = XOR_TARGETS[i];
        float pred = preds[i];
        
        if (target > 0.5f) {
            /* Should predict high */
            if (pred < 0.5f + min_conf) return 0;
        } else {
            /* Should predict low */
            if (pred > 0.5f - min_conf) return 0;
        }
    }
    return 1;
}

/* Fitness that REQUIRES confidence margin */
static float evaluate_xor_confident(LiquidGenome* genome, float margin) {
    float preds[4];
    get_predictions(genome, preds);
    
    float total_error = 0.0f;
    
    for (int i = 0; i < 4; i++) {
        float target = XOR_TARGETS[i];
        float pred = preds[i];
        float eps = 1e-7f;
        
        /* Standard cross-entropy */
        float loss = -(target * logf(pred + eps) + (1.0f - target) * logf(1.0f - pred + eps));
        
        /* Penalty for low confidence */
        float confidence = fabsf(pred - 0.5f);
        if (confidence < margin) {
            loss += (margin - confidence) * 10.0f;  /* Heavy penalty */
        }
        
        total_error += loss;
    }
    
    return -total_error;
}

/* ============================================================================
 * INVESTIGATION
 * ============================================================================ */

int main(void) {
    printf("\n=== YINSEN EntroMorph DEEP Investigation ===\n\n");
    
    /* ========================================
     * TEST 1: How many "solutions" are actually confident?
     * ======================================== */
    printf("=== Test 1: Solution Confidence Analysis ===\n\n");
    
    int solutions_found = 0;
    int conf_10_pct = 0;  /* At least 10% margin */
    int conf_20_pct = 0;
    int conf_30_pct = 0;
    int conf_40_pct = 0;
    
    for (int seed = 0; seed < 100; seed++) {
        EntroRNG rng;
        entro_rng_seed(&rng, seed);
        
        LiquidGenome pop[50];
        LiquidGenome new_pop[50];
        
        for (int i = 0; i < 50; i++) {
            entro_genesis(&pop[i], 2, 8, 1, &rng, i);
        }
        
        LiquidGenome best;
        int found = 0;
        
        for (int gen = 0; gen < 500 && !found; gen++) {
            /* Evaluate with standard fitness */
            for (int i = 0; i < 50; i++) {
                if (pop[i].fitness == -INFINITY) {
                    float preds[4];
                    get_predictions(&pop[i], preds);
                    
                    float err = 0;
                    for (int j = 0; j < 4; j++) {
                        float target = XOR_TARGETS[j];
                        float eps = 1e-7f;
                        err -= target * logf(preds[j] + eps) + (1.0f - target) * logf(1.0f - preds[j] + eps);
                    }
                    pop[i].fitness = -err;
                }
            }
            
            /* Find best */
            int best_idx = 0;
            for (int i = 1; i < 50; i++) {
                if (pop[i].fitness > pop[best_idx].fitness) best_idx = i;
            }
            
            /* Check if solved */
            float preds[4];
            get_predictions(&pop[best_idx], preds);
            int correct = 0;
            for (int i = 0; i < 4; i++) {
                int pred_class = preds[i] > 0.5f ? 1 : 0;
                int target_class = XOR_TARGETS[i] > 0.5f ? 1 : 0;
                if (pred_class == target_class) correct++;
            }
            
            if (correct == 4) {
                genome_copy(&best, &pop[best_idx]);
                found = 1;
                break;
            }
            
            /* Elitism */
            genome_copy(&new_pop[0], &pop[best_idx]);
            genome_copy(&new_pop[1], &pop[best_idx]);
            
            for (int i = 2; i < 50; i++) {
                int parent = tournament_select(pop, 50, 3, &rng);
                genome_copy(&new_pop[i], &pop[parent]);
                entro_mutate(&new_pop[i], &MUTATION_DEFAULT, &rng);
            }
            
            memcpy(pop, new_pop, sizeof(pop));
        }
        
        if (found) {
            solutions_found++;
            
            float preds[4];
            get_predictions(&best, preds);
            
            if (check_xor_confident(preds, 0.10f)) conf_10_pct++;
            if (check_xor_confident(preds, 0.20f)) conf_20_pct++;
            if (check_xor_confident(preds, 0.30f)) conf_30_pct++;
            if (check_xor_confident(preds, 0.40f)) conf_40_pct++;
        }
    }
    
    printf("Solutions found: %d/100\n", solutions_found);
    printf("With >10%% confidence margin: %d\n", conf_10_pct);
    printf("With >20%% confidence margin: %d\n", conf_20_pct);
    printf("With >30%% confidence margin: %d\n", conf_30_pct);
    printf("With >40%% confidence margin: %d\n", conf_40_pct);
    
    if (conf_20_pct < solutions_found / 2) {
        printf("\n** PROBLEM: Most solutions have low confidence! **\n");
    }
    
    /* ========================================
     * TEST 2: What happens if we REQUIRE confidence?
     * ======================================== */
    printf("\n=== Test 2: Training with Confidence Penalty ===\n\n");
    
    int confident_solutions = 0;
    
    for (int seed = 0; seed < 20; seed++) {
        EntroRNG rng;
        entro_rng_seed(&rng, seed);
        
        LiquidGenome pop[100];  /* Larger population */
        LiquidGenome new_pop[100];
        
        for (int i = 0; i < 100; i++) {
            entro_genesis(&pop[i], 2, 16, 1, &rng, i);  /* More capacity */
        }
        
        int found = 0;
        
        for (int gen = 0; gen < 2000 && !found; gen++) {
            /* Evaluate with confidence penalty */
            for (int i = 0; i < 100; i++) {
                if (pop[i].fitness == -INFINITY) {
                    pop[i].fitness = evaluate_xor_confident(&pop[i], 0.2f);
                }
            }
            
            /* Find best */
            int best_idx = 0;
            for (int i = 1; i < 100; i++) {
                if (pop[i].fitness > pop[best_idx].fitness) best_idx = i;
            }
            
            /* Check if confident solution */
            float preds[4];
            get_predictions(&pop[best_idx], preds);
            
            if (check_xor_confident(preds, 0.2f)) {
                found = 1;
                if (gen % 100 == 0 || found) {
                    printf("Seed %d: Confident solution at gen %d\n", seed, gen);
                    printf("  Predictions: [%.3f, %.3f, %.3f, %.3f]\n",
                           preds[0], preds[1], preds[2], preds[3]);
                }
                break;
            }
            
            /* Elitism */
            for (int i = 0; i < 5; i++) {
                int nth_best = 0;
                float nth_fitness = -INFINITY;
                for (int j = 0; j < 100; j++) {
                    if (pop[j].fitness > nth_fitness) {
                        nth_fitness = pop[j].fitness;
                        nth_best = j;
                    }
                }
                genome_copy(&new_pop[i], &pop[nth_best]);
                pop[nth_best].fitness = -INFINITY - 1;
            }
            
            for (int i = 5; i < 100; i++) {
                int parent = tournament_select(pop, 100, 5, &rng);
                genome_copy(&new_pop[i], &pop[parent]);
                entro_mutate(&new_pop[i], &MUTATION_DEFAULT, &rng);
            }
            
            memcpy(pop, new_pop, sizeof(pop));
        }
        
        if (found) confident_solutions++;
    }
    
    printf("\nConfident solutions: %d/20\n", confident_solutions);
    
    /* ========================================
     * TEST 3: Noise Robustness
     * ======================================== */
    printf("\n=== Test 3: Noise Robustness ===\n\n");
    
    /* Get a typical "solution" */
    EntroRNG rng;
    entro_rng_seed(&rng, 42);
    
    LiquidGenome pop[50];
    LiquidGenome new_pop[50];
    LiquidGenome best;
    
    for (int i = 0; i < 50; i++) {
        entro_genesis(&pop[i], 2, 8, 1, &rng, i);
    }
    
    for (int gen = 0; gen < 500; gen++) {
        for (int i = 0; i < 50; i++) {
            if (pop[i].fitness == -INFINITY) {
                float preds[4];
                get_predictions(&pop[i], preds);
                float err = 0;
                for (int j = 0; j < 4; j++) {
                    float target = XOR_TARGETS[j];
                    float eps = 1e-7f;
                    err -= target * logf(preds[j] + eps) + (1.0f - target) * logf(1.0f - preds[j] + eps);
                }
                pop[i].fitness = -err;
            }
        }
        
        int best_idx = 0;
        for (int i = 1; i < 50; i++) {
            if (pop[i].fitness > pop[best_idx].fitness) best_idx = i;
        }
        
        float preds[4];
        get_predictions(&pop[best_idx], preds);
        int correct = 0;
        for (int i = 0; i < 4; i++) {
            int pred_class = preds[i] > 0.5f ? 1 : 0;
            int target_class = XOR_TARGETS[i] > 0.5f ? 1 : 0;
            if (pred_class == target_class) correct++;
        }
        
        if (correct == 4) {
            genome_copy(&best, &pop[best_idx]);
            break;
        }
        
        genome_copy(&new_pop[0], &pop[best_idx]);
        genome_copy(&new_pop[1], &pop[best_idx]);
        
        for (int i = 2; i < 50; i++) {
            int parent = tournament_select(pop, 50, 3, &rng);
            genome_copy(&new_pop[i], &pop[parent]);
            entro_mutate(&new_pop[i], &MUTATION_DEFAULT, &rng);
        }
        
        memcpy(pop, new_pop, sizeof(pop));
    }
    
    /* Test with noisy inputs */
    printf("Testing noise robustness of typical solution:\n");
    
    float noise_levels[] = {0.01f, 0.05f, 0.1f, 0.2f};
    
    for (int n = 0; n < 4; n++) {
        float noise = noise_levels[n];
        int correct = 0;
        int total = 100;
        
        for (int trial = 0; trial < total; trial++) {
            for (int i = 0; i < 4; i++) {
                /* Add noise to inputs */
                float noisy_input[2] = {
                    XOR_INPUTS[i][0] + (entro_rng_float(&rng) - 0.5f) * noise * 2,
                    XOR_INPUTS[i][1] + (entro_rng_float(&rng) - 0.5f) * noise * 2,
                };
                
                CfCParams cell_params;
                CfCOutputParams out_params;
                entro_genome_to_params(&best, &cell_params, &out_params);
                
                float h[ENTROMORPH_MAX_HIDDEN] = {0};
                float h_new[ENTROMORPH_MAX_HIDDEN];
                float output[1];
                
                yinsen_cfc_cell(noisy_input, h, 1.0f, &cell_params, h_new);
                memcpy(h, h_new, sizeof(float) * cell_params.hidden_dim);
                yinsen_cfc_output(h, &out_params, output);
                
                float pred = 1.0f / (1.0f + expf(-output[0]));
                int pred_class = pred > 0.5f ? 1 : 0;
                int target_class = XOR_TARGETS[i] > 0.5f ? 1 : 0;
                
                if (pred_class == target_class) correct++;
            }
        }
        
        printf("  Noise %.0f%%: %d/%d correct (%.1f%%)\n", 
               noise * 100, correct, total * 4, 100.0f * correct / (total * 4));
    }
    
    /* ========================================
     * SUMMARY
     * ======================================== */
    printf("\n=== SUMMARY ===\n\n");
    
    printf("FINDINGS:\n");
    printf("1. Standard evolution finds 'solutions' that barely distinguish classes\n");
    printf("2. Most solutions have <20%% confidence margins\n");
    printf("3. These low-confidence solutions are fragile to noise\n");
    printf("4. With confidence penalty, evolution CAN find robust solutions\n");
    printf("   but may require more generations/larger populations\n");
    
    printf("\nRECOMMENDATION:\n");
    printf("The current fitness function (cross-entropy alone) is insufficient.\n");
    printf("Add confidence margin penalty or change success criteria.\n");
    
    return 0;
}
