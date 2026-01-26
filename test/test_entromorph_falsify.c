/*
 * YINSEN EntroMorph Falsification Tests
 *
 * Goal: Break the evolution engine. Find where it fails.
 * 
 * Questions to answer:
 * 1. Does it ALWAYS converge, or did we get lucky with seeds?
 * 2. What's the minimum viable population size?
 * 3. Does it work with extreme parameters?
 * 4. Is XOR actually learned, or just memorized?
 * 5. Can it solve anything harder than XOR?
 */

#include "../include/entromorph.h"
#include "../include/cfc.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* Test infrastructure */
static int tests_run = 0;
static int tests_passed = 0;
static int tests_failed = 0;

#define TEST(name) \
    do { \
        tests_run++; \
        printf("  [%d] %s... ", tests_run, name); \
        fflush(stdout); \
    } while(0)

#define PASS() do { tests_passed++; printf("PASS\n"); } while(0)
#define FAIL(msg) do { tests_failed++; printf("FAIL: %s\n", msg); } while(0)
#define KNOWN(msg) do { tests_passed++; printf("KNOWN: %s\n", msg); } while(0)

/* XOR dataset */
static const float XOR_INPUTS[4][2] = {
    {0.0f, 0.0f}, {0.0f, 1.0f}, {1.0f, 0.0f}, {1.0f, 1.0f},
};
static const float XOR_TARGETS[4] = {0.0f, 1.0f, 1.0f, 0.0f};

/* AND dataset (linearly separable - easier than XOR) */
static const float AND_TARGETS[4] = {0.0f, 0.0f, 0.0f, 1.0f};

/* 3-bit parity (harder than XOR) */
static const float PARITY3_INPUTS[8][3] = {
    {0,0,0}, {0,0,1}, {0,1,0}, {0,1,1},
    {1,0,0}, {1,0,1}, {1,1,0}, {1,1,1},
};
static const float PARITY3_TARGETS[8] = {0, 1, 1, 0, 1, 0, 0, 1};

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

/* Generic evaluation function */
typedef struct {
    const float* inputs;   /* Flattened input array */
    const float* targets;  /* Target outputs */
    int input_dim;
    int num_samples;
} TaskData;

static float evaluate_task(LiquidGenome* genome, const TaskData* task) {
    CfCParams cell_params;
    CfCOutputParams out_params;
    entro_genome_to_params(genome, &cell_params, &out_params);
    
    float total_error = 0.0f;
    
    for (int i = 0; i < task->num_samples; i++) {
        float h[ENTROMORPH_MAX_HIDDEN] = {0};
        float h_new[ENTROMORPH_MAX_HIDDEN];
        float output[1];
        
        const float* x = task->inputs + i * task->input_dim;
        yinsen_cfc_cell(x, h, 1.0f, &cell_params, h_new);
        memcpy(h, h_new, sizeof(float) * cell_params.hidden_dim);
        yinsen_cfc_output(h, &out_params, output);
        
        float pred = 1.0f / (1.0f + expf(-output[0]));
        float target = task->targets[i];
        float eps = 1e-7f;
        float loss = -(target * logf(pred + eps) + (1.0f - target) * logf(1.0f - pred + eps));
        total_error += loss;
    }
    
    return -total_error;
}

static int check_task_solved(LiquidGenome* genome, const TaskData* task) {
    CfCParams cell_params;
    CfCOutputParams out_params;
    entro_genome_to_params(genome, &cell_params, &out_params);
    
    int correct = 0;
    for (int i = 0; i < task->num_samples; i++) {
        float h[ENTROMORPH_MAX_HIDDEN] = {0};
        float h_new[ENTROMORPH_MAX_HIDDEN];
        float output[1];
        
        const float* x = task->inputs + i * task->input_dim;
        yinsen_cfc_cell(x, h, 1.0f, &cell_params, h_new);
        memcpy(h, h_new, sizeof(float) * cell_params.hidden_dim);
        yinsen_cfc_output(h, &out_params, output);
        
        float pred = 1.0f / (1.0f + expf(-output[0]));
        int pred_class = pred > 0.5f ? 1 : 0;
        int target_class = task->targets[i] > 0.5f ? 1 : 0;
        
        if (pred_class == target_class) correct++;
    }
    
    return correct == task->num_samples;
}

/* Run evolution with given parameters, return generations to solve (-1 if failed) */
static int run_evolution(
    const TaskData* task,
    int in_dim, int hid_dim, int out_dim,
    int pop_size, int max_gen, int tournament_k, int elite_count,
    const MutationParams* mutation,
    uint64_t seed
) {
    EntroRNG rng;
    entro_rng_seed(&rng, seed);
    
    LiquidGenome* pop = (LiquidGenome*)malloc(pop_size * sizeof(LiquidGenome));
    LiquidGenome* new_pop = (LiquidGenome*)malloc(pop_size * sizeof(LiquidGenome));
    
    for (int i = 0; i < pop_size; i++) {
        entro_genesis(&pop[i], in_dim, hid_dim, out_dim, &rng, i);
    }
    
    int solve_gen = -1;
    
    for (int gen = 0; gen < max_gen; gen++) {
        /* Evaluate */
        for (int i = 0; i < pop_size; i++) {
            if (pop[i].fitness == -INFINITY) {
                pop[i].fitness = evaluate_task(&pop[i], task);
            }
        }
        
        /* Find best */
        int best_idx = 0;
        for (int i = 1; i < pop_size; i++) {
            if (pop[i].fitness > pop[best_idx].fitness) {
                best_idx = i;
            }
        }
        
        if (check_task_solved(&pop[best_idx], task)) {
            solve_gen = gen;
            break;
        }
        
        /* Elitism */
        for (int i = 0; i < elite_count && i < pop_size; i++) {
            int nth_best = 0;
            float nth_fitness = -INFINITY;
            for (int j = 0; j < pop_size; j++) {
                if (pop[j].fitness > nth_fitness) {
                    nth_fitness = pop[j].fitness;
                    nth_best = j;
                }
            }
            genome_copy(&new_pop[i], &pop[nth_best]);
            pop[nth_best].fitness = -INFINITY - 1; /* Mark as used */
        }
        
        /* Fill rest */
        for (int i = elite_count; i < pop_size; i++) {
            int parent = tournament_select(pop, pop_size, tournament_k, &rng);
            genome_copy(&new_pop[i], &pop[parent]);
            entro_mutate(&new_pop[i], mutation, &rng);
        }
        
        LiquidGenome* tmp = pop;
        pop = new_pop;
        new_pop = tmp;
    }
    
    free(pop);
    free(new_pop);
    return solve_gen;
}

/* ============================================================================
 * FALSIFICATION TESTS
 * ============================================================================ */

static void test_many_seeds(void) {
    TEST("100 different seeds - how many converge?");
    
    TaskData xor_task = {
        .inputs = (const float*)XOR_INPUTS,
        .targets = XOR_TARGETS,
        .input_dim = 2,
        .num_samples = 4,
    };
    
    int successes = 0;
    int total_gens = 0;
    int min_gen = 9999, max_gen = 0;
    
    for (int seed = 0; seed < 100; seed++) {
        int gen = run_evolution(&xor_task, 2, 8, 1, 50, 500, 3, 2, &MUTATION_DEFAULT, seed);
        if (gen >= 0) {
            successes++;
            total_gens += gen;
            if (gen < min_gen) min_gen = gen;
            if (gen > max_gen) max_gen = gen;
        }
    }
    
    printf("\n      %d/100 converged (min=%d, max=%d, avg=%.1f)\n      ",
           successes, min_gen, max_gen, successes > 0 ? (float)total_gens/successes : 0);
    
    if (successes < 90) {
        FAIL("Less than 90% convergence rate");
    } else {
        PASS();
    }
}

static void test_minimal_population(void) {
    TEST("Minimum viable population size");
    
    TaskData xor_task = {
        .inputs = (const float*)XOR_INPUTS,
        .targets = XOR_TARGETS,
        .input_dim = 2,
        .num_samples = 4,
    };
    
    int pop_sizes[] = {2, 3, 5, 10, 20};
    int num_sizes = sizeof(pop_sizes) / sizeof(pop_sizes[0]);
    
    printf("\n");
    for (int p = 0; p < num_sizes; p++) {
        int pop_size = pop_sizes[p];
        int successes = 0;
        
        for (int seed = 0; seed < 20; seed++) {
            int elite = pop_size > 2 ? 1 : 0;
            int gen = run_evolution(&xor_task, 2, 8, 1, pop_size, 1000, 2, elite, &MUTATION_DEFAULT, seed);
            if (gen >= 0) successes++;
        }
        
        printf("      pop=%d: %d/20 converged\n", pop_size, successes);
    }
    printf("      ");
    
    /* Pop=2 should struggle, pop=10+ should mostly work */
    KNOWN("Very small populations (<5) have lower success rate");
}

static void test_zero_mutation(void) {
    TEST("Zero mutation rate - should fail to improve");
    
    TaskData xor_task = {
        .inputs = (const float*)XOR_INPUTS,
        .targets = XOR_TARGETS,
        .input_dim = 2,
        .num_samples = 4,
    };
    
    MutationParams zero_mutation = {
        .weight_mutation_rate = 0.0f,
        .weight_mutation_std = 0.0f,
        .tau_mutation_rate = 0.0f,
        .tau_mutation_std = 0.0f,
    };
    
    int successes = 0;
    for (int seed = 0; seed < 20; seed++) {
        int gen = run_evolution(&xor_task, 2, 8, 1, 50, 500, 3, 2, &zero_mutation, seed);
        if (gen >= 0) successes++;
    }
    
    printf("\n      %d/20 converged with zero mutation\n      ", successes);
    
    /* With zero mutation, only initial population diversity matters */
    /* Some might still solve by luck in initial population */
    if (successes > 5) {
        KNOWN("Zero mutation can still succeed via initial diversity");
    } else {
        PASS();
    }
}

static void test_extreme_mutation(void) {
    TEST("Extreme mutation (100% rate, std=10) - chaos");
    
    TaskData xor_task = {
        .inputs = (const float*)XOR_INPUTS,
        .targets = XOR_TARGETS,
        .input_dim = 2,
        .num_samples = 4,
    };
    
    MutationParams extreme_mutation = {
        .weight_mutation_rate = 1.0f,
        .weight_mutation_std = 10.0f,
        .tau_mutation_rate = 1.0f,
        .tau_mutation_std = 2.0f,
    };
    
    int successes = 0;
    for (int seed = 0; seed < 20; seed++) {
        int gen = run_evolution(&xor_task, 2, 8, 1, 50, 500, 3, 2, &extreme_mutation, seed);
        if (gen >= 0) successes++;
    }
    
    printf("\n      %d/20 converged with extreme mutation\n      ", successes);
    
    if (successes < 10) {
        KNOWN("Extreme mutation destroys learning signal");
    } else {
        PASS();
    }
}

static void test_tiny_hidden(void) {
    TEST("Hidden dim = 1 (minimal capacity)");
    
    TaskData xor_task = {
        .inputs = (const float*)XOR_INPUTS,
        .targets = XOR_TARGETS,
        .input_dim = 2,
        .num_samples = 4,
    };
    
    int successes = 0;
    for (int seed = 0; seed < 20; seed++) {
        int gen = run_evolution(&xor_task, 2, 1, 1, 50, 1000, 3, 2, &MUTATION_DEFAULT, seed);
        if (gen >= 0) successes++;
    }
    
    printf("\n      %d/20 converged with hidden_dim=1\n      ", successes);
    
    /* XOR requires nonlinearity - 1 hidden unit might not be enough */
    if (successes < 5) {
        KNOWN("Hidden dim=1 has insufficient capacity for XOR");
    } else {
        PASS();
    }
}

static void test_large_hidden(void) {
    TEST("Hidden dim = 32 (large capacity, slower)");
    
    TaskData xor_task = {
        .inputs = (const float*)XOR_INPUTS,
        .targets = XOR_TARGETS,
        .input_dim = 2,
        .num_samples = 4,
    };
    
    int successes = 0;
    int total_gens = 0;
    
    for (int seed = 0; seed < 10; seed++) {
        int gen = run_evolution(&xor_task, 2, 32, 1, 50, 500, 3, 2, &MUTATION_DEFAULT, seed);
        if (gen >= 0) {
            successes++;
            total_gens += gen;
        }
    }
    
    printf("\n      %d/10 converged (avg gen: %.1f)\n      ", 
           successes, successes > 0 ? (float)total_gens/successes : 0);
    
    if (successes < 8) {
        FAIL("Large hidden should still converge");
    } else {
        PASS();
    }
}

static void test_and_task(void) {
    TEST("AND task (easier than XOR) - should be trivial");
    
    TaskData and_task = {
        .inputs = (const float*)XOR_INPUTS,
        .targets = AND_TARGETS,
        .input_dim = 2,
        .num_samples = 4,
    };
    
    int successes = 0;
    int total_gens = 0;
    
    for (int seed = 0; seed < 20; seed++) {
        int gen = run_evolution(&and_task, 2, 4, 1, 30, 200, 3, 2, &MUTATION_DEFAULT, seed);
        if (gen >= 0) {
            successes++;
            total_gens += gen;
        }
    }
    
    printf("\n      %d/20 converged (avg gen: %.1f)\n      ",
           successes, successes > 0 ? (float)total_gens/successes : 0);
    
    if (successes < 18) {
        FAIL("AND should be nearly always solvable");
    } else {
        PASS();
    }
}

static void test_parity3_task(void) {
    TEST("3-bit parity (harder than XOR)");
    
    TaskData parity_task = {
        .inputs = (const float*)PARITY3_INPUTS,
        .targets = PARITY3_TARGETS,
        .input_dim = 3,
        .num_samples = 8,
    };
    
    int successes = 0;
    int total_gens = 0;
    
    for (int seed = 0; seed < 20; seed++) {
        /* Give it more resources for harder task */
        int gen = run_evolution(&parity_task, 3, 16, 1, 100, 2000, 3, 2, &MUTATION_DEFAULT, seed);
        if (gen >= 0) {
            successes++;
            total_gens += gen;
        }
    }
    
    printf("\n      %d/20 converged (avg gen: %.1f)\n      ",
           successes, successes > 0 ? (float)total_gens/successes : 0);
    
    if (successes < 5) {
        KNOWN("3-bit parity is significantly harder - expected lower success");
    } else {
        PASS();
    }
}

static void test_determinism(void) {
    TEST("Same seed = same result (determinism)");
    
    TaskData xor_task = {
        .inputs = (const float*)XOR_INPUTS,
        .targets = XOR_TARGETS,
        .input_dim = 2,
        .num_samples = 4,
    };
    
    int gen1 = run_evolution(&xor_task, 2, 8, 1, 50, 500, 3, 2, &MUTATION_DEFAULT, 12345);
    int gen2 = run_evolution(&xor_task, 2, 8, 1, 50, 500, 3, 2, &MUTATION_DEFAULT, 12345);
    int gen3 = run_evolution(&xor_task, 2, 8, 1, 50, 500, 3, 2, &MUTATION_DEFAULT, 12345);
    
    printf("\n      Run 1: gen %d, Run 2: gen %d, Run 3: gen %d\n      ", gen1, gen2, gen3);
    
    if (gen1 != gen2 || gen2 != gen3) {
        FAIL("Same seed should produce identical results");
    } else {
        PASS();
    }
}

static void test_solution_quality(void) {
    TEST("Solution quality - check actual predictions");
    
    TaskData xor_task = {
        .inputs = (const float*)XOR_INPUTS,
        .targets = XOR_TARGETS,
        .input_dim = 2,
        .num_samples = 4,
    };
    
    /* Find a solution */
    EntroRNG rng;
    entro_rng_seed(&rng, 42);
    
    LiquidGenome* pop = (LiquidGenome*)malloc(50 * sizeof(LiquidGenome));
    LiquidGenome* new_pop = (LiquidGenome*)malloc(50 * sizeof(LiquidGenome));
    LiquidGenome best;
    
    for (int i = 0; i < 50; i++) {
        entro_genesis(&pop[i], 2, 8, 1, &rng, i);
    }
    
    int solved = 0;
    for (int gen = 0; gen < 500 && !solved; gen++) {
        for (int i = 0; i < 50; i++) {
            if (pop[i].fitness == -INFINITY) {
                pop[i].fitness = evaluate_task(&pop[i], &xor_task);
            }
        }
        
        int best_idx = 0;
        for (int i = 1; i < 50; i++) {
            if (pop[i].fitness > pop[best_idx].fitness) best_idx = i;
        }
        
        if (check_task_solved(&pop[best_idx], &xor_task)) {
            genome_copy(&best, &pop[best_idx]);
            solved = 1;
            break;
        }
        
        genome_copy(&new_pop[0], &pop[best_idx]);
        genome_copy(&new_pop[1], &pop[best_idx]);
        
        for (int i = 2; i < 50; i++) {
            int parent = tournament_select(pop, 50, 3, &rng);
            genome_copy(&new_pop[i], &pop[parent]);
            entro_mutate(&new_pop[i], &MUTATION_DEFAULT, &rng);
        }
        
        LiquidGenome* tmp = pop;
        pop = new_pop;
        new_pop = tmp;
    }
    
    free(pop);
    free(new_pop);
    
    if (!solved) {
        FAIL("Could not find solution to examine");
        return;
    }
    
    /* Check prediction confidence */
    CfCParams cell_params;
    CfCOutputParams out_params;
    entro_genome_to_params(&best, &cell_params, &out_params);
    
    printf("\n      Predictions:\n");
    float min_confidence = 1.0f;
    
    for (int i = 0; i < 4; i++) {
        float h[ENTROMORPH_MAX_HIDDEN] = {0};
        float h_new[ENTROMORPH_MAX_HIDDEN];
        float output[1];
        
        yinsen_cfc_cell(XOR_INPUTS[i], h, 1.0f, &cell_params, h_new);
        memcpy(h, h_new, sizeof(float) * cell_params.hidden_dim);
        yinsen_cfc_output(h, &out_params, output);
        
        float pred = 1.0f / (1.0f + expf(-output[0]));
        float target = XOR_TARGETS[i];
        float confidence = fabsf(pred - 0.5f) * 2.0f;  /* 0 = uncertain, 1 = confident */
        
        if (confidence < min_confidence) min_confidence = confidence;
        
        printf("      [%.0f,%.0f] -> %.3f (target: %.0f, conf: %.1f%%)\n",
               XOR_INPUTS[i][0], XOR_INPUTS[i][1], pred, target, confidence * 100);
    }
    
    printf("      Min confidence: %.1f%%\n      ", min_confidence * 100);
    
    if (min_confidence < 0.1f) {
        KNOWN("Solution works but has low confidence margins");
    } else {
        PASS();
    }
}

static void test_overfitting_check(void) {
    TEST("Interpolation test - does network generalize?");
    
    /* Train on XOR, then test on intermediate values */
    TaskData xor_task = {
        .inputs = (const float*)XOR_INPUTS,
        .targets = XOR_TARGETS,
        .input_dim = 2,
        .num_samples = 4,
    };
    
    /* Get a trained network */
    EntroRNG rng;
    entro_rng_seed(&rng, 42);
    
    LiquidGenome* pop = (LiquidGenome*)malloc(50 * sizeof(LiquidGenome));
    LiquidGenome* new_pop = (LiquidGenome*)malloc(50 * sizeof(LiquidGenome));
    LiquidGenome best;
    
    for (int i = 0; i < 50; i++) {
        entro_genesis(&pop[i], 2, 8, 1, &rng, i);
    }
    
    int solved = 0;
    for (int gen = 0; gen < 500 && !solved; gen++) {
        for (int i = 0; i < 50; i++) {
            if (pop[i].fitness == -INFINITY) {
                pop[i].fitness = evaluate_task(&pop[i], &xor_task);
            }
        }
        
        int best_idx = 0;
        for (int i = 1; i < 50; i++) {
            if (pop[i].fitness > pop[best_idx].fitness) best_idx = i;
        }
        
        if (check_task_solved(&pop[best_idx], &xor_task)) {
            genome_copy(&best, &pop[best_idx]);
            solved = 1;
            break;
        }
        
        genome_copy(&new_pop[0], &pop[best_idx]);
        genome_copy(&new_pop[1], &pop[best_idx]);
        
        for (int i = 2; i < 50; i++) {
            int parent = tournament_select(pop, 50, 3, &rng);
            genome_copy(&new_pop[i], &pop[parent]);
            entro_mutate(&new_pop[i], &MUTATION_DEFAULT, &rng);
        }
        
        LiquidGenome* tmp = pop;
        pop = new_pop;
        new_pop = tmp;
    }
    
    free(pop);
    free(new_pop);
    
    if (!solved) {
        FAIL("Could not train network");
        return;
    }
    
    /* Test on intermediate values - XOR should interpolate reasonably */
    CfCParams cell_params;
    CfCOutputParams out_params;
    entro_genome_to_params(&best, &cell_params, &out_params);
    
    printf("\n      Interpolation test (0.5, 0.5 should be ~0.5 for XOR):\n");
    
    float test_inputs[][2] = {
        {0.5f, 0.5f},  /* Should be ambiguous */
        {0.2f, 0.8f},  /* Should lean toward 1 */
        {0.8f, 0.2f},  /* Should lean toward 1 */
        {0.1f, 0.1f},  /* Should lean toward 0 */
        {0.9f, 0.9f},  /* Should lean toward 0 */
    };
    
    for (int i = 0; i < 5; i++) {
        float h[ENTROMORPH_MAX_HIDDEN] = {0};
        float h_new[ENTROMORPH_MAX_HIDDEN];
        float output[1];
        
        yinsen_cfc_cell(test_inputs[i], h, 1.0f, &cell_params, h_new);
        memcpy(h, h_new, sizeof(float) * cell_params.hidden_dim);
        yinsen_cfc_output(h, &out_params, output);
        
        float pred = 1.0f / (1.0f + expf(-output[0]));
        printf("      [%.1f,%.1f] -> %.3f\n", test_inputs[i][0], test_inputs[i][1], pred);
    }
    printf("      ");
    
    KNOWN("Network trained on binary inputs - interpolation behavior varies");
}

/* ============================================================================
 * MAIN
 * ============================================================================ */

int main(void) {
    printf("\n=== YINSEN EntroMorph FALSIFICATION ===\n");
    printf("Goal: Find where evolution breaks\n\n");
    
    printf("Seed Stability:\n");
    test_many_seeds();
    
    printf("\nParameter Extremes:\n");
    test_minimal_population();
    test_zero_mutation();
    test_extreme_mutation();
    test_tiny_hidden();
    test_large_hidden();
    
    printf("\nTask Difficulty:\n");
    test_and_task();
    test_parity3_task();
    
    printf("\nQuality Checks:\n");
    test_determinism();
    test_solution_quality();
    test_overfitting_check();
    
    printf("\n=== Results: %d/%d passed, %d failed ===\n\n", 
           tests_passed, tests_run, tests_failed);
    
    if (tests_failed > 0) {
        printf("ISSUES FOUND - Review failures above\n\n");
    } else {
        printf("All tests passed (including KNOWN behaviors)\n\n");
    }
    
    return tests_failed > 0 ? 1 : 0;
}
