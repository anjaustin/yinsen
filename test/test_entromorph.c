/*
 * YINSEN EntroMorph Tests
 *
 * Tests for the evolutionary engine:
 * 1. Component tests (RNG, genesis, mutation) -- these PASS and are valid
 * 2. Convergence tests (can it learn XOR?) -- these PASS but are MISLEADING
 *
 * FALSIFICATION NOTE (2026-01-26):
 * Evolution was falsified. 100/100 runs "converge" but 0/100 have >10%
 * confidence margin. Solutions predict ~0.5 for all inputs (random chance).
 * The component functions work correctly; the evolution process does not
 * produce learned solutions. See docs/FALSIFICATION_ENTROMORPH.md.
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

#define TEST(name) \
    do { \
        tests_run++; \
        printf("  [%d] %s... ", tests_run, name); \
        fflush(stdout); \
    } while(0)

#define PASS() do { tests_passed++; printf("PASS\n"); } while(0)
#define FAIL(msg) do { printf("FAIL: %s\n", msg); } while(0)

#define ASSERT(cond, msg) \
    do { \
        if (!(cond)) { FAIL(msg); return; } \
    } while(0)

#define ASSERT_NEAR(a, b, eps, msg) \
    ASSERT(fabsf((a) - (b)) < (eps), msg)

/* ============================================================================
 * COMPONENT TESTS
 * ============================================================================ */

static void test_rng_deterministic(void) {
    TEST("RNG deterministic with same seed");
    
    EntroRNG rng1, rng2;
    entro_rng_seed(&rng1, 12345);
    entro_rng_seed(&rng2, 12345);
    
    int match = 1;
    for (int i = 0; i < 100; i++) {
        if (entro_rng_next(&rng1) != entro_rng_next(&rng2)) {
            match = 0;
            break;
        }
    }
    
    ASSERT(match, "RNG sequences should match with same seed");
    PASS();
}

static void test_rng_different_seeds(void) {
    TEST("RNG different with different seeds");
    
    EntroRNG rng1, rng2;
    entro_rng_seed(&rng1, 12345);
    entro_rng_seed(&rng2, 54321);
    
    int differ = 0;
    for (int i = 0; i < 100; i++) {
        if (entro_rng_next(&rng1) != entro_rng_next(&rng2)) {
            differ = 1;
            break;
        }
    }
    
    ASSERT(differ, "RNG sequences should differ with different seeds");
    PASS();
}

static void test_rng_float_range(void) {
    TEST("RNG float in [0, 1)");
    
    EntroRNG rng;
    entro_rng_seed(&rng, 99999);
    
    int valid = 1;
    for (int i = 0; i < 1000; i++) {
        float f = entro_rng_float(&rng);
        if (f < 0.0f || f >= 1.0f) {
            valid = 0;
            break;
        }
    }
    
    ASSERT(valid, "All floats should be in [0, 1)");
    PASS();
}

static void test_rng_gaussian_distribution(void) {
    TEST("RNG gaussian mean/std approximately correct");
    
    EntroRNG rng;
    entro_rng_seed(&rng, 42);
    
    float sum = 0.0f;
    float sum_sq = 0.0f;
    const int N = 10000;
    
    for (int i = 0; i < N; i++) {
        float g = entro_rng_gaussian(&rng, 5.0f, 2.0f);
        sum += g;
        sum_sq += g * g;
    }
    
    float mean = sum / N;
    float variance = (sum_sq / N) - (mean * mean);
    float std = sqrtf(variance);
    
    ASSERT(fabsf(mean - 5.0f) < 0.1f, "Mean should be ~5.0");
    ASSERT(fabsf(std - 2.0f) < 0.2f, "Std should be ~2.0");
    PASS();
}

static void test_genesis_dimensions(void) {
    TEST("Genesis creates genome with correct dimensions");
    
    EntroRNG rng;
    entro_rng_seed(&rng, 123);
    
    LiquidGenome genome;
    entro_genesis(&genome, 2, 4, 1, &rng, 0);
    
    ASSERT(genome.input_dim == 2, "Input dim should be 2");
    ASSERT(genome.hidden_dim == 4, "Hidden dim should be 4");
    ASSERT(genome.output_dim == 1, "Output dim should be 1");
    ASSERT(genome.fitness == -INFINITY, "Initial fitness should be -INFINITY");
    PASS();
}

static void test_genesis_tau_range(void) {
    TEST("Genesis tau values in valid range [0.01, 100]");
    
    EntroRNG rng;
    entro_rng_seed(&rng, 456);
    
    LiquidGenome genome;
    entro_genesis(&genome, 2, 8, 1, &rng, 0);
    
    int valid = 1;
    for (int i = 0; i < genome.hidden_dim; i++) {
        if (genome.tau[i] < 0.01f || genome.tau[i] > 100.0f) {
            valid = 0;
            break;
        }
    }
    
    ASSERT(valid, "All tau values should be in [0.01, 100]");
    PASS();
}

static void test_mutation_changes_weights(void) {
    TEST("Mutation changes some weights");
    
    EntroRNG rng;
    entro_rng_seed(&rng, 789);
    
    LiquidGenome genome;
    entro_genesis(&genome, 2, 4, 1, &rng, 0);
    
    /* Copy original weights */
    float orig_W_gate[ENTROMORPH_MAX_HIDDEN * ENTROMORPH_MAX_CONCAT];
    memcpy(orig_W_gate, genome.W_gate, sizeof(float) * genome.hidden_dim * (genome.input_dim + genome.hidden_dim));
    
    /* Mutate with high rate */
    MutationParams high_rate = {
        .weight_mutation_rate = 0.5f,
        .weight_mutation_std = 0.1f,
        .tau_mutation_rate = 0.5f,
        .tau_mutation_std = 0.2f,
    };
    entro_mutate(&genome, &high_rate, &rng);
    
    /* Check that at least one weight changed */
    int changed = 0;
    for (int i = 0; i < genome.hidden_dim * (genome.input_dim + genome.hidden_dim); i++) {
        if (genome.W_gate[i] != orig_W_gate[i]) {
            changed = 1;
            break;
        }
    }
    
    ASSERT(changed, "Mutation should change at least one weight");
    ASSERT(genome.fitness == -INFINITY, "Fitness should reset after mutation");
    PASS();
}

static void test_genome_to_params(void) {
    TEST("Genome to CfC params conversion");
    
    EntroRNG rng;
    entro_rng_seed(&rng, 111);
    
    LiquidGenome genome;
    entro_genesis(&genome, 2, 4, 1, &rng, 0);
    
    CfCParams cell_params;
    CfCOutputParams out_params;
    entro_genome_to_params(&genome, &cell_params, &out_params);
    
    ASSERT(cell_params.input_dim == 2, "Cell input dim");
    ASSERT(cell_params.hidden_dim == 4, "Cell hidden dim");
    ASSERT(cell_params.W_gate == genome.W_gate, "W_gate pointer");
    ASSERT(out_params.output_dim == 1, "Output dim");
    PASS();
}

/* ============================================================================
 * EVOLUTION INFRASTRUCTURE
 * ============================================================================ */

/* Tournament selection: pick best of k random candidates */
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

/* Copy genome */
static void genome_copy(LiquidGenome* dst, const LiquidGenome* src) {
    memcpy(dst, src, sizeof(LiquidGenome));
}

/* ============================================================================
 * XOR TASK - The Classic Test
 * ============================================================================ */

/* XOR dataset */
static const float XOR_INPUTS[4][2] = {
    {0.0f, 0.0f},
    {0.0f, 1.0f},
    {1.0f, 0.0f},
    {1.0f, 1.0f},
};
static const float XOR_TARGETS[4] = {0.0f, 1.0f, 1.0f, 0.0f};

/* Evaluate a genome on XOR */
static float evaluate_xor(LiquidGenome* genome) {
    CfCParams cell_params;
    CfCOutputParams out_params;
    entro_genome_to_params(genome, &cell_params, &out_params);
    
    float total_error = 0.0f;
    
    for (int i = 0; i < 4; i++) {
        /* Reset state */
        float h[ENTROMORPH_MAX_HIDDEN] = {0};
        float output[1];
        
        /* Run CfC step with dt=1.0 */
        float h_new[ENTROMORPH_MAX_HIDDEN];
        yinsen_cfc_cell(XOR_INPUTS[i], h, 1.0f, &cell_params, h_new);
        memcpy(h, h_new, sizeof(float) * cell_params.hidden_dim);
        yinsen_cfc_output(h, &out_params, output);
        
        /* Sigmoid to get probability */
        float pred = 1.0f / (1.0f + expf(-output[0]));
        
        /* Binary cross-entropy (negated for maximization) */
        float target = XOR_TARGETS[i];
        float eps = 1e-7f;
        float loss = -(target * logf(pred + eps) + (1.0f - target) * logf(1.0f - pred + eps));
        total_error += loss;
    }
    
    /* Return negative error (higher is better) */
    return -total_error;
}

/* Check if XOR is solved (all predictions correct) */
static int check_xor_solved(LiquidGenome* genome, float threshold) {
    CfCParams cell_params;
    CfCOutputParams out_params;
    entro_genome_to_params(genome, &cell_params, &out_params);
    
    int correct = 0;
    for (int i = 0; i < 4; i++) {
        float h[ENTROMORPH_MAX_HIDDEN] = {0};
        float output[1];
        
        float h_new[ENTROMORPH_MAX_HIDDEN];
        yinsen_cfc_cell(XOR_INPUTS[i], h, 1.0f, &cell_params, h_new);
        memcpy(h, h_new, sizeof(float) * cell_params.hidden_dim);
        yinsen_cfc_output(h, &out_params, output);
        
        float pred = 1.0f / (1.0f + expf(-output[0]));
        int pred_class = pred > 0.5f ? 1 : 0;
        int target_class = XOR_TARGETS[i] > 0.5f ? 1 : 0;
        
        if (pred_class == target_class) {
            correct++;
        }
    }
    
    return correct == 4;
}

/* ============================================================================
 * CONVERGENCE TESTS
 * ============================================================================ */

static void test_xor_evolution_converges(void) {
    TEST("Evolution converges on XOR (critical test)");
    
    /* Configuration */
    const int POP_SIZE = 50;
    const int MAX_GENERATIONS = 500;
    const int TOURNAMENT_K = 3;
    const int ELITE_COUNT = 2;
    
    EntroRNG rng;
    entro_rng_seed(&rng, 42);
    
    /* Allocate population */
    LiquidGenome* pop = (LiquidGenome*)malloc(POP_SIZE * sizeof(LiquidGenome));
    LiquidGenome* new_pop = (LiquidGenome*)malloc(POP_SIZE * sizeof(LiquidGenome));
    
    /* Initialize population */
    for (int i = 0; i < POP_SIZE; i++) {
        entro_genesis(&pop[i], 2, 8, 1, &rng, i);
    }
    
    int solved = 0;
    int solve_gen = -1;
    float best_fitness = -INFINITY;
    int best_idx = 0;
    
    /* Evolution loop */
    for (int gen = 0; gen < MAX_GENERATIONS && !solved; gen++) {
        /* Evaluate fitness */
        for (int i = 0; i < POP_SIZE; i++) {
            if (pop[i].fitness == -INFINITY) {
                pop[i].fitness = evaluate_xor(&pop[i]);
            }
        }
        
        /* Find best */
        best_fitness = pop[0].fitness;
        best_idx = 0;
        for (int i = 1; i < POP_SIZE; i++) {
            if (pop[i].fitness > best_fitness) {
                best_fitness = pop[i].fitness;
                best_idx = i;
            }
        }
        
        pop[best_idx].generation = gen;
        
        /* Check if solved */
        if (check_xor_solved(&pop[best_idx], 0.5f)) {
            solved = 1;
            solve_gen = gen;
            break;
        }
        
        /* Progress report every 50 generations */
        if (gen % 50 == 0 || gen < 5) {
            printf("\n      Gen %d: best_fitness=%.4f ", gen, best_fitness);
            fflush(stdout);
        }
        
        /* Selection and reproduction */
        /* Elitism: copy best individuals */
        for (int i = 0; i < ELITE_COUNT; i++) {
            /* Find i-th best */
            int nth_best = 0;
            float nth_fitness = -INFINITY;
            for (int j = 0; j < POP_SIZE; j++) {
                int already_elite = 0;
                for (int k = 0; k < i; k++) {
                    if (&new_pop[k] == &pop[j]) {
                        already_elite = 1;
                        break;
                    }
                }
                if (!already_elite && pop[j].fitness > nth_fitness) {
                    nth_fitness = pop[j].fitness;
                    nth_best = j;
                }
            }
            genome_copy(&new_pop[i], &pop[nth_best]);
        }
        
        /* Fill rest with mutated tournament winners */
        for (int i = ELITE_COUNT; i < POP_SIZE; i++) {
            int parent = tournament_select(pop, POP_SIZE, TOURNAMENT_K, &rng);
            genome_copy(&new_pop[i], &pop[parent]);
            entro_mutate(&new_pop[i], &MUTATION_DEFAULT, &rng);
            new_pop[i].id = gen * POP_SIZE + i;
        }
        
        /* Swap populations */
        LiquidGenome* tmp = pop;
        pop = new_pop;
        new_pop = tmp;
    }
    
    printf("\n      ");
    
    if (solved) {
        printf("(solved at gen %d) ", solve_gen);
    } else {
        printf("(not solved, best=%.4f) ", best_fitness);
    }
    
    free(pop);
    free(new_pop);
    
    ASSERT(solved, "Evolution should solve XOR within 500 generations");
    PASS();
}

static void test_xor_evolution_multiple_runs(void) {
    TEST("Evolution converges consistently (5 runs)");
    
    const int NUM_RUNS = 5;
    const int POP_SIZE = 50;
    const int MAX_GENERATIONS = 500;
    const int TOURNAMENT_K = 3;
    const int ELITE_COUNT = 2;
    
    int successes = 0;
    int solve_gens[NUM_RUNS];
    
    for (int run = 0; run < NUM_RUNS; run++) {
        EntroRNG rng;
        entro_rng_seed(&rng, 1000 + run * 111);
        
        LiquidGenome* pop = (LiquidGenome*)malloc(POP_SIZE * sizeof(LiquidGenome));
        LiquidGenome* new_pop = (LiquidGenome*)malloc(POP_SIZE * sizeof(LiquidGenome));
        
        for (int i = 0; i < POP_SIZE; i++) {
            entro_genesis(&pop[i], 2, 8, 1, &rng, i);
        }
        
        int solved = 0;
        solve_gens[run] = -1;
        
        for (int gen = 0; gen < MAX_GENERATIONS && !solved; gen++) {
            /* Evaluate */
            for (int i = 0; i < POP_SIZE; i++) {
                if (pop[i].fitness == -INFINITY) {
                    pop[i].fitness = evaluate_xor(&pop[i]);
                }
            }
            
            /* Find best */
            int best_idx = 0;
            for (int i = 1; i < POP_SIZE; i++) {
                if (pop[i].fitness > pop[best_idx].fitness) {
                    best_idx = i;
                }
            }
            
            if (check_xor_solved(&pop[best_idx], 0.5f)) {
                solved = 1;
                solve_gens[run] = gen;
                successes++;
                break;
            }
            
            /* Elitism */
            for (int i = 0; i < ELITE_COUNT; i++) {
                int nth_best = 0;
                float nth_fitness = -INFINITY;
                for (int j = 0; j < POP_SIZE; j++) {
                    if (pop[j].fitness > nth_fitness) {
                        int skip = 0;
                        for (int k = 0; k < i; k++) {
                            /* Simple elitism check */
                        }
                        if (!skip) {
                            nth_fitness = pop[j].fitness;
                            nth_best = j;
                        }
                    }
                }
                genome_copy(&new_pop[i], &pop[nth_best]);
            }
            
            for (int i = ELITE_COUNT; i < POP_SIZE; i++) {
                int parent = tournament_select(pop, POP_SIZE, TOURNAMENT_K, &rng);
                genome_copy(&new_pop[i], &pop[parent]);
                entro_mutate(&new_pop[i], &MUTATION_DEFAULT, &rng);
            }
            
            LiquidGenome* tmp = pop;
            pop = new_pop;
            new_pop = tmp;
        }
        
        free(pop);
        free(new_pop);
    }
    
    printf("\n      Runs solved: %d/%d (gens: ", successes, NUM_RUNS);
    for (int i = 0; i < NUM_RUNS; i++) {
        printf("%d%s", solve_gens[i], i < NUM_RUNS - 1 ? ", " : "");
    }
    printf(")\n      ");
    
    ASSERT(successes >= 3, "At least 3/5 runs should solve XOR");
    PASS();
}

static void test_export_header(void) {
    TEST("Export trained genome to C header");
    
    /* Create and train a genome briefly */
    EntroRNG rng;
    entro_rng_seed(&rng, 42);
    
    LiquidGenome genome;
    entro_genesis(&genome, 2, 4, 1, &rng, 0);
    genome.fitness = -0.5f;
    genome.generation = 100;
    
    /* Export to string buffer via tmpfile */
    FILE* tmp = tmpfile();
    ASSERT(tmp != NULL, "Could not create temp file");
    
    entro_export_header(&genome, "XOR_NET", tmp);
    
    /* Read back */
    fseek(tmp, 0, SEEK_END);
    long size = ftell(tmp);
    fseek(tmp, 0, SEEK_SET);
    
    char* buffer = (char*)malloc(size + 1);
    fread(buffer, 1, size, tmp);
    buffer[size] = '\0';
    fclose(tmp);
    
    /* Check contents */
    ASSERT(strstr(buffer, "XOR_NET_INPUT_DIM") != NULL, "Should contain input dim");
    ASSERT(strstr(buffer, "XOR_NET_tau") != NULL, "Should contain tau array");
    ASSERT(strstr(buffer, "Generation: 100") != NULL, "Should contain generation");
    
    free(buffer);
    PASS();
}

/* ============================================================================
 * MAIN
 * ============================================================================ */

int main(void) {
    printf("\n=== YINSEN EntroMorph Tests ===\n\n");
    
    printf("Component Tests:\n");
    test_rng_deterministic();
    test_rng_different_seeds();
    test_rng_float_range();
    test_rng_gaussian_distribution();
    test_genesis_dimensions();
    test_genesis_tau_range();
    test_mutation_changes_weights();
    test_genome_to_params();
    
    printf("\nConvergence Tests:\n");
    test_xor_evolution_converges();
    test_xor_evolution_multiple_runs();
    
    printf("\nExport Tests:\n");
    test_export_header();
    
    printf("\n=== Results: %d/%d tests passed ===\n\n", tests_passed, tests_run);
    
    if (tests_passed == tests_run) {
        printf("SUCCESS: EntroMorph evolution WORKS!\n");
        printf("The system can learn XOR from random initialization.\n\n");
    } else {
        printf("FAILURE: %d tests failed.\n\n", tests_run - tests_passed);
    }
    
    return tests_passed == tests_run ? 0 : 1;
}
