/*
 * Basic CfC_CELL Test Suite
 * Tests core functionality on Cortex-M4
 */

#include <stdio.h>
#include <string.h>
#include <math.h>
#include <assert.h>
#include "cfc_cell_chip.h"

#define EPSILON 1e-5

// Test 1: Zero input, zero state
void test_zero_input(void) {
    float x[4] = {0, 0, 0, 0};
    float h_prev[8] = {0};
    float h_new[8];
    
    float W_gate[96] = {0};
    float b_gate[8] = {0};
    float W_cand[96] = {0};
    float b_cand[8] = {0};
    float tau[1] = {1.0};
    
    CFC_CELL_GENERIC(x, h_prev, 0.01, W_gate, b_gate, 
                     W_cand, b_cand, tau, 1, 4, 8, h_new);
    
    // With zero weights, output should be zero
    for (int i = 0; i < 8; i++) {
        assert(fabs(h_new[i]) < EPSILON);
    }
    
    printf("✓ Test 1: Zero input passed\n");
}

// Test 2: Non-zero input
void test_nonzero_input(void) {
    float x[4] = {0.5, -0.3, 0.8, -0.2};
    float h_prev[8] = {0};
    float h_new[8];
    
    float W_gate[96], b_gate[8];
    float W_cand[96], b_cand[8];
    float tau[1] = {1.0};
    
    // Initialize with small random weights
    for (int i = 0; i < 96; i++) {
        W_gate[i] = 0.1;
        W_cand[i] = 0.1;
    }
    for (int i = 0; i < 8; i++) {
        b_gate[i] = 0.0;
        b_cand[i] = 0.0;
    }
    
    CFC_CELL_GENERIC(x, h_prev, 0.01, W_gate, b_gate, 
                     W_cand, b_cand, tau, 1, 4, 8, h_new);
    
    // Output should be non-zero and finite
    int nonzero_count = 0;
    for (int i = 0; i < 8; i++) {
        assert(isfinite(h_new[i]));
        if (fabs(h_new[i]) > EPSILON) nonzero_count++;
    }
    assert(nonzero_count > 0);
    
    printf("✓ Test 2: Non-zero input passed\n");
}

// Test 3: State persistence
void test_state_persistence(void) {
    float x[4] = {1.0, 0.0, 0.0, 0.0};
    float h[8] = {0};
    float h_new[8];
    
    float W_gate[96], b_gate[8];
    float W_cand[96], b_cand[8];
    float tau[1] = {1.0};
    
    for (int i = 0; i < 96; i++) {
        W_gate[i] = W_cand[i] = 0.1;
    }
    for (int i = 0; i < 8; i++) {
        b_gate[i] = b_cand[i] = 0.0;
    }
    
    // Run multiple steps
    for (int t = 0; t < 10; t++) {
        CFC_CELL_GENERIC(x, h, 0.01, W_gate, b_gate, 
                         W_cand, b_cand, tau, 1, 4, 8, h_new);
        memcpy(h, h_new, sizeof(h_new));
    }
    
    // State should have evolved
    int changed_count = 0;
    for (int i = 0; i < 8; i++) {
        if (fabs(h[i]) > EPSILON) changed_count++;
    }
    assert(changed_count > 0);
    
    printf("✓ Test 3: State persistence passed\n");
}

// Test 4: Determinism
void test_determinism(void) {
    float x[4] = {0.5, -0.3, 0.8, -0.2};
    float h_prev[8] = {0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8};
    float h_new1[8], h_new2[8];
    
    float W_gate[96], b_gate[8];
    float W_cand[96], b_cand[8];
    float tau[1] = {1.0};
    
    for (int i = 0; i < 96; i++) {
        W_gate[i] = W_cand[i] = 0.1;
    }
    for (int i = 0; i < 8; i++) {
        b_gate[i] = b_cand[i] = 0.0;
    }
    
    // Run twice with same inputs
    CFC_CELL_GENERIC(x, h_prev, 0.01, W_gate, b_gate, 
                     W_cand, b_cand, tau, 1, 4, 8, h_new1);
    CFC_CELL_GENERIC(x, h_prev, 0.01, W_gate, b_gate, 
                     W_cand, b_cand, tau, 1, 4, 8, h_new2);
    
    // Outputs should be identical
    for (int i = 0; i < 8; i++) {
        assert(h_new1[i] == h_new2[i]);
    }
    
    printf("✓ Test 4: Determinism passed\n");
}

// Test 5: Different timesteps
// NOTE: h_prev must be non-zero for dt to matter, because
// decay only affects the retention term: (1-gate) * h_prev * decay.
// If h_prev=0, retention=0 regardless of dt.
void test_different_dt(void) {
    float x[4] = {0.5, -0.3, 0.8, -0.2};
    float h_prev[8] = {0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8};
    float h_new_small[8], h_new_large[8];
    
    float W_gate[96], b_gate[8];
    float W_cand[96], b_cand[8];
    float tau[1] = {1.0};
    
    for (int i = 0; i < 96; i++) {
        W_gate[i] = W_cand[i] = 0.1;
    }
    for (int i = 0; i < 8; i++) {
        b_gate[i] = b_cand[i] = 0.0;
    }
    
    // Small timestep
    CFC_CELL_GENERIC(x, h_prev, 0.001, W_gate, b_gate, 
                     W_cand, b_cand, tau, 1, 4, 8, h_new_small);
    
    // Large timestep
    CFC_CELL_GENERIC(x, h_prev, 0.1, W_gate, b_gate, 
                     W_cand, b_cand, tau, 1, 4, 8, h_new_large);
    
    // Outputs should differ
    int different_count = 0;
    for (int i = 0; i < 8; i++) {
        if (fabs(h_new_small[i] - h_new_large[i]) > EPSILON) {
            different_count++;
        }
    }
    assert(different_count > 0);
    
    printf("✓ Test 5: Different timesteps passed\n");
}

int main(void) {
    printf("CfC_CELL Test Suite\n");
    printf("===================\n\n");
    
    test_zero_input();
    test_nonzero_input();
    test_state_persistence();
    test_determinism();
    test_different_dt();
    
    printf("\n✓ All tests passed!\n");
    return 0;
}
