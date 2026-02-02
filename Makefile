# YINSEN Makefile
#
# Build verified ternary neural computation code.

CC = gcc
CFLAGS = -Wall -Wextra -O2 -std=c11 -I./include
LDFLAGS = -lm

# Directories
BUILD_DIR = build
TEST_DIR = test
EXAMPLES_DIR = examples

# Targets
.PHONY: all clean test test-encoding test-all falsify evolve prove4x4 examples

all: test examples

# Create build directory
$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

# Tests
test: $(BUILD_DIR) $(BUILD_DIR)/test_shapes $(BUILD_DIR)/test_cfc $(BUILD_DIR)/test_ternary $(BUILD_DIR)/test_cfc_ternary
	@echo "Running tests..."
	@$(BUILD_DIR)/test_shapes
	@$(BUILD_DIR)/test_cfc
	@$(BUILD_DIR)/test_ternary
	@$(BUILD_DIR)/test_cfc_ternary

# Run all test suites (core + encoding + falsification + evolution + 4x4 proof)
test-all: test test-encoding falsify evolve prove4x4
	@echo "All test suites complete."

# Encoding canonical test
test-encoding: $(BUILD_DIR) $(BUILD_DIR)/test_encoding_canonical
	@echo "Running canonical encoding tests..."
	@$(BUILD_DIR)/test_encoding_canonical

# Falsification tests (edge cases, try to break the code)
falsify: $(BUILD_DIR) $(BUILD_DIR)/test_falsify
	@echo "Running falsification tests..."
	@$(BUILD_DIR)/test_falsify

# Evolution tests (entromorph convergence)
evolve: $(BUILD_DIR) $(BUILD_DIR)/test_entromorph
	@echo "Running evolution tests..."
	@$(BUILD_DIR)/test_entromorph

# 4x4 ternary matvec exhaustive proof (43M combinations, ~2-5 min)
prove4x4: $(BUILD_DIR) $(BUILD_DIR)/test_ternary_4x4
	@echo "Running 4x4 ternary matvec exhaustive proof..."
	@$(BUILD_DIR)/test_ternary_4x4

$(BUILD_DIR)/test_shapes: $(TEST_DIR)/test_shapes.c include/apu.h include/onnx_shapes.h
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

$(BUILD_DIR)/test_cfc: $(TEST_DIR)/test_cfc.c include/cfc.h include/onnx_shapes.h
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

$(BUILD_DIR)/test_ternary: $(TEST_DIR)/test_ternary.c include/ternary.h include/trit_encoding.h
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

$(BUILD_DIR)/test_cfc_ternary: $(TEST_DIR)/test_cfc_ternary.c include/cfc_ternary.h include/ternary.h include/onnx_shapes.h
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

$(BUILD_DIR)/test_falsify: $(TEST_DIR)/test_falsify.c include/ternary.h include/cfc_ternary.h include/onnx_shapes.h
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

$(BUILD_DIR)/test_entromorph: $(TEST_DIR)/test_entromorph.c include/entromorph.h include/cfc.h include/onnx_shapes.h
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

$(BUILD_DIR)/test_encoding_canonical: $(TEST_DIR)/test_encoding_canonical.c include/ternary.h include/trit_encoding.h
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

$(BUILD_DIR)/test_ternary_4x4: $(TEST_DIR)/test_ternary_4x4.c include/ternary.h include/trit_encoding.h
	$(CC) $(CFLAGS) -O3 -o $@ $< $(LDFLAGS)

# Examples
examples: $(BUILD_DIR) $(BUILD_DIR)/hello_xor $(BUILD_DIR)/hello_ternary $(BUILD_DIR)/train_sine $(BUILD_DIR)/train_sine_v2 $(BUILD_DIR)/diagnostic_v3

$(BUILD_DIR)/hello_xor: $(EXAMPLES_DIR)/hello_xor.c include/apu.h
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

$(BUILD_DIR)/hello_ternary: $(EXAMPLES_DIR)/hello_ternary.c include/ternary.h
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

$(BUILD_DIR)/train_sine: $(EXAMPLES_DIR)/train_sine.c include/ternary.h include/cfc_ternary.h include/trit_encoding.h
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

$(BUILD_DIR)/train_sine_v2: $(EXAMPLES_DIR)/train_sine_v2.c include/ternary.h include/cfc_ternary.h include/trit_encoding.h
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

$(BUILD_DIR)/diagnostic_v3: $(EXAMPLES_DIR)/diagnostic_v3.c include/ternary.h include/trit_encoding.h
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

# Clean
clean:
	rm -rf $(BUILD_DIR)

# Run all
run: test
	@echo "\nRunning examples..."
	@$(BUILD_DIR)/hello_xor
	@$(BUILD_DIR)/hello_ternary
