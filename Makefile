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
.PHONY: all clean test examples

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

$(BUILD_DIR)/test_shapes: $(TEST_DIR)/test_shapes.c include/apu.h include/onnx_shapes.h
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

$(BUILD_DIR)/test_cfc: $(TEST_DIR)/test_cfc.c include/cfc.h include/onnx_shapes.h
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

$(BUILD_DIR)/test_ternary: $(TEST_DIR)/test_ternary.c include/ternary.h
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

$(BUILD_DIR)/test_cfc_ternary: $(TEST_DIR)/test_cfc_ternary.c include/cfc_ternary.h include/ternary.h include/onnx_shapes.h
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

# Examples
examples: $(BUILD_DIR) $(BUILD_DIR)/hello_xor $(BUILD_DIR)/hello_ternary

$(BUILD_DIR)/hello_xor: $(EXAMPLES_DIR)/hello_xor.c include/apu.h
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

$(BUILD_DIR)/hello_ternary: $(EXAMPLES_DIR)/hello_ternary.c include/ternary.h
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

# Clean
clean:
	rm -rf $(BUILD_DIR)

# Run all
run: test
	@echo "\nRunning examples..."
	@$(BUILD_DIR)/hello_xor
	@$(BUILD_DIR)/hello_ternary
