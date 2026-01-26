# Makefile for NEON Ternary Kernels

CC = clang
# Enable dotprod for SDOT and i8mm for SMMLA on M4
# Use -mcpu=apple-m4 which includes all ARMv8.6 features
# Add +i8mm explicitly to ensure __ARM_FEATURE_MATMUL_INT8 is defined
CFLAGS = -Wall -Wextra -O3 -ffast-math -std=c11 -mcpu=apple-m4 -march=armv8.6-a+i8mm

# Source files
LIB_SRC = neon_ternary.c
TEST_SRC = test_neon.c

# Output
LIB = libneon_ternary.a
TEST_BIN = test_neon

.PHONY: all test clean

all: $(LIB) $(TEST_BIN)

$(LIB): neon_ternary.o
	ar rcs $@ $^
	@echo "Built $@"

$(TEST_BIN): test_neon.o $(LIB)
	$(CC) $(CFLAGS) -o $@ $< -L. -lneon_ternary
	@echo "Built $@"

%.o: %.c
	$(CC) $(CFLAGS) -c -o $@ $<

test: $(TEST_BIN)
	./$(TEST_BIN)

clean:
	rm -f *.o $(LIB) $(TEST_BIN)
