# Makefile for NEON Ternary Kernels

CC = clang
# Enable dotprod for SDOT instruction on M4
# Use -mcpu=apple-m4 for best optimization on Apple Silicon
CFLAGS = -Wall -Wextra -O3 -ffast-math -std=c11 -mcpu=apple-m4

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
