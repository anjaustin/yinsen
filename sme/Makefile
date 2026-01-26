# Makefile for Yinsen SME 16x16 Ternary Kernels
#
# Targets:
#   make          - Build library and tests (without SME intrinsics)
#   make sme      - Build with SME intrinsics (requires -march=armv9-a+sme)
#   make test     - Run tests
#   make bench    - Run benchmarks
#   make clean    - Clean build artifacts
#
# On M4, use 'make sme' to enable actual SME instructions.
# On non-M4 or for testing, 'make' builds with reference fallback.

CC = clang
CFLAGS_COMMON = -Wall -Wextra -Wpedantic -O3 -ffast-math
CFLAGS_STD = $(CFLAGS_COMMON) -std=c11
CFLAGS_SME = $(CFLAGS_COMMON) -std=c11 -march=armv9-a+sme -D__ARM_FEATURE_SME=1

# Default: build without SME intrinsics (uses reference implementation)
CFLAGS = $(CFLAGS_STD)

# Source files
LIB_SRC = ternary_sme.c
TEST_SRC = test_sme.c
BENCH_SRC = bench_sme.c

# Output
LIB = libternary_sme.a
TEST_BIN = test_sme
BENCH_BIN = bench_sme

# Objects
LIB_OBJ = $(LIB_SRC:.c=.o)
TEST_OBJ = $(TEST_SRC:.c=.o)
BENCH_OBJ = $(BENCH_SRC:.c=.o)

.PHONY: all sme test bench clean

all: $(LIB) $(TEST_BIN)

# Build with SME intrinsics enabled
sme: CFLAGS = $(CFLAGS_SME)
sme: clean all
	@echo "Built with SME intrinsics enabled"

$(LIB): $(LIB_OBJ)
	ar rcs $@ $^
	@echo "Built $@"

$(TEST_BIN): $(TEST_OBJ) $(LIB)
	$(CC) $(CFLAGS) -o $@ $< -L. -lternary_sme -lm
	@echo "Built $@"

$(BENCH_BIN): $(BENCH_OBJ) $(LIB)
	$(CC) $(CFLAGS) -o $@ $< -L. -lternary_sme -lm
	@echo "Built $@"

%.o: %.c
	$(CC) $(CFLAGS) -c -o $@ $<

test: $(TEST_BIN)
	./$(TEST_BIN)

bench: $(BENCH_BIN)
	./$(BENCH_BIN)

clean:
	rm -f $(LIB_OBJ) $(TEST_OBJ) $(BENCH_OBJ)
	rm -f $(LIB) $(TEST_BIN) $(BENCH_BIN)

# Dependencies
ternary_sme.o: ternary_sme.c ternary_sme.h
test_sme.o: test_sme.c ternary_sme.h
bench_sme.o: bench_sme.c ternary_sme.h

# Install to parent for integration
install: $(LIB)
	cp $(LIB) ../lib/
	cp ternary_sme.h ../include/
	@echo "Installed to ../lib and ../include"
