# Makefile for Yinsen SME 16x16 Ternary Kernels
#
# Targets:
#   make          - Build library and tests with ASM SME kernels
#   make test     - Run tests
#   make bench    - Run benchmarks
#   make clean    - Clean build artifacts
#
# SME is now always enabled via assembly kernels that work on M4.
# The library auto-detects SME at runtime and falls back to scalar if unavailable.

CC = clang
AS = clang
CFLAGS = -Wall -Wextra -Wpedantic -O3 -ffast-math -std=c11
CFLAGS_SME = $(CFLAGS) -march=armv9-a+sme
ASFLAGS = -march=armv9-a+sme

# Source files
LIB_C_SRC = ternary_sme.c sme_fused_linear.c
LIB_ASM_SRC = sme_kernels.s sme_drain_kernel.s
TEST_SRC = test_sme.c
BENCH_SRC = bench_sme.c
TEST_FUSED_SRC = test_fused_linear.c
TEST_GOLD_SRC = test_gold_standard.c

# Output
LIB = libternary_sme.a
TEST_BIN = test_sme
BENCH_BIN = bench_sme
TEST_FUSED_BIN = test_fused_linear
TEST_GOLD_BIN = test_gold_standard

# Objects
LIB_C_OBJ = $(LIB_C_SRC:.c=.o)
LIB_ASM_OBJ = $(LIB_ASM_SRC:.s=.o)
LIB_OBJ = $(LIB_C_OBJ) $(LIB_ASM_OBJ)
TEST_OBJ = $(TEST_SRC:.c=.o)
BENCH_OBJ = $(BENCH_SRC:.c=.o)
TEST_FUSED_OBJ = $(TEST_FUSED_SRC:.c=.o)
TEST_GOLD_OBJ = $(TEST_GOLD_SRC:.c=.o)

.PHONY: all test bench test-fused test-gold clean

all: $(LIB) $(TEST_BIN) $(TEST_FUSED_BIN) $(TEST_GOLD_BIN)

$(LIB): $(LIB_OBJ)
	ar rcs $@ $^
	@echo "Built $@ with ASM SME kernels"

$(TEST_BIN): $(TEST_OBJ) $(LIB)
	$(CC) $(CFLAGS) -o $@ $< -L. -lternary_sme -lm
	@echo "Built $@"

$(BENCH_BIN): $(BENCH_OBJ) $(LIB)
	$(CC) $(CFLAGS) -o $@ $< -L. -lternary_sme -lm
	@echo "Built $@"

$(TEST_FUSED_BIN): $(TEST_FUSED_OBJ) $(LIB)
	$(CC) $(CFLAGS) -o $@ $< -L. -lternary_sme -lm
	@echo "Built $@"

$(TEST_GOLD_BIN): $(TEST_GOLD_OBJ) $(LIB)
	$(CC) $(CFLAGS) -o $@ $< -L. -lternary_sme -lm
	@echo "Built $@"

%.o: %.c
	$(CC) $(CFLAGS) -c -o $@ $<

# Files with SME inline assembly need the arch flag
sme_fused_linear.o: sme_fused_linear.c
	$(CC) $(CFLAGS_SME) -c -o $@ $<

test_gold_standard.o: test_gold_standard.c
	$(CC) $(CFLAGS) -c -o $@ $<

%.o: %.s
	$(AS) $(ASFLAGS) -c -o $@ $<

test: $(TEST_BIN)
	./$(TEST_BIN)

bench: $(BENCH_BIN)
	./$(BENCH_BIN)

test-fused: $(TEST_FUSED_BIN)
	./$(TEST_FUSED_BIN)

test-gold: $(TEST_GOLD_BIN)
	./$(TEST_GOLD_BIN)

clean:
	rm -f $(LIB_C_OBJ) $(LIB_ASM_OBJ) $(TEST_OBJ) $(BENCH_OBJ) $(TEST_FUSED_OBJ) $(TEST_GOLD_OBJ)
	rm -f $(LIB) $(TEST_BIN) $(BENCH_BIN) $(TEST_FUSED_BIN) $(TEST_GOLD_BIN)

# Dependencies
ternary_sme.o: ternary_sme.c ternary_sme.h
sme_fused_linear.o: sme_fused_linear.c
sme_kernels.o: sme_kernels.s
sme_drain_kernel.o: sme_drain_kernel.s
test_sme.o: test_sme.c ternary_sme.h
test_fused_linear.o: test_fused_linear.c
test_gold_standard.o: test_gold_standard.c
bench_sme.o: bench_sme.c ternary_sme.h

# Install to parent for integration
install: $(LIB)
	cp $(LIB) ../lib/
	cp ternary_sme.h ../include/
	@echo "Installed to ../lib and ../include"
