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
ASFLAGS = -march=armv9-a+sme

# Source files
LIB_C_SRC = ternary_sme.c
LIB_ASM_SRC = sme_kernels.s
TEST_SRC = test_sme.c
BENCH_SRC = bench_sme.c

# Output
LIB = libternary_sme.a
TEST_BIN = test_sme
BENCH_BIN = bench_sme

# Objects
LIB_C_OBJ = $(LIB_C_SRC:.c=.o)
LIB_ASM_OBJ = $(LIB_ASM_SRC:.s=.o)
LIB_OBJ = $(LIB_C_OBJ) $(LIB_ASM_OBJ)
TEST_OBJ = $(TEST_SRC:.c=.o)
BENCH_OBJ = $(BENCH_SRC:.c=.o)

.PHONY: all test bench clean

all: $(LIB) $(TEST_BIN)

$(LIB): $(LIB_OBJ)
	ar rcs $@ $^
	@echo "Built $@ with ASM SME kernels"

$(TEST_BIN): $(TEST_OBJ) $(LIB)
	$(CC) $(CFLAGS) -o $@ $< -L. -lternary_sme -lm
	@echo "Built $@"

$(BENCH_BIN): $(BENCH_OBJ) $(LIB)
	$(CC) $(CFLAGS) -o $@ $< -L. -lternary_sme -lm
	@echo "Built $@"

%.o: %.c
	$(CC) $(CFLAGS) -c -o $@ $<

%.o: %.s
	$(AS) $(ASFLAGS) -c -o $@ $<

test: $(TEST_BIN)
	./$(TEST_BIN)

bench: $(BENCH_BIN)
	./$(BENCH_BIN)

clean:
	rm -f $(LIB_C_OBJ) $(LIB_ASM_OBJ) $(TEST_OBJ) $(BENCH_OBJ)
	rm -f $(LIB) $(TEST_BIN) $(BENCH_BIN)

# Dependencies
ternary_sme.o: ternary_sme.c ternary_sme.h
sme_kernels.o: sme_kernels.s
test_sme.o: test_sme.c ternary_sme.h
bench_sme.o: bench_sme.c ternary_sme.h

# Install to parent for integration
install: $(LIB)
	cp $(LIB) ../lib/
	cp ternary_sme.h ../include/
	@echo "Installed to ../lib and ../include"
