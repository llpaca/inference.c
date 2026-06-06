# Makefile for llama.c — OpenBLAS and Intel MKL backends
#
# Targets:
#   make              → build OpenBLAS binary (llama)
#   make mkl          → build Intel MKL binary (llama_mkl)
#   make all          → build both
#   make bench        → run quick benchmark (requires MODEL and PROMPT)
#   make clean        → remove binaries
#
# Variables:
#   MODEL=<path>      → model directory (required for bench)
#   PROMPT="<text>"   → prompt text     (required for bench)
#   TOKENS=100        → max tokens for bench (default 100)
#
# Examples:
#   make
#   make mkl
#   make all
#   make bench MODEL=./models/llama-3.2-1b PROMPT="The history of Rome is"
#   make bench MODEL=./models/llama-3.2-1b PROMPT="Hello world" TOKENS=50

CC      := gcc
CFLAGS  := -O3 -march=native -std=c11 -Wall -Wextra -Wno-unused-variable \
            -Wno-unused-parameter -Wno-sign-compare

# OpenBLAS
OBLAS_INC := -I/usr/include/x86_64-linux-gnu
OBLAS_LIB := -lopenblas -lm

# Intel MKL  (uses single dynamic library: libmkl_rt)
MKL_INC   := -I/usr/include/mkl
MKL_LIB   := -lmkl_rt -lm

# Benchmark settings
MODEL  ?= ./model
PROMPT ?= "Once upon a time"
TOKENS ?= 100

# ── Default target: OpenBLAS ──────────────────────────────────────────
.PHONY: all openblas mkl bench bench_openblas bench_mkl clean help

all: openblas mkl

openblas: llama
llama: llama.c
	@echo "[build] OpenBLAS → llama"
	$(CC) $(CFLAGS) $(OBLAS_INC) -o $@ $< $(OBLAS_LIB)
	@echo "  ✓  ./llama  (OpenBLAS)"

# ── MKL target ────────────────────────────────────────────────────────
mkl: llama_mkl
llama_mkl: llama.c
	@echo "[build] Intel MKL → llama_mkl"
	$(CC) $(CFLAGS) $(MKL_INC) -DUSE_MKL -o $@ $< $(MKL_LIB)
	@echo "  ✓  ./llama_mkl  (Intel MKL)"

# ── Benchmarks ────────────────────────────────────────────────────────
bench: all
	@chmod +x bench.sh
	./bench.sh "$(MODEL)" $(PROMPT) $(TOKENS)

bench_openblas: llama
	./llama "$(MODEL)" $(PROMPT) $(TOKENS)

bench_mkl: llama_mkl
	./llama_mkl "$(MODEL)" $(PROMPT) $(TOKENS)

# ── Clean ─────────────────────────────────────────────────────────────
clean:
	rm -f llama llama_mkl

# ── Help ──────────────────────────────────────────────────────────────
help:
	@echo "Targets:"
	@echo "  make             Build OpenBLAS binary (./llama)"
	@echo "  make mkl         Build MKL binary (./llama_mkl)"
	@echo "  make all         Build both"
	@echo "  make bench       Run comparison benchmark"
	@echo "  make clean       Remove binaries"
	@echo ""
	@echo "Variables (for bench):"
	@echo "  MODEL=<dir>      Path to model directory"
	@echo "  PROMPT=\"<text>\"  Prompt string"
	@echo "  TOKENS=N         Max tokens to generate (default 100)"
	@echo ""
	@echo "Manual build:"
	@echo "  OpenBLAS: gcc -O3 -march=native -o llama      llama.c -lm -lopenblas"
	@echo "  MKL:      gcc -O3 -march=native -DUSE_MKL -o llama_mkl llama.c -lm -lmkl_rt"
