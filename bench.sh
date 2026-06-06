#!/usr/bin/env bash
# bench.sh — Compare OpenBLAS vs Intel MKL for llama.c
#
# Usage:
#   ./bench.sh <model_dir> "<prompt>" [max_tokens]
#
# Builds both binaries if not already built, then runs them and
# prints a side-by-side throughput comparison.

set -euo pipefail

MODEL_DIR="${1:-}"
PROMPT="${2:-}"
MAX_TOKENS="${3:-100}"

if [[ -z "$MODEL_DIR" || -z "$PROMPT" ]]; then
    echo "Usage: $0 <model_dir> \"<prompt>\" [max_tokens]"
    echo "  max_tokens  default 100"
    exit 1
fi

# ── Colours ────────────────────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
CYAN='\033[0;36m'; BOLD='\033[1m'; RESET='\033[0m'

echo -e "${BOLD}╔══════════════════════════════════════════╗${RESET}"
echo -e "${BOLD}║   llama.c  │  OpenBLAS vs Intel MKL     ║${RESET}"
echo -e "${BOLD}╚══════════════════════════════════════════╝${RESET}"
echo

# ── Build step ─────────────────────────────────────────────────────────
BUILD_FLAGS="-O3 -march=native -lm"

echo -e "${CYAN}[1/2] Building OpenBLAS binary...${RESET}"
gcc $BUILD_FLAGS -o llama_openblas llama.c -lopenblas 2>&1 \
    && echo -e "  ${GREEN}✓ llama_openblas${RESET}" \
    || { echo -e "  ${RED}✗ OpenBLAS build failed${RESET}"; exit 1; }

echo -e "${CYAN}[2/2] Building Intel MKL binary...${RESET}"
gcc $BUILD_FLAGS -DUSE_MKL -I/usr/include/mkl -o llama_mkl llama.c -lmkl_rt 2>&1 \
    && echo -e "  ${GREEN}✓ llama_mkl${RESET}" \
    || { echo -e "  ${RED}✗ MKL build failed${RESET}"; exit 1; }

echo

# ── Helper: extract a field from the timing block ─────────────────────
extract_field() {
    local output="$1"
    local field="$2"
    # Matches "  field : VALUE" with optional units after
    echo "$output" | grep "^\s*${field}" | head -1 \
        | sed 's/.*: *//' | awk '{print $1}'
}

run_and_time() {
    local binary="$1"
    local label="$2"

    echo -e "${YELLOW}▶ Running ${label}...${RESET}"
    # Capture all output; suppress normal stdout to terminal during run
    local out
    out=$( ./"$binary" "$MODEL_DIR" "$PROMPT" "$MAX_TOKENS" 0 2>&1 )
    echo "$out" | grep -v "^\[" | grep -v "^╔\|^║\|^╚\|^══\|^---\|^$" \
        | head -20   # show generated text preview
    echo

    # Pull numbers from the timing block
    local generated prefill_rate throughput
    generated=$(   extract_field "$out" "Generated")
    prefill_rate=$(echo "$out" | grep "Prefill time" | sed 's/.*(\(.*\) tok\/s).*/\1/' | head -1)
    throughput=$(  extract_field "$out" "Throughput")

    echo "$generated $prefill_rate $throughput"
}

# ── Run both ───────────────────────────────────────────────────────────
echo -e "${BOLD}━━━ Run 1: OpenBLAS ━━━━━━━━━━━━━━━━━━━━━━━━${RESET}"
read OB_GEN OB_PRE OB_TPUT <<< $(run_and_time llama_openblas "OpenBLAS")

echo -e "${BOLD}━━━ Run 2: Intel MKL ━━━━━━━━━━━━━━━━━━━━━━━${RESET}"
read MK_GEN MK_PRE MK_TPUT <<< $(run_and_time llama_mkl "Intel MKL")

# ── Comparison table ───────────────────────────────────────────────────
echo -e "${BOLD}╔══════════════════════════════════════════════╗${RESET}"
echo -e "${BOLD}║              BENCHMARK RESULTS               ║${RESET}"
echo -e "${BOLD}╠══════════════════════╦═══════════╦═══════════╣${RESET}"
printf  "${BOLD}║ %-20s ║ %9s ║ %9s ║${RESET}\n" "Metric" "OpenBLAS" "Intel MKL"
echo -e "${BOLD}╠══════════════════════╬═══════════╬═══════════╣${RESET}"
printf  "║ %-20s ║ %8s  ║ %8s  ║\n" "Tokens generated"  "${OB_GEN}" "${MK_GEN}"
printf  "║ %-20s ║ %6s t/s ║ %6s t/s ║\n" "Prefill speed"    "${OB_PRE}" "${MK_PRE}"
printf  "║ %-20s ║ %6s t/s ║ %6s t/s ║\n" "Generate speed"   "${OB_TPUT}" "${MK_TPUT}"
echo -e "${BOLD}╚══════════════════════╩═══════════╩═══════════╝${RESET}"

# ── Speedup ────────────────────────────────────────────────────────────
if [[ -n "$OB_TPUT" && -n "$MK_TPUT" && "$OB_TPUT" != "0" ]]; then
    SPEEDUP=$(awk "BEGIN { printf \"%.2f\", ${MK_TPUT}/${OB_TPUT} }")
    if awk "BEGIN { exit !(${MK_TPUT} > ${OB_TPUT}) }"; then
        echo -e "\n  ${GREEN}MKL is ${SPEEDUP}× faster than OpenBLAS on this run.${RESET}"
    elif awk "BEGIN { exit !(${OB_TPUT} > ${MK_TPUT}) }"; then
        INV=$(awk "BEGIN { printf \"%.2f\", ${OB_TPUT}/${MK_TPUT} }")
        echo -e "\n  ${YELLOW}OpenBLAS is ${INV}× faster than MKL on this run.${RESET}"
        echo -e "  ${YELLOW}(OpenBLAS sometimes wins on AMD CPUs or small thread counts.)${RESET}"
    else
        echo -e "\n  Both backends performed similarly."
    fi
fi

echo -e "\n${CYAN}Tip: Set MKL_NUM_THREADS=1 or OMP_NUM_THREADS=1 to force single-threaded comparison.${RESET}"
echo -e "${CYAN}     MKL_VERBOSE=1 ./llama_mkl ... shows which AVX kernel MKL dispatches to.${RESET}"
