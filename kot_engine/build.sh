#!/usr/bin/env bash
# build.sh - compiles all Kotlin sources into an executable fat-ish jar
set -e
KOTLINC="${KOTLINC:-kotlinc}"
SRC_DIR="src/main/kotlin"
OUT_JAR="inference.jar"

echo "[build] compiling with $($KOTLINC -version 2>&1)"
$KOTLINC ${SRC_DIR}/*.kt -include-runtime -d "$OUT_JAR"
echo "[build] wrote $OUT_JAR"
