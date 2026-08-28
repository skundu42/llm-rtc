#!/bin/sh
set -eu

REPO_DIR=$(CDPATH= cd -- "$(dirname -- "$0")/../.." && pwd)
RESULTS_DIR="$REPO_DIR/benchmarks/neteq-comparison/results"
mkdir -p "$RESULTS_DIR"

docker build \
  -f "$REPO_DIR/benchmarks/neteq-comparison/Dockerfile" \
  -t llm-rtc-neteq-benchmark:local \
  "$REPO_DIR"

docker run --rm \
  --user "$(id -u):$(id -g)" \
  -e HOME=/tmp \
  -v "$RESULTS_DIR:/results" \
  llm-rtc-neteq-benchmark:local /results
