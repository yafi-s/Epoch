#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

RESULTS_DIR="${RESULTS_DIR:-results/modal_benchmark_t4_w10}"

poetry run python scripts/rank_modal_runs.py \
  --results-dir "$RESULTS_DIR" \
  --write-markdown
