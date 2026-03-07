#!/usr/bin/env python3
"""Generate benchmark comparison charts from local + Modal artifacts."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

try:
    from benchmarks.visualization import (
        DEFAULT_BINS,
        DEFAULT_LOCAL_PER_SEED_DIR,
        DEFAULT_MODAL_RUN_PATH,
        DEFAULT_MODAL_SUMMARY_PATH,
        DEFAULT_OUTPUT_DIR,
        generate_visualizations,
    )
except ModuleNotFoundError:
    REPO_ROOT = Path(__file__).resolve().parents[1]
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    from benchmarks.visualization import (
        DEFAULT_BINS,
        DEFAULT_LOCAL_PER_SEED_DIR,
        DEFAULT_MODAL_RUN_PATH,
        DEFAULT_MODAL_SUMMARY_PATH,
        DEFAULT_OUTPUT_DIR,
        generate_visualizations,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot local-vs-Modal benchmark comparisons.")
    parser.add_argument(
        "--local-per-seed-dir",
        default=str(DEFAULT_LOCAL_PER_SEED_DIR),
        help="Directory containing local per-seed JSON metrics.",
    )
    parser.add_argument(
        "--modal-run-json",
        default=str(DEFAULT_MODAL_RUN_PATH),
        help="Path to Modal GA run metrics JSON (with generations).",
    )
    parser.add_argument(
        "--modal-summary-json",
        default=str(DEFAULT_MODAL_SUMMARY_PATH),
        help="Path to Modal GA summary JSON.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Output directory for charts and markdown summary.",
    )
    parser.add_argument(
        "--bins",
        type=int,
        default=DEFAULT_BINS,
        help="Number of generation-equivalent progress bins.",
    )
    args = parser.parse_args()

    outputs = generate_visualizations(
        local_per_seed_dir=Path(args.local_per_seed_dir),
        modal_run_path=Path(args.modal_run_json),
        modal_summary_path=Path(args.modal_summary_json),
        output_dir=Path(args.output_dir),
        bins=args.bins,
    )

    print("Visualization artifacts generated:")
    print(f"  Fitness chart: {outputs['fitness_chart']}")
    print(f"  Runtime chart: {outputs['runtime_chart']}")
    print(f"  Radar chart: {outputs['radar_chart']}")
    print(f"  Summary: {outputs['summary']}")


if __name__ == "__main__":
    main()
