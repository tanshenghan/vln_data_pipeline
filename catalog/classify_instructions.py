#!/usr/bin/env python3
"""CLI for fixed-taxonomy instruction classification."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

try:
    from .instruction_classification import DEFAULT_REVIEW_THRESHOLD, classify_inventory
except ImportError:  # Direct execution from a repository checkout.
    from instruction_classification import DEFAULT_REVIEW_THRESHOLD, classify_inventory


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary-output", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--api-base", default="https://ai.skywalk.cloud")
    parser.add_argument("--api-key-env", default="SKYWALK_API_KEY")
    parser.add_argument("--model", default="gpt-5.4-mini")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--review-threshold", type=float, default=DEFAULT_REVIEW_THRESHOLD)
    parser.add_argument("--limit-units", type=int, help="Classify only the first N pending API units for calibration")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    api_key = os.environ.get(args.api_key_env, "")
    if not api_key:
        raise SystemExit(f"missing API key environment variable: {args.api_key_env}")
    summary = classify_inventory(
        inventory_path=args.inventory,
        output_path=args.output,
        summary_path=args.summary_output,
        checkpoint_path=args.checkpoint,
        api_key=api_key,
        api_base=args.api_base,
        model=args.model,
        batch_size=args.batch_size,
        workers=args.workers,
        review_threshold=args.review_threshold,
        limit_units=args.limit_units,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

