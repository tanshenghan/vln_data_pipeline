#!/usr/bin/env python3
"""Build a traceable JSONL inventory of dataset instructions from metadata."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from instruction_inventory import build_instruction_inventory


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True, help="Downloaded dataset root to scan read-only.")
    parser.add_argument("--output", type=Path, required=True, help="Destination instruction_inventory.jsonl.")
    parser.add_argument("--summary-output", type=Path, help="Destination summary JSON; defaults beside the JSONL.")
    parser.add_argument("--catalog", type=Path, help="Existing catalog.json to include as the classified baseline.")
    parser.add_argument(
        "--conversion-root",
        type=Path,
        action="append",
        default=[],
        help="LeRobot root containing meta/episodes.jsonl; repeat for PFS and CFS roots.",
    )
    parser.add_argument(
        "--metadata-workers",
        type=int,
        default=16,
        help="Bounded worker count for many small metadata files (default: 16).",
    )
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    summary_output = args.summary_output or args.output.with_suffix(".summary.json")
    summary = build_instruction_inventory(
        dataset_root=args.root,
        output=args.output,
        summary_output=summary_output,
        catalog_path=args.catalog,
        conversion_roots=args.conversion_root,
        metadata_workers=args.metadata_workers,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
