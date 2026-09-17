#!/usr/bin/env python3
"""Build a VLN catalogue from a source configuration, without Habitat or GPUs."""
import argparse
from pathlib import Path

try:
    from .vln_catalog import build_database, json_text
except ImportError:
    from vln_catalog import build_database, json_text


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = build_database(args.config, args.output)
    print(json_text({"output": str(args.output.resolve()), "counts": result["counts"]}))


if __name__ == "__main__":
    main()
