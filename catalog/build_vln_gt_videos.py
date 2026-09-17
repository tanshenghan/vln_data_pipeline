"""Bind R2R train LeRobot observations to exact VLN catalog records."""
import argparse
import json
from pathlib import Path

from vln_gt_video import build_index


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("catalog", "output", "video-root", "renderer", "python"):
        parser.add_argument("--" + name, type=Path, required=True)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--dataset", type=Path)
    inputs.add_argument("--sources", type=Path, help="JSON list of R2R/RxR dataset roots or validated shard roots")
    parser.add_argument("--source-id", default="r2r-vlnce-v1-3")
    args = parser.parse_args()
    if args.sources:
        from vln_video_collection import build_collection
        result = build_collection(args.catalog, json.loads(args.sources.read_text()), args.output, args.video_root, args.renderer, args.python)
    else:
        result = build_index(args.catalog, args.dataset, args.output, args.video_root, args.renderer, args.python, args.source_id)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
