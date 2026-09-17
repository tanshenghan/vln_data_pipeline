#!/usr/bin/env python3
"""Generate the dataset semantic explorer catalog."""

from __future__ import annotations

import argparse
from pathlib import Path

from catalog import scan_catalog, write_catalog
from classification_overlay import enrich_instruction_classification
from conversion_status import enrich_conversion_status
from embedding import DEFAULT_MODEL_ID, DEFAULT_MODEL_REVISION, build_embedding


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True, help="Downloaded dataset root to scan read-only.")
    parser.add_argument("--output", type=Path, required=True, help="Destination catalog.json path.")
    parser.add_argument(
        "--instruction-classification",
        type=Path,
        help="Optional current-taxonomy classification JSONL or checkpoint to overlay onto catalog tasks.",
    )
    parser.add_argument(
        "--instruction-inventory",
        type=Path,
        help="Optional full instruction JSONL used to add compact task rows for metadata-only datasets.",
    )
    parser.add_argument(
        "--conversion-pfs-root",
        type=Path,
        help="LeRobot output root on PFS. Omit it to report affected records as status unknown.",
    )
    parser.add_argument(
        "--conversion-cfs-root",
        type=Path,
        help="LeRobot output root on CFS. Omit it to report affected records as status unknown.",
    )
    parser.add_argument(
        "--skip-embedding",
        action="store_true",
        help="Omit the optional two-dimensional task-term embedding.",
    )
    parser.add_argument("--embedding-model", default=DEFAULT_MODEL_ID, help="Hugging Face model id or local model path.")
    parser.add_argument("--embedding-revision", default=DEFAULT_MODEL_REVISION, help="Model branch, tag, or commit.")
    parser.add_argument("--embedding-cache-dir", type=Path, help="Persistent Hugging Face model cache directory.")
    parser.add_argument(
        "--embedding-local-files-only",
        action="store_true",
        help="Never access Hugging Face while loading the embedding model.",
    )
    parser.add_argument("--embedding-device", default="auto", help="Torch device: auto, cpu, cuda, or cuda:N.")
    parser.add_argument("--embedding-batch-size", type=int, default=64)
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    catalog = scan_catalog(args.root, instruction_inventory=args.instruction_inventory)
    enrich_instruction_classification(catalog, args.instruction_classification)
    enrich_conversion_status(
        catalog,
        pfs_root=args.conversion_pfs_root,
        cfs_root=args.conversion_cfs_root,
    )
    if not args.skip_embedding:
        catalog["embedding"] = build_embedding(
            catalog["records"],
            generated_at=catalog["generated_at"],
            model_name_or_path=args.embedding_model,
            revision=args.embedding_revision,
            cache_dir=args.embedding_cache_dir,
            local_files_only=args.embedding_local_files_only,
            device=args.embedding_device,
            batch_size=args.embedding_batch_size,
        )
    write_catalog(catalog, args.output)
    stats = catalog["stats"]
    embedding_text = f", {catalog['embedding']['point_count']} embedding points" if "embedding" in catalog else ""
    conversion = stats["conversion_by_status"]
    print(
        f"wrote {args.output}: {stats['records']} path records, "
        f"{stats['unique_tasks']} dataset-task pairs, {stats['datasets']} datasets, "
        f"{conversion['converted']} converted, {conversion['partial']} partial, "
        f"{conversion['not_converted']} not converted, {conversion['excluded']} excluded, "
        f"{conversion['unknown']} unknown{embedding_text}"
    )


if __name__ == "__main__":
    main()
