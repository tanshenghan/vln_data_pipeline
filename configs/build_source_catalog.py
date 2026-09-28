"""Index configured R2R/RxR and optional ScaleVLN/SRDF into personal runtime."""
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from apps.api.settings import settings
from catalog.vln_catalog import build_database
from workers.command_runner import atomic_json


def source_configuration(config=settings):
    sources = json.loads((ROOT / "catalog/vln_sources.example.json").read_text())
    for source in sources["sources"]:
        source["root"] = str(config.dataset_root(source["dataset"].lower()).resolve())
    scale_root = config.dataset_root("scalevln")
    if (scale_root / "source_manifest.json").is_file():
        from data_pipeline.scalevln_source import prepared_manifest, SOURCE_ID, REVISION
        prepared_manifest(scale_root)
        sources["sources"].append({"id": SOURCE_ID, "dataset": "ScaleVLN", "version": "StreamVLN-CE " + REVISION,
            "format": "vlnce", "root": str(scale_root.resolve()), "pattern": "{split}/{split}.json.gz",
            "splits": ["train"], "roles": ["guide"]})
    srdf_root = config.dataset_root("srdf")
    if (srdf_root / "source_manifest.json").is_file():
        from data_pipeline import srdf_source
        srdf_source.prepared_manifest(srdf_root, verify_hashes=True)
        sources["sources"].append({"id": srdf_source.SOURCE_ID, "dataset": "SRDF",
            "version": "HM3D round-3 public (discrete) " + srdf_source.REVISION,
            "format": "r2r", "root": str(srdf_root.resolve()), "pattern": "{split}/{split}.jsonl",
            "connectivity": str((srdf_root / "connectivity").resolve()),
            "splits": ["train"], "roles": ["guide"]})
    return sources


def main():
    settings.runtime_root.mkdir(parents=True, exist_ok=True)
    config_path = settings.runtime_root / "sources.local.json"
    atomic_json(config_path, source_configuration())
    result = build_database(config_path, settings.catalog_db)
    print(json.dumps({"output": str(settings.catalog_db), "counts": result["counts"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
