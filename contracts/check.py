"""BetaNav metadata compatibility plus the authoritative independent GT validator."""
import argparse
import json
from pathlib import Path

CONTRACT_PATH = Path(__file__).with_name("beta_nav_v1.json")


def check_metadata(dataset):
    root = Path(dataset)
    contract = json.loads(CONTRACT_PATH.read_text())
    metadata = json.loads((root / "meta/beta_nav.json").read_text())
    for key, expected in contract["metadata"].items():
        if key not in metadata or metadata[key] != expected:
            raise ValueError(f"BetaNav contract mismatch: {key}")
    for key, feature in contract["features"].items():
        actual = metadata["features"].get(key, {})
        if any(actual.get(field) != feature[field] for field in ("dtype", "shape")):
            raise ValueError(f"BetaNav contract feature mismatch: {key}")
    if not metadata.get("episodes"):
        raise ValueError("BetaNav contract requires episodes")
    for episode in metadata["episodes"]:
        for section in ("pointing_config", "camera"):
            for key, expected in contract[section].items():
                if episode.get(section, {}).get(key) != expected:
                    raise ValueError(f"BetaNav current contract mismatch: {section}.{key}")
    return {"contract": contract["id"], "upstream_commit": contract["upstream_commit"], "metadata_compatible": True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--report", type=Path, required=True, help="New independent validation report")
    args = parser.parse_args()
    if args.report.exists():
        parser.error("report already exists")
    result = check_metadata(args.dataset)
    # Reuse exactly the canonical validator, including official loader, all arrays,
    # masks, recomputed geometry and bundled RVQ hashes. No source capture is passed.
    from data_pipeline.validate_lerobot import validate_lerobot
    result.update(validate_lerobot(args.dataset))
    args.report.parent.mkdir(parents=True, exist_ok=True)
    with args.report.open("x") as handle:
        json.dump(result, handle, indent=2)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
