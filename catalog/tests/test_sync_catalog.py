from __future__ import annotations

import copy
import unittest

from catalog.sync_catalog import (
    PublicationRejected,
    catalog_diff,
    catalog_digest,
    validate_candidate,
)


def sample_catalog(*, status: str = "converted", generated_at: str = "2026-09-02T00:00:00+08:00") -> dict:
    return {
        "schema_version": 1,
        "generated_at": generated_at,
        "dataset_root": "/mnt/cfs/data/datasets",
        "stats": {
            "records": 1,
            "datasets": 1,
            "conversion_by_status": {
                "converted": int(status == "converted"),
                "partial": int(status == "partial"),
                "not_converted": int(status == "not_converted"),
                "excluded": int(status == "excluded"),
                "unknown": int(status == "unknown"),
            },
        },
        "conversion": {"checked_at": generated_at},
        "records": [
            {
                "id": "record-a",
                "dataset": "Dataset-A",
                "native_task": "pick_item",
                "source_path": "Dataset-A/pick_item",
                "conversion_status": status,
                "conversion_checked_at": generated_at,
            }
        ],
        "embedding": {"point_count": 1, "generated_at": "2026-09-01T00:00:00+08:00"},
    }


class DatasetSemanticCatalogSyncTest(unittest.TestCase):
    def test_digest_ignores_scan_timestamps_and_embedding(self) -> None:
        first = sample_catalog()
        second = sample_catalog(generated_at="2026-09-02T01:00:00+08:00")
        second["embedding"] = {"point_count": 99}
        self.assertEqual(catalog_digest(first), catalog_digest(second))
        self.assertFalse(catalog_diff(first, second)["changed"])

    def test_diff_detects_additions_and_conversion_changes(self) -> None:
        previous = sample_catalog(status="partial")
        candidate = sample_catalog(status="converted")
        added = copy.deepcopy(candidate["records"][0])
        added["id"] = "record-b"
        added["native_task"] = "place_item"
        candidate["records"].append(added)
        candidate["stats"]["records"] = 2
        diff = catalog_diff(previous, candidate)
        self.assertTrue(diff["changed"])
        self.assertEqual(diff["added_ids"], ["record-b"])
        self.assertEqual(diff["removed_ids"], [])
        self.assertEqual(diff["conversion_changes"][0]["from"], "partial")
        self.assertEqual(diff["conversion_changes"][0]["to"], "converted")

    def test_guards_reject_removal_and_conversion_regression(self) -> None:
        previous = sample_catalog(status="converted")
        removed = sample_catalog(status="converted")
        removed["records"] = []
        removed["stats"]["records"] = 0
        with self.assertRaisesRegex(PublicationRejected, "removes 1 records"):
            validate_candidate(
                previous,
                removed,
                catalog_diff(previous, removed),
                minimum_records=0,
                allow_removals=False,
                max_unknown_increase=0,
            )

        regressed = sample_catalog(status="unknown")
        with self.assertRaisesRegex(PublicationRejected, "protected conversion regressions"):
            validate_candidate(
                previous,
                regressed,
                catalog_diff(previous, regressed),
                minimum_records=1,
                allow_removals=False,
                max_unknown_increase=1,
            )


if __name__ == "__main__":
    unittest.main()
