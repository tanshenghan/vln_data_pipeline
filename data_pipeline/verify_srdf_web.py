"""Read-only SRDF browsing acceptance. Never starts a GT or cloud job."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from urllib.parse import urlsplit

import httpx
from playwright.sync_api import sync_playwright


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8801")
    parser.add_argument("--browser-executable", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    base = args.base_url.rstrip("/")
    report = {"checked_at": datetime.now(timezone.utc).isoformat(), "base_url": base,
              "complete": False, "collection_tested": False, "gt_collection_accepted": False}
    with httpx.Client(base_url=base, timeout=90) as client:
        response = client.get("/api/health")
        response.raise_for_status()
        status = response.json()["srdf"]
        assert status["source_ready"] and not status["can_collect"]
        report["source"] = status
        before = {r["id"] for r in client.get("/api/runs").json()}
        first = client.get("/api/sources", params={"dataset": "SRDF", "limit": 2}).json()
        second = client.get("/api/sources", params={"dataset": "SRDF", "offset": 2, "limit": 2}).json()
        assert first["stats"]["records"] == status["total_episodes"]
        assert len(first["items"]) == len(second["items"]) == 2
        assert not ({r["id"] for r in first["items"]} & {r["id"] for r in second["items"]})
        assert all(not r["can_collect"] and r["language"] == "en" for r in first["items"])
        record = first["items"][0]
        response = client.get("/vln-api/episode", params={"id": record["id"]})
        response.raise_for_status()
        detail = response.json()
        assert detail["instruction_id"] == detail["episode_id"]
        assert detail["reference_path"] and detail["viewpoints"] and not detail["ground_truth"]["available"]
        assert len(detail["reference_path"]) == len(detail["viewpoints"])
        report["checked_record"] = {k: detail[k] for k in ("id", "episode_id", "instruction", "scene", "point_count")}
        with sync_playwright() as p:
            browser = p.chromium.launch(executable_path=str(args.browser_executable), headless=True,
                                        args=["--no-sandbox"])
            page = browser.new_page(viewport={"width": 1568, "height": 1100})
            page.set_default_timeout(90000)
            errors, writes = [], []

            def readonly(route):
                if route.request.method not in ("GET", "HEAD", "OPTIONS"):
                    writes.append({"method": route.request.method, "path": urlsplit(route.request.url).path})
                    route.abort()
                else:
                    route.continue_()

            page.route("**/*", readonly)
            page.on("pageerror", lambda error: errors.append(str(error)))
            page.goto(base + "/?dataset=SRDF&split=train&language=en&role=guide#atlas", wait_until="domcontentloaded")
            page.wait_for_function("document.querySelector('#detail').textContent.includes('SRDF 公开 HM3D round-3')")
            assert page.locator('#filters [name="dataset"]').input_value() == "SRDF"
            assert page.locator("#detail svg.route").is_visible()
            assert "不是连续动作或逐帧 GT" in page.locator("#detail").inner_text()
            assert page.locator('#detail .source-pipeline-actions button').count() == 2
            assert all(button.is_disabled() for button in page.locator('#detail .source-pipeline-actions button').all())
            page.screenshot(path=str(args.output / "atlas.png"))
            page.goto(base + "/#gt", wait_until="domcontentloaded")
            form = page.locator("#gt-form")
            form.locator('[name="source_dataset"]').select_option("srdf")
            page.wait_for_function("document.querySelector('#gt-ready').textContent.includes('SRDF-CE')")
            assert page.locator("#start-gt").is_disabled()
            assert page.locator("#check-capacity").is_disabled()
            assert form.locator('[name="split"]').input_value() == "train"
            assert form.locator('[name="full_dataset"]').is_disabled()
            assert form.locator('[name="collection_mode"]').input_value() == "custom"
            assert form.locator('[name="collection_mode"] option[value="continuous"]').evaluate("o=>o.disabled")
            report["readiness_message"] = page.locator("#gt-ready").inner_text()
            page.screenshot(path=str(args.output / "gt-blocked.png"))
            form.locator('[name="source_dataset"]').select_option("r2r")
            assert not page.locator("#check-capacity").is_disabled()
            assert not form.locator('[name="collection_mode"] option[value="continuous"]').evaluate("o=>o.disabled")
            assert not writes, writes
            assert not errors, errors
            browser.close()
        after = {r["id"] for r in client.get("/api/runs").json()}
        report.update(complete=True, blocked_writes=writes, browser_errors=errors,
                      run_ids_unchanged=before == after)
    (args.output / "evidence.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
