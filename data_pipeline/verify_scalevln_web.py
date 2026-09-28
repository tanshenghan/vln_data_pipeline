"""Opt-in browser acceptance: actually starts one local ScaleVLN GT run."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import time

from playwright.sync_api import sync_playwright


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", required=True, help="Your own deployment, e.g. http://127.0.0.1:8801")
    parser.add_argument("--chromium", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="New evidence directory")
    parser.add_argument("--episode-id", help="Omit to test default ready-scene selection")
    parser.add_argument("--start-collection", action="store_true", required=True,
                        help="Explicitly authorize one real collection; this is not a read-only smoke test")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    evidence = {"started_at": datetime.now(timezone.utc).isoformat(), "url": args.url,
                "requested_episode_id": args.episode_id, "complete": False}
    errors = []
    try:
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch(executable_path=str(args.chromium), headless=True,
                                                 args=["--no-sandbox"])
            page = browser.new_page(viewport={"width": 1568, "height": 1150})
            page.on("pageerror", lambda error: errors.append(str(error)))
            page.goto(args.url.rstrip("/") + "/#gt", wait_until="networkidle")
            form = page.locator("#gt-form")
            form.locator('[name="source_dataset"]').select_option("scalevln")
            form.locator('[name="name"]').fill("ScaleVLN 网页端到端验收 " + evidence["started_at"])
            form.locator('[name="count"]').fill("1")
            if args.episode_id:
                form.locator('[name="episode_ids"]').fill(args.episode_id)
            assert form.locator('[name="split"]').input_value() == "train"
            assert form.locator('[name="role"]').input_value() == "guide"
            assert form.locator('[name="execution"]').input_value() == "local"
            assert form.locator('[name="full_dataset"]').is_disabled()
            assert form.locator('[name="split"] option[value="test"]').evaluate("o => o.disabled")
            page.wait_for_function("!document.querySelector('#start-gt').disabled")
            evidence["readiness"] = page.locator("#gt-ready").inner_text()
            with page.expect_response(lambda r: r.url.endswith("/api/preflight") and r.request.method == "POST") as event:
                page.locator("#check-capacity").click()
            response = event.value
            assert response.ok, response.text()
            evidence["preflight"] = response.json()
            assert evidence["preflight"]["episodes"] == 1 and evidence["preflight"]["can_start"]
            page.screenshot(path=str(args.output / "form.png"), full_page=True)
            with page.expect_response(lambda r: r.url.endswith("/api/runs") and r.request.method == "POST") as event:
                page.locator("#start-gt").click()
            response = event.value
            assert response.status == 202, response.text()
            run_id = response.json()["id"]
            evidence["run_id"] = run_id
            print(json.dumps({"submitted_run_id": run_id}), flush=True)
            deadline = time.monotonic() + 300
            while time.monotonic() < deadline:
                response = page.request.get(args.url.rstrip("/") + "/api/runs/" + run_id)
                assert response.ok, response.text()
                run = response.json()
                if run["status"] in ("succeeded", "failed", "cancelled", "paused"):
                    break
                page.wait_for_timeout(3000)
            evidence["run"] = run
            assert run["status"] == "succeeded", run.get("error") or run["status"]
            kinds = {artifact["kind"] for artifact in run["artifacts"]}
            assert {"dataset", "video"} <= kinds, kinds
            dataset = next(a for a in run["artifacts"] if a["kind"] == "dataset")
            response = page.request.get(args.url.rstrip("/") + "/api/artifacts/" + dataset["id"]
                                        + "?file=meta/platform_validation.json")
            assert response.ok, response.text()
            validation = response.json()
            for check in ("complete", "self_contained", "official_loader_all_frames",
                          "all_30_source_arrays_exact", "geometry_recomputed"):
                assert validation[check], validation
            evidence["validation"] = validation
            page.goto(args.url.rstrip("/") + "/#run/" + run_id, wait_until="networkidle")
            page.wait_for_function("document.querySelector('#run-content video')?.readyState >= 2")
            page.locator("#run-content video").first.evaluate("v => v.play()")
            page.wait_for_function("document.querySelector('#run-content video')?.currentTime > 0")
            page.locator("#run-content video").first.evaluate("v => v.pause()")
            evidence["video_playback"] = True
            page.screenshot(path=str(args.output / "completed.png"), full_page=True)
            assert not errors, errors
            evidence["complete"] = True
            browser.close()
    finally:
        evidence["browser_errors"] = errors
        evidence["finished_at"] = datetime.now(timezone.utc).isoformat()
        (args.output / "evidence.json").write_text(json.dumps(evidence, ensure_ascii=False, indent=2) + "\n")
        print(json.dumps({k: v for k, v in evidence.items() if k not in ("run", "validation")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
