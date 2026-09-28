"""Read-only browser acceptance for all three native ObjectNav catalogs."""
import argparse
import json
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

import httpx
from playwright.sync_api import sync_playwright


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8801")
    parser.add_argument("--browser-executable", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    base = args.base_url.rstrip("/")
    with httpx.Client(base_url=base, timeout=60) as client:
        response = client.get("/api/objectnav/datasets")
        response.raise_for_status()
        sources = response.json()["items"]
        assert len(sources) == 3 and all(s["available"] for s in sources)
        before_runs = {r["id"] for r in client.get("/api/runs").json()}
        report = {"base_url": base, "datasets": [], "browser_checked": [],
                  "collection_tested": False, "gt_collection_accepted": False}
        for source in sources:
            split_reports = {}
            for split, content in source["splits"].items():
                scene = content["scenes"][0]["scene"]
                params = {"dataset": source["id"], "split": split, "scene": scene, "limit": 2}
                response = client.get("/api/objectnav/episodes", params=params)
                response.raise_for_status()
                page = response.json()
                assert page["items"] and not page["can_collect"]
                row = page["items"][0]
                detail = client.get("/api/objectnav/episode", params={**params, "source_row": row["source_row"]})
                detail.raise_for_status()
                detail = detail.json()
                assert detail["id"] == row["id"] and detail["goal_objects"]
                assert not detail["original_language_instructions"] and not detail["can_collect"]
                split_reports[split] = {"episodes": content["episodes"], "scenes": len(content["scenes"]),
                    "checked_scene": scene, "source_episode_id": row["source_episode_id"],
                    "source_scene_id": detail["source_episode"]["scene_id"], "semantic_version": detail["semantic_version"]}
            report["datasets"].append({"id": source["id"], "splits": split_reports})
        with sync_playwright() as p:
            kwargs = {"headless": True}
            if args.browser_executable:
                kwargs["executable_path"] = str(args.browser_executable)
            browser = p.chromium.launch(**kwargs)
            page = browser.new_page(viewport={"width": 1440, "height": 1050})
            page.set_default_timeout(60000)
            # This acceptance never submits jobs or generates remote videos.
            blocked_writes, errors = [], []
            def readonly(route):
                if route.request.method not in ("GET", "HEAD", "OPTIONS"):
                    blocked_writes.append({"method": route.request.method, "path": urlsplit(route.request.url).path})
                    route.abort()
                else:
                    route.continue_()
            page.route("**/*", readonly)
            page.on("pageerror", lambda error: errors.append(str(error)))
            page.goto(base + "/#objectnav", wait_until="domcontentloaded")
            page.wait_for_selector("#objectnav-table tbody tr")
            assert page.locator("#objectnav").is_visible()
            assert page.locator("#objectnav-dataset option").count() == 3
            for source in sources:
                def episodes_response(response):
                    parsed = urlsplit(response.url)
                    return parsed.path == "/api/objectnav/episodes" and parse_qs(parsed.query).get("dataset") == [source["id"]]
                with page.expect_response(episodes_response):
                    page.locator("#objectnav-dataset").select_option(source["id"])
                page.wait_for_selector("#objectnav-table tbody tr")
                assert page.locator("#objectnav-status").inner_text().startswith(source["label"])
                with page.expect_response(lambda r: urlsplit(r.url).path == "/api/objectnav/episode"):
                    page.locator("[data-objectnav-row]").first.click()
                page.wait_for_selector("#objectnav-detail:not([hidden])")
                assert source["semantics"] in page.locator("#objectnav-detail").inner_text()
                category = page.locator("#objectnav-category option").nth(1).get_attribute("value")
                with page.expect_response(episodes_response):
                    page.locator("#objectnav-category").select_option(category)
                page.wait_for_selector("#objectnav-table tbody tr")
                assert all(c == category for c in page.locator("#objectnav-table tbody tr td:nth-child(2)").all_text_contents())
                if not page.locator("#objectnav-next").is_disabled():
                    with page.expect_response(episodes_response):
                        page.locator("#objectnav-next").click()
                    page.wait_for_function("document.getElementById('objectnav-page').textContent.startsWith('31–')")
                page.evaluate("window.scrollTo(0,0)")
                page.screenshot(path=str(args.output / (source["id"] + ".png")))
                report["browser_checked"].append(source["id"])
            assert not blocked_writes, blocked_writes
            assert not errors, errors
            browser.close()
        after_runs = {r["id"] for r in client.get("/api/runs").json()}
        report.update(complete=True, blocked_writes=blocked_writes, browser_errors=errors,
                      run_ids_unchanged=before_runs == after_runs)
        (args.output / "evidence.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
        print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
