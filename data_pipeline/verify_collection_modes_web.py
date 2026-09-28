"""Read-only browser acceptance of custom/continuous collection requests."""
import argparse
import json
from pathlib import Path

from playwright.sync_api import sync_playwright


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", required=True)
    parser.add_argument("--chromium", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    errors, requests, checks = [], [], []
    with sync_playwright() as p:
        browser = p.chromium.launch(executable_path=str(args.chromium), headless=True,
                                   args=["--no-sandbox"])
        page = browser.new_page(viewport={"width": 1500, "height": 1100})
        # Test-only read access to module-scoped functions; no production globals
        # or intercepted writes, and the application logic remains unchanged.
        def inspect_module(route):
            response = route.fetch()
            route.fulfill(response=response, body=response.text()
                + "\nwindow.__collectionInspection={state,collectionRequest,refresh};\n")
        page.route(args.url.rstrip("/") + "/app.js", inspect_module)
        page.on("pageerror", lambda error: errors.append(str(error)))
        page.on("request", lambda request: requests.append(request.method)
                if request.url.startswith(args.url) else None)
        page.goto(args.url.rstrip("/") + "/#gt", wait_until="networkidle")
        page.wait_for_function("!!window.__collectionInspection?.state.health && !!window.__collectionInspection?.state.compute")
        form = page.locator("#gt-form")
        mode = form.locator('[name="collection_mode"]')
        def select_devices(devices):
            # Check desired devices first so the last-unchecked fallback never
            # inserts GPU0 when selecting a single cloud device.
            for device in devices:
                page.locator(f'#compute-options input[value="{device}"]').check()
            for checkbox in page.locator("#compute-options input").all():
                if checkbox.get_attribute("value") not in devices:
                    checkbox.uncheck()

        def selected_devices():
            return page.locator("#compute-options input:checked").evaluate_all("els=>els.map(e=>e.value)")

        assert mode.locator("option").all_text_contents() == ["自定义", "持续采集"]
        for dataset in ("r2r", "rxr", "scalevln"):
            form.locator('[name="source_dataset"]').select_option(dataset)
            devices = ("local",) if dataset == "scalevln" else ("local", "baidu-pro6000", "baidu-b200")
            for device in devices:
                # Selecting one device must not implicitly choose continuous mode.
                select_devices([device])
                for value in ("custom", "continuous"):
                    mode.select_option(value)
                    assert page.locator("#gt-count-field").is_visible()
                    assert page.locator("#gt-custom-count").is_visible() == (value == "custom")
                    for count in ((1, 8, 32) if value == "custom" else (1,)):
                        if value == "custom":
                            form.locator('[name="count"]').fill(str(count))
                        request = page.evaluate("window.__collectionInspection.collectionRequest(document.querySelector('#gt-form'),'gt')")
                        assert request["continuous"] == (value == "continuous")
                        assert request["path"] == ("/collection-plans" if value == "continuous" else "/runs")
                        assert request["payload"]["execution"] == ("local" if device == "local" else "cloud_idle")
                        if value == "custom":
                            assert request["payload"]["count"] == count
                            assert request["payload"]["compute_pool"] == ("auto" if device == "local" else device)
                        else:
                            assert request["payload"]["compute_pools"] == [device]
                        checks.append({"dataset": dataset, "device": device, "mode": value,
                            "count": count if value == "custom" else None,
                            "submit_enabled": not page.locator("#start-gt").is_disabled()})
        assert page.locator('#compute-options input[value="baidu-pro6000"]').is_disabled()
        assert page.locator('#compute-options input[value="baidu-b200"]').is_disabled()
        assert "云端 HM3D 与源数据尚未部署验收" in page.locator("#gt-ready").inner_text()
        for dataset in ("r2r", "rxr"):
            form.locator('[name="source_dataset"]').select_option(dataset)
            mode.select_option("continuous")
            for devices in (["local", "baidu-pro6000"], ["local", "baidu-b200"],
                            ["local", "baidu-pro6000", "baidu-b200"]):
                select_devices(devices)
                assert selected_devices() == devices
                assert mode.input_value() == "continuous"
                # Exercise the real periodic refresh without sleeping a poll interval.
                page.evaluate("window.__collectionInspection.refresh()")
                assert selected_devices() == devices
                request = page.evaluate("window.__collectionInspection.collectionRequest(document.querySelector('#gt-form'),'gt')")
                assert request["path"] == "/collection-plans"
                assert request["payload"]["execution"] == "mixed"
                assert request["payload"]["compute_pools"] == devices
                assert request["payload"]["compute_pool"] == "auto"
                pools = page.evaluate("window.__collectionInspection.state.compute.pools")
                enabled = all(any(p["id"] == d and p["enabled"] for p in pools) for d in devices if d != "local")
                assert page.locator("#start-gt").is_enabled() == enabled
                assert "GPU0：尚未接入" not in page.locator("#gt-ready").inner_text()
                checks.append({"dataset": dataset, "devices": devices, "mode": "continuous",
                    "execution": "mixed", "selection_survives_refresh": True, "submit_enabled": enabled})
            # A mixed GT selection must not leak into the independent Seedance form.
            request = page.evaluate("window.__collectionInspection.collectionRequest(document.querySelector('#seedance-form'),'seedance')")
            assert request["path"] == "/runs" and request["payload"]["execution"] == "local"
            # Never silently promote bounded custom work into continuous work.
            mode.select_option("custom")
            assert selected_devices() == ["local", "baidu-pro6000", "baidu-b200"]
            assert page.locator("#start-gt").is_disabled()
            assert "自定义数量请只选择一台设备" in page.locator("#gt-ready").inner_text()
            mode.select_option("continuous")
        page.screenshot(path=str(args.output / "continuous.png"), full_page=True)
        form.screenshot(path=str(args.output / "mixed-form.png"))
        mode.select_option("custom")
        select_devices(["local"])
        form.locator('[name="count"]').fill("8")
        page.screenshot(path=str(args.output / "custom.png"), full_page=True)
        page.set_viewport_size({"width": 390, "height": 844})
        page.screenshot(path=str(args.output / "mobile.png"), full_page=True)
        plans = page.request.get(args.url.rstrip("/") + "/api/collection-plans").json()
        evidence = {"complete": True, "checks": checks, "page_errors": errors,
            "post_requests": requests.count("POST"), "plans": [
                {key: plan.get(key) for key in ("id", "name", "status", "compute_pools")}
                for plan in plans]}
        assert not errors, errors
        assert not evidence["post_requests"], "Read-only UI acceptance must not create runs"
        (args.output / "evidence.json").write_text(json.dumps(evidence, ensure_ascii=False, indent=2) + "\n")
        browser.close()
    print(json.dumps({"complete": True, "checks": len(checks), "post_requests": 0,
                      "output": str(args.output)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
