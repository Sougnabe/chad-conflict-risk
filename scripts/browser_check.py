"""
Cross-browser and device compatibility check of the running dashboard (Playwright).

For each browser engine (Chromium, Firefox, WebKit) and screen (desktop, mobile), opens the
dashboard, measures the time until the content and the map are displayed, records JavaScript
console errors and saves a screenshot. Chromium is also run as a low-end device (4x slower CPU,
"Fast 3G" network). Results: reports/performance/browsers.json, browsers.md, screenshots/.

Usage (dashboard running on port 8599):
    streamlit run app/app.py --server.port 8599 --server.headless true
    python scripts/browser_check.py
"""

import json
import time

import _bootstrap  # noqa: F401
from playwright.sync_api import sync_playwright

from chadrisk.config import PATHS

URL = "http://localhost:8599"
OUT = PATHS.reports / "performance"
SHOTS = OUT / "screenshots"
TIMEOUT_MS = 120_000
FAST_3G = {"offline": False, "latency": 150, "downloadThroughput": 1.6e6 / 8,
           "uploadThroughput": 750e3 / 8}

SCENARIOS = [
    # (name, engine, device or viewport, low_end)
    ("Chromium desktop", "chromium", {"viewport": {"width": 1366, "height": 768}}, False),
    ("Firefox desktop", "firefox", {"viewport": {"width": 1366, "height": 768}}, False),
    ("WebKit desktop (Safari engine)", "webkit", {"viewport": {"width": 1366, "height": 768}},
     False),
    ("Chromium mobile (Pixel 5)", "chromium", "Pixel 5", False),
    ("WebKit mobile (iPhone 13)", "webkit", "iPhone 13", False),
    ("Firefox small screen (390 px)", "firefox", {"viewport": {"width": 390, "height": 844}},
     False),
    ("Chromium low-end (CPU x4, Fast 3G)", "chromium", {"viewport": {"width": 1366,
                                                                     "height": 768}}, True),
]


def map_drawn(page) -> bool:
    """True when a Leaflet map with department polygons is displayed in any frame."""
    for frame in page.frames:
        try:
            if frame.locator("path.leaflet-interactive").count() > 0:
                return True
        except Exception:  # frame detached while the app re-renders
            continue
    return False


def run(playwright, name, engine, device, low_end):
    browser = getattr(playwright, engine).launch()
    options = playwright.devices[device] if isinstance(device, str) else device
    context = browser.new_context(**options)
    page = context.new_page()
    errors = []
    page.on("console", lambda m: errors.append(m.text) if m.type == "error" else None)
    heap_mb = None
    if low_end:
        cdp = context.new_cdp_session(page)
        cdp.send("Emulation.setCPUThrottlingRate", {"rate": 4})
        cdp.send("Network.enable")
        cdp.send("Network.emulateNetworkConditions", FAST_3G)

    start = time.perf_counter()
    page.goto(URL, timeout=TIMEOUT_MS)
    page.get_by_text("Conflict risk outlook").first.wait_for(timeout=TIMEOUT_MS)
    content_s = time.perf_counter() - start
    deadline = time.perf_counter() + TIMEOUT_MS / 1000
    while not map_drawn(page) and time.perf_counter() < deadline:
        page.wait_for_timeout(250)
    map_s = time.perf_counter() - start if map_drawn(page) else None

    # Interaction: open the Findings page and switch to French
    page.get_by_text("Findings").first.click()
    page.get_by_text("Research question").first.wait_for(timeout=TIMEOUT_MS)
    page.get_by_text("Français").first.click()
    page.get_by_text("Question de recherche").first.wait_for(timeout=TIMEOUT_MS)
    interaction_ok = True

    if engine == "chromium":
        metrics = context.new_cdp_session(page)
        metrics.send("Performance.enable")
        values = {m["name"]: m["value"] for m in
                  metrics.send("Performance.getMetrics")["metrics"]}
        heap_mb = round(values.get("JSHeapUsedSize", 0) / 1e6, 1)

    shot = SHOTS / f"{name.split(' (')[0].lower().replace(' ', '_')}.png"
    page.get_by_text("Perspectives actuelles").first.click()
    page.wait_for_timeout(3000)
    page.screenshot(path=str(shot), full_page=False)
    browser.close()
    return {"scenario": name, "engine": engine, "content_s": round(content_s, 1),
            "map_s": round(map_s, 1) if map_s else None, "interactions": interaction_ok,
            "console_errors": len(errors), "js_heap_mb": heap_mb,
            "screenshot": shot.relative_to(PATHS.root).as_posix()}


SHOTS.mkdir(parents=True, exist_ok=True)
results = []
with sync_playwright() as p:
    for scenario in SCENARIOS:
        try:
            results.append(run(p, *scenario))
        except Exception as error:  # record the failure and continue with the others
            results.append({"scenario": scenario[0], "engine": scenario[1],
                            "error": str(error)[:200]})
        print(results[-1])

(OUT / "browsers.json").write_text(json.dumps(results, indent=2), "utf8")
lines = ["| Scenario | Content shown (s) | Map drawn (s) | Findings + French | Console errors "
         "| JS heap (MB) |", "|---|---|---|---|---|---|"]
for r in results:
    if "error" in r:
        lines.append(f"| {r['scenario']} | failed: {r['error']} | | | | |")
    else:
        lines.append(f"| {r['scenario']} | {r['content_s']} | {r['map_s']} | "
                     f"{'ok' if r['interactions'] else 'failed'} | {r['console_errors']} | "
                     f"{r['js_heap_mb'] or '-'} |")
(OUT / "browsers.md").write_text("\n".join(lines) + "\n", "utf8")
