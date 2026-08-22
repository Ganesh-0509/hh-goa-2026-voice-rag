from playwright.sync_api import sync_playwright

URL = "http://127.0.0.1:8300/"
import os
OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".impeccable", "review")

with sync_playwright() as p:
    browser = p.chromium.launch()
    page = browser.new_page(viewport={"width": 1440, "height": 900})
    page.goto(URL, wait_until="networkidle")
    page.screenshot(path=f"{OUT_DIR}/desktop_v2.png", full_page=True)
    print("desktop_v2.png saved")

    # Force the recording state visually (can't grant real mic access headless)
    page.eval_on_selector("#recordBtn", "el => el.classList.add('recording')")
    page.eval_on_selector("#recordText", "el => el.textContent = 'Stop & Process Question'")
    page.wait_for_timeout(400)
    page.screenshot(path=f"{OUT_DIR}/desktop_recording.png", full_page=True)
    print("desktop_recording.png saved")

    browser.close()
