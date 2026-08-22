import sys
from playwright.sync_api import sync_playwright

URL = "http://127.0.0.1:8300/"
OUT_DIR = sys.argv[1] if len(sys.argv) > 1 else "."

with sync_playwright() as p:
    browser = p.chromium.launch()

    desktop = browser.new_page(viewport={"width": 1440, "height": 900})
    desktop.goto(URL, wait_until="networkidle")
    desktop.screenshot(path=f"{OUT_DIR}/desktop.png", full_page=True)
    print("desktop.png saved")

    mobile = browser.new_page(viewport={"width": 390, "height": 844})
    mobile.goto(URL, wait_until="networkidle")
    mobile.screenshot(path=f"{OUT_DIR}/mobile.png", full_page=True)
    print("mobile.png saved")

    browser.close()
