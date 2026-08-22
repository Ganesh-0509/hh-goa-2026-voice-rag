import sys
from playwright.sync_api import sync_playwright

URL = "http://127.0.0.1:8300/"
OUT_DIR = sys.argv[1] if len(sys.argv) > 1 else "."

with sync_playwright() as p:
    browser = p.chromium.launch()
    page = browser.new_page(viewport={"width": 1440, "height": 900})
    page.goto(URL, wait_until="networkidle")

    # Grounded-answer state (real query pulled from the indexed corpus)
    page.fill("#textInput", "कॉर्पोरेशन क्या है?")
    page.click("#submitTextBtn")
    page.wait_for_selector("#statusBadge.status-success, #statusBadge.status-abstained", timeout=15000)
    page.screenshot(path=f"{OUT_DIR}/desktop_grounded.png", full_page=True)
    print("desktop_grounded.png saved")

    # Abstained state (off-topic query)
    page.fill("#textInput", "What is the capital of Goa?")
    page.click("#submitTextBtn")
    page.wait_for_timeout(500)
    page.wait_for_selector("#statusBadge.status-abstained", timeout=15000)
    page.screenshot(path=f"{OUT_DIR}/desktop_abstained.png", full_page=True)
    print("desktop_abstained.png saved")

    browser.close()
