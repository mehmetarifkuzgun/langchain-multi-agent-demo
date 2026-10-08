"""Drive the Streamlit app (offline mode) with Playwright and save README screenshots.

    pip install playwright && playwright install chromium
    python scripts/capture_screenshots.py            # writes docs/img/*.png

Everything shown is produced by the scripted offline models (see offline.py).
"""
import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "docs" / "img"
PORT = 8765
DOC1 = ("Challenges of AI in Education. 1. Privacy and Data Security: student data protection is paramount. "
        "2. Digital Divide: equitable access to AI-powered tools must be ensured. "
        "3. Teacher Training: educators need support to integrate AI effectively.")
DOC2 = ("Benefits of AI in Education. Intelligent tutoring systems adapt lessons to each student and give "
        "real-time feedback. Automated assessment frees teachers to focus on mentoring.")
TOPIC = "The Impact of Artificial Intelligence on Modern Education"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    env = {**os.environ, "MULTI_AGENT_OFFLINE": "1"}
    srv = subprocess.Popen([sys.executable, "-m", "streamlit", "run", "streamlit_ui.py", "--server.port", str(PORT),
                            "--server.headless", "true", "--browser.gatherUsageStats", "false"],
                           cwd=ROOT, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        for _ in range(60):
            try:
                urllib.request.urlopen(f"http://localhost:{PORT}/_stcore/health", timeout=1)
                break
            except Exception:
                time.sleep(1)
        with sync_playwright() as p:
            b = p.chromium.launch()
            page = b.new_page(viewport={"width": 1400, "height": 900})
            page.goto(f"http://localhost:{PORT}")
            page.get_by_role("button", name="Initialize Multi-Agent System").click()
            page.get_by_text("System Online").wait_for()
            for text, src in ((DOC1, "AI_Education_Challenges"), (DOC2, "AI_Education_Benefits")):
                page.get_by_placeholder("Enter your document text here...").fill(text)
                page.keyboard.press("Control+Enter")
                page.get_by_placeholder("e.g., 'Research Paper 2024'").fill(src)
                page.keyboard.press("Enter")
                page.get_by_role("button", name="Add Document").click()
                time.sleep(2.5)  # the app reruns right after adding, so the toast is gone already
            page.get_by_placeholder("e.g., 'The Impact of Artificial Intelligence on Modern Education'").fill(TOPIC)
            page.keyboard.press("Enter")
            page.get_by_role("button", name="Run Complete Workflow").click()
            page.get_by_text("Workflow completed").wait_for(timeout=60000)
            time.sleep(1)
            page.screenshot(path=str(OUT / "streamlit-start.png"))
            page.set_viewport_size({"width": 1400, "height": 2600})  # Streamlit scrolls an inner container
            time.sleep(1.5)
            page.screenshot(path=str(OUT / "streamlit-results.png"))
            b.close()
    finally:
        srv.terminate()


if __name__ == "__main__":
    main()
