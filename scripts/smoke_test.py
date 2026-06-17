"""Headless smoke test: run every Streamlit page and fail on any uncaught exception."""
import sys
from pathlib import Path

from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "app"
sys.path.insert(0, str(APP))  # so pages can `import lib`

PAGES = [APP / "Home.py", *sorted((APP / "pages").glob("*.py"))]

failures = 0
for page in PAGES:
    at = AppTest.from_file(str(page), default_timeout=90)
    at.run()
    if at.exception:
        failures += 1
        print(f"FAIL  {page.name}")
        for exc in at.exception:
            print(f"      {exc.type}: {exc.message}")
    else:
        print(f"OK    {page.name}  (widgets: {len(at.selectbox) + len(at.slider)})")

print(f"\n{'ALL PAGES OK' if failures == 0 else f'{failures} PAGE(S) FAILED'}")
sys.exit(1 if failures else 0)
