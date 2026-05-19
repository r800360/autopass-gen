from pathlib import Path
import os
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "agents"))
os.environ.setdefault("AUTOPASS_MOCK_LLM", "1")

collect_ignore = ["agents/test_full_simulation.py", "agents/test_navigate_modes.py"]
