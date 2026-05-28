from pathlib import Path
import os
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "agents"))

# Offline test profile — production defaults are CARLA + real LLM (see autopass/config.py)
os.environ["AUTOPASS_TEST_MODE"] = "1"
os.environ["AUTOPASS_MOCK_LLM"] = "1"
os.environ["AUTOPASS_PERCEPTION_BACKEND"] = "visual"
os.environ["AUTOPASS_CONTROL_MODE"] = "kinematic"

collect_ignore = ["agents/test_full_simulation.py", "agents/test_navigate_modes.py"]
