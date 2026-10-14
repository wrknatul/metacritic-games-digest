import json
import os
from pathlib import Path

import pytest

# settings are read at import time, so the environment is prepared first
os.environ.setdefault("SCHEDULER_ENABLED", "false")
os.environ.setdefault("LLM_API_KEY", "")
os.environ.setdefault("LETSPLAY_ENABLED", "false")

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture
def fixture_text():
    return lambda name: (FIXTURES / name).read_text(encoding="utf-8")


@pytest.fixture
def fixture_json():
    return lambda name: json.loads((FIXTURES / name).read_text(encoding="utf-8"))
