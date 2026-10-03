"""The local probe must run without keys and never label simulation as live."""

import json
import subprocess
import sys
from pathlib import Path


def test_mock_probe_runs_as_installed_code(tmp_path):
    path = Path(__file__).resolve().parents[2] / "scripts" / "smoke_models.py"
    assert path.exists(), "model probe script is missing"
    process = subprocess.run([sys.executable, str(path), "--output-dir", str(tmp_path)],
                             capture_output=True, text=True, encoding="utf-8", timeout=60)
    assert process.returncode == 0, process.stderr
    result = json.loads(process.stdout)
    assert result["mode"] == "mock"
    assert result["live_verified"] is False
    assert result["embedding_dimension"] == 1024
    assert result["chroma_query_succeeded"] is True
