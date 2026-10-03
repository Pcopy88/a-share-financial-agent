"""Catch installed entry points failing outside pytest's injected src path."""

import os
import subprocess
import sys


def test_installed_package_is_importable_from_another_directory(tmp_path):
    environment = os.environ.copy()
    environment.pop("PYTHONPATH", None)
    result = subprocess.run([sys.executable, "-c",
                             "import annual_report_agent; print(annual_report_agent.__version__)"],
                            cwd=tmp_path, env=environment, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "0.1.0"
