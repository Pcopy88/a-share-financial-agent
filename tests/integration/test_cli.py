import os
import subprocess
import sys


def test_module_command_help_and_doctor_without_credentials(tmp_path):
    env = os.environ | {"PYTHONIOENCODING": "utf-8", "ARA_MODE": "mock",
                        "ARA_DATA_DIR": str(tmp_path / "data")}
    for command in (["--help"], ["doctor"]):
        result = subprocess.run([sys.executable, "-m", "annual_report_agent", *command],
            env=env, capture_output=True, text=True, encoding="utf-8", timeout=30)
        assert result.returncode == 0, result.stderr
        assert result.stdout.strip()
    assert "DEEPSEEK_API_KEY" in result.stdout
