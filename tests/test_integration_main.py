import subprocess
import sys
from pathlib import Path


def test_main_script_runs_successfully():
    """
    Integration test.

    Goal:
    - Run `python main.py` as a subprocess
    - Verify that it finishes without errors (return code = 0)
    - Verify that expected text appears in the console output
    """

    # Project root directory
    repo_root = Path(__file__).resolve().parents[1]

    # Run main.py
    result = subprocess.run(
        [sys.executable, "main.py"],
        cwd=repo_root,
        capture_output=True,
        text=True,
    )

    # 1) Script should not crash
    assert result.returncode == 0, f"main.py failed: {result.stderr}"

    # 2) Output should contain some expected info
    assert "Accuracy" in result.stdout