import os
import subprocess
import sys
from pathlib import Path


SCRIPT = Path(__file__).parents[2] / "scripts" / "seed_demo_data.py"


def test_seed_script_creates_default_sqlite_parent_directory(tmp_path) -> None:
    clean_cwd = tmp_path / "clean-project"
    clean_cwd.mkdir()
    environment = os.environ.copy()
    environment.pop("SMART_CS_DATABASE_URL", None)

    completed = subprocess.run(
        [sys.executable, str(SCRIPT)],
        cwd=clean_cwd,
        env=environment,
        capture_output=True,
        text=True,
        check=True,
    )

    assert (clean_cwd / "data" / "smart_cs.db").exists()
    assert "C001" in completed.stdout
    assert "O1001" in completed.stdout
