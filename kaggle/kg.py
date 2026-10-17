"""Run the Kaggle CLI with credentials taken from the repo-local, git-ignored ``.env``.

    python kaggle/kg.py kernels status vaishakprasadbhat/rs-uep-jscc-train

Nothing is read from outside the project folder, and no secret is ever printed.
"""
import os
import shutil
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from experiments.reproducibility import load_local_env  # noqa: E402


def main() -> int:
    load_local_env()
    local = REPO / ".venv" / ("Scripts" if os.name == "nt" else "bin") / ("kaggle.exe" if os.name == "nt" else "kaggle")
    executable = str(local) if local.exists() else shutil.which("kaggle")
    if not executable:
        raise SystemExit("kaggle CLI not found; install it into .venv (see README)")
    return subprocess.call([executable, *sys.argv[1:]])


if __name__ == "__main__":
    sys.exit(main())
