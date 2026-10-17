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
    # Call the CLI through the interpreter, so it works wherever the package is installed.
    code = "import sys; from kaggle.cli import main; sys.argv[0] = 'kaggle'; main()"
    return subprocess.call([sys.executable, "-c", code, *sys.argv[1:]])


if __name__ == "__main__":
    sys.exit(main())
