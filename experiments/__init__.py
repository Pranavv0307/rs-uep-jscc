import sys as _sys

# Insights contain symbols such as "±" and "Δ"; a Windows console defaults to cp1252 and would crash on print.
for _stream in (_sys.stdout, _sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

# Credentials live in the git-ignored <repo>/.env (see experiments/reproducibility.py); load them for every script
# so W&B and Kaggle always use the project's accounts, never whatever happens to be logged in on the machine.
try:
    from experiments.reproducibility import load_local_env as _load_local_env

    _load_local_env()
except Exception:  # noqa: BLE001 - a missing .env must never break an experiment
    pass
