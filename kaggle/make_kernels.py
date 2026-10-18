"""Generate one Kaggle kernel folder per job from the template ``kernel_train/run.py``.

    python kaggle/make_kernels.py tests timing          # writes kaggle/kernels/<slug>/{run.py,kernel-metadata.json}
    python kaggle/kg.py kernels push -p kaggle/kernels/rs-uep-jscc-timing

Separate kernels (own slug) can run in parallel; each is bounded by per-step timeouts.
"""
import json
import os
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
OWNER = "vaishakprasadbhat"
GPU_JOBS = {"timing", "smoke", "train_base", "train_phase2", "train_kn", "train_all", "v2_a", "v2_b", "v3_a", "v3_b", "v1x_a", "v1x_b"}


def make(spec: str) -> Path:
    """``spec`` is a job name, optionally ``job:cpu`` to force a CPU session (no GPU queue)."""
    job, _, hardware = spec.partition(":")
    force_cpu = hardware == "cpu"
    slug = "rs-uep-jscc-" + job.replace("_", "-") + ("-cpu" if force_cpu else "")
    folder = HERE / "kernels" / slug
    folder.mkdir(parents=True, exist_ok=True)
    code = (HERE / "kernel_train" / "run.py").read_text(encoding="utf-8")
    code = re.sub(r'^JOB = "[a-z_]+"', f'JOB = "{job}"', code, count=1, flags=re.M)
    schemes = os.environ.get("EVAL_SCHEMES")  # fixed from the validation selection before test jobs are generated
    sources = os.environ.get("EVAL_SOURCES")
    if sources:
        code = re.sub(r'^EVAL_SOURCES = "[^"]*"', f'EVAL_SOURCES = "{sources}"', code, count=1, flags=re.M)
    if schemes:
        code = re.sub(r'^EVAL_SCHEMES = "[^"]*"', f'EVAL_SCHEMES = "{schemes}"', code, count=1, flags=re.M)
    (folder / "run.py").write_text(code, encoding="utf-8")
    metadata = {
        "id": f"{OWNER}/{slug}", "title": slug, "code_file": "run.py", "language": "python",
        "kernel_type": "script", "is_private": "true", "enable_gpu": "true" if (job in GPU_JOBS and not force_cpu) else "false",
        "enable_tpu": "false", "enable_internet": "true", "dataset_sources": ["pankrzysiu/cifar10-python"] + (["vaishakprasadbhat/rs-uep-jscc-models"] if job.startswith(("eval_", "evalval_")) else []), "competition_sources": [],
        "kernel_sources": [], "model_sources": [],
    }
    (folder / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return folder


if __name__ == "__main__":
    for job in sys.argv[1:]:
        print(make(job))
