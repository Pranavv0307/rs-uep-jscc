"""Thin, failure-tolerant Weights & Biases wrapper used by every script.

Logging is always optional (``--wandb``). If wandb is missing, not logged in or
offline, the experiment still runs and all results are still written to disk;
the wrapper only prints a warning. Set ``WANDB_MODE=offline`` to log locally.
"""
import json
import os
from typing import Dict, Iterable, List, Optional


def init_run(enabled: bool, project: str, name: Optional[str], config: Optional[dict] = None,
             tags: Optional[List[str]] = None, group: Optional[str] = None, job_type: Optional[str] = None):
    """Return a wandb run, or ``None`` when logging is disabled / unavailable."""
    if not enabled:
        return None
    try:
        import wandb
        return wandb.init(project=project, name=name, config=config, tags=tags, group=group, job_type=job_type,
                          reinit=True)
    except Exception as error:  # noqa: BLE001 - logging must never kill an experiment
        print(f"WARNING: wandb disabled ({type(error).__name__}: {error})")
        return None


def log_metrics(run, metrics: Dict, step: Optional[int] = None) -> None:
    if run is not None:
        run.log(metrics, step=step)


def set_summary(run, summary: Dict) -> None:
    if run is not None:
        for key, value in summary.items():
            run.summary[key] = value


def _cell(value):
    if isinstance(value, (int, float, str, bool)) or value is None:
        return value
    return json.dumps(value, default=str)


def log_table(run, key: str, rows: Iterable[Dict]) -> None:
    rows = list(rows)
    if run is None or not rows:
        return
    import wandb
    columns = list(rows[0].keys())
    table = wandb.Table(columns=columns)
    for row in rows:
        table.add_data(*[_cell(row.get(column)) for column in columns])
    run.log({key: table})


def log_images(run, key: str, paths: Iterable[str]) -> None:
    if run is None:
        return
    import wandb
    for path in paths:
        if path.lower().endswith((".png", ".jpg", ".jpeg")) and os.path.exists(path):
            run.log({f"{key}/{os.path.splitext(os.path.basename(path))[0]}": wandb.Image(path)})


def log_artifact(run, name: str, artifact_type: str, paths: Iterable[str]) -> None:
    if run is None:
        return
    import wandb
    artifact = wandb.Artifact(name, type=artifact_type)
    for path in paths:
        if os.path.exists(path):
            artifact.add_file(path)
    run.log_artifact(artifact)


def finish(run) -> None:
    if run is not None:
        run.finish()
