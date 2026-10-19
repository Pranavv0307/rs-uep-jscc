# Preflight

{'PASS': 27, 'WARN': 4, 'FAIL': 2, 'SKIP': 0}

| status | check | detail |
|---|---|---|
| PASS | python | 3.14.0 on Windows-11-10.0.26200-SP0 |
| PASS | package torch | 2.10.0+cpu |
| PASS | package torchvision | 0.25.0+cpu |
| PASS | package numpy | 2.4.0 |
| PASS | package PIL | 12.1.0 |
| PASS | package matplotlib | 3.10.8 |
| PASS | package yaml | 6.0.3 |
| PASS | package tqdm | 4.70.1 |
| PASS | package pytest | 9.1.1 |
| PASS | package wandb | 0.27.0 |
| WARN | cuda | no GPU: training will be CPU-only |
| PASS | cpu cores | 16 |
| PASS | Pillow JPEG 2000 | available |
| PASS | Pillow WebP | available |
| WARN | lpips (optional) | not installed: LPIPS column is skipped on Kodak |
| WARN | BPG codec (optional) | not installed: digital baseline uses JPEG 2000 / JPEG / WebP |
| PASS | repo root | C:\Users\Subhash\Documents\vaisakkkkkkkkkkk\rs-uep-jscc |
| PASS | git branch | phase0-5-pipeline @ a4f8b0c |
| WARN | git working tree | uncommitted changes (results will record dirty_repo=true) |
| PASS | secrets and outputs are git-ignored | ok |
| PASS | no credential file is tracked | ok |
| PASS | free disk space | 42.5 GB |
| PASS | results/ is writable |  |
| PASS | dataset cifar10 | 50000 train / 10000 test, md5 verified |
| FAIL | dataset cifar100 | RuntimeError: not usable in C:\Users\Subhash\Documents\vaisakkkkkkkkkkk\rs-uep-jscc\data (download: python -m experiments.download_data --cifar100) |
| FAIL | dataset kodak | 0/24 images in C:\Users\Subhash\Documents\vaisakkkkkkkkkkk\rs-uep-jscc\data\kodak |
| PASS | seed-42 checkpoint best_val.pth | sha256 b07b868ade0abd3c... |
| PASS | seed-42 checkpoint oracle_ranking.json | sha256 2c3cb3a23dba10a1..., ranking is a permutation: True |
| PASS | W&B key in environment or .env | present (86 characters) |
| PASS | W&B live test | logged one run to project 'rs-uep-jscc': https://wandb.ai/vaishak-prasad-bhat/rs-uep-jscc/runs/frk8wppx |
| PASS | Kaggle credentials | KAGGLE_API_TOKEN set in .env |
| PASS | kaggle package | importable |
| PASS | Kaggle API login | authenticated |
