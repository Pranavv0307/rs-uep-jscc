"""Erasure sweep with the frozen static oracle ranking (and, optionally, more sources).

Thin wrapper over ``experiments/eval_rs_uep.py`` (shared code, so mask pairing,
side-info accounting, seed statistics and provenance cannot drift). It makes
``--oracle_ranking_file`` mandatory and defaults the importance sources to the
oracle.

The ranking must come from the 40k training split
(``experiments/train_adjsccq_train_split.py`` writes ``oracle_ranking.json``).
A ranking whose ``source_split`` is not ``train`` triggers a warning and is
recorded in the output metadata.

Example::

    python -m experiments.eval_rs_uep_oracle \\
        --checkpoint checkpoints/train_split_adjsccq/best_val.pth \\
        --oracle_ranking_file checkpoints/train_split_adjsccq/oracle_ranking.json \\
        --split test --seeds 0,1,2,3,4 --include_no_rs --output_dir results/eval/oracle_test
"""
from experiments.eval_rs_uep import build_parser, run_and_save


def main():
    parser = build_parser(__doc__)
    parser.set_defaults(importance="oracle", output_dir="results/eval/oracle_sweep", name="oracle_sweep")
    args = parser.parse_args()
    if not args.oracle_ranking_file:
        parser.error("--oracle_ranking_file is required")
    run_and_save(args)


if __name__ == "__main__":
    main()
