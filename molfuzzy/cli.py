"""
Command-line interface for MolFuzzy.

Usage
-----
    molfuzzy run scenario.json
    molfuzzy run scenario.json --geometry tetrahedral --learning-rate 0.5
    molfuzzy sensitivity scenario.json

``scenario.json`` schema
-------------------------
{
  "criterion_labels": ["EE", "ADC", "EC", "PM"],
  "alternative_labels": ["MNT", "JTM", "SMRT", "SCH", "AI"],
  "criteria_matrices": [ [[null,"H","H","H"], ...], ... one per expert ... ],
  "decision_matrices": [ [["H","S","S","S"], ...], ... one per expert ... ]
}
Linguistic cells use single-letter codes N/L/M/S/H (Negligible ... High);
diagonal cells of ``criteria_matrices`` must be ``null``.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .mfv import GEOMETRIES
from .pipeline import DecisionPipeline
from .sensitivity import rankings_are_stable, run_sensitivity_analysis


def _load_scenario(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def _cmd_run(args: argparse.Namespace) -> int:
    scenario = _load_scenario(args.scenario)
    pipeline = DecisionPipeline(
        criterion_labels=scenario["criterion_labels"],
        alternative_labels=scenario["alternative_labels"],
        geometry=args.geometry,
        learning_rate=args.learning_rate,
        aco_seed=args.seed,
    )
    result = pipeline.run(
        criteria_matrices=scenario["criteria_matrices"],
        decision_matrices=scenario["decision_matrices"],
    )
    print(result.summary())
    if args.json:
        payload = {
            "criterion_weights": dict(
                zip(scenario["criterion_labels"], result.criteria.weights.tolist())
            ),
            "ranking": result.ranking.ranking_labels,
            "fitness": result.ranking.fitness,
        }
        print(json.dumps(payload, indent=2))
    return 0


def _cmd_sensitivity(args: argparse.Namespace) -> int:
    scenario = _load_scenario(args.scenario)
    rows = run_sensitivity_analysis(
        criterion_labels=scenario["criterion_labels"],
        alternative_labels=scenario["alternative_labels"],
        criteria_matrices=scenario["criteria_matrices"],
        decision_matrices=scenario["decision_matrices"],
    )
    for row in rows:
        print(
            f"lr={row.learning_rate:<4} geometry={row.geometry:<20} "
            f"order={' > '.join(row.ranking_order)}"
        )
    stable = rankings_are_stable(rows)
    # NOTE: this compares the *criterion-weight* rank order of Stage 3 only.
    # It says nothing about the Stage 4 alternative ranking, which is not
    # swept here (the ACO stage is stochastic; use a fixed --seed for
    # reproducible comparisons).
    print(f"\nCriterion-weight order stable across all scenarios: {stable}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="molfuzzy")
    sub = parser.add_subparsers(dest="command", required=True)

    run_p = sub.add_parser("run", help="Run the 4-stage decision pipeline once")
    run_p.add_argument("scenario", help="Path to a scenario JSON file")
    run_p.add_argument("--geometry", choices=GEOMETRIES, default="linear")
    run_p.add_argument("--learning-rate", type=float, default=0.1)
    run_p.add_argument("--seed", type=int, default=42)
    run_p.add_argument("--json", action="store_true", help="Also print JSON output")
    run_p.set_defaults(func=_cmd_run)

    sens_p = sub.add_parser(
        "sensitivity", help="Sweep learning rates x molecular geometries"
    )
    sens_p.add_argument("scenario", help="Path to a scenario JSON file")
    sens_p.set_defaults(func=_cmd_sensitivity)

    return parser


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
