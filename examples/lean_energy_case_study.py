"""
Illustrative example: adaptive lean-energy manufacturing investment
strategy selection.

Reproduces the worked case study of the reference methodology (three
experts, four criteria -- EE, ADC, EC, PM -- and five process
alternatives -- MNT, JTM, SMRT, SCH, AI) using MolFuzzy end to end:
LOM expert weighting -> Q-learning balancing -> cognitive-map criterion
weighting -> ACO alternative ranking.

Run with:  python examples/lean_energy_case_study.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from molfuzzy import DecisionPipeline

CRITERIA = ["EE", "ADC", "EC", "PM"]
ALTERNATIVES = ["MNT", "JTM", "SMRT", "SCH", "AI"]

# --- Table 2: linguistic evaluations of the criteria (criteria x criteria) ---
# N=Negligible, L=Low, M=Moderate, S=Significant, H=High
CRITERIA_MATRICES = [
    # Expert 1
    [
        [None, "H", "H", "H"],
        ["M", None, "S", "S"],
        ["M", "H", None, "S"],
        ["H", "H", "S", None],
    ],
    # Expert 2
    [
        [None, "S", "H", "H"],
        ["H", None, "S", "S"],
        ["H", "H", None, "S"],
        ["S", "S", "H", None],
    ],
    # Expert 3
    [
        [None, "H", "H", "H"],
        ["S", None, "H", "S"],
        ["S", "H", None, "S"],
        ["S", "S", "H", None],
    ],
]

# --- Table 2: linguistic evaluations of the alternatives (alternatives x criteria) ---
DECISION_MATRICES = [
    # Expert 1
    [
        ["H", "S", "S", "S"],  # MNT
        ["S", "S", "H", "H"],  # JTM
        ["H", "S", "H", "H"],  # SMRT
        ["S", "S", "H", "H"],  # SCH
        ["S", "S", "S", "S"],  # AI
    ],
    # Expert 2
    [
        ["M", "M", "S", "S"],
        ["M", "M", "S", "S"],
        ["S", "M", "H", "H"],
        ["H", "M", "S", "S"],
        ["S", "M", "S", "S"],
    ],
    # Expert 3
    [
        ["S", "M", "S", "S"],
        ["M", "M", "S", "S"],
        ["S", "S", "S", "H"],
        ["S", "S", "S", "H"],
        ["S", "M", "S", "H"],
    ],
]


def main() -> None:
    pipeline = DecisionPipeline(
        criterion_labels=CRITERIA,
        alternative_labels=ALTERNATIVES,
        geometry="linear",
        learning_rate=0.1,
    )
    result = pipeline.run(
        criteria_matrices=CRITERIA_MATRICES,
        decision_matrices=DECISION_MATRICES,
    )

    print("=== Stage 1: LOM expert weights ===")
    for i, w in enumerate(result.expert_weights["weights"], start=1):
        tag = " (benchmark)" if i - 1 == result.expert_weights["benchmark"] else ""
        print(f"  Expert {i}: {w:.4f}{tag}")

    print("\n=== Stage 2: Q-learning balancing ===")
    print(f"  Iterations to convergence: {result.balancing.iterations}")

    print("\n=== Stage 3: Cognitive-map criterion weights ===")
    for label, w in zip(CRITERIA, result.criteria.weights):
        print(f"  {label}: {w:.4f}")
    order = sorted(zip(CRITERIA, result.criteria.weights), key=lambda x: -x[1])
    print("  Ranking: " + " > ".join(c for c, _ in order))

    print("\n=== Stage 4: ACO alternative ranking ===")
    print("  " + " > ".join(result.ranking.ranking_labels))
    print(f"  Path fitness: {result.ranking.fitness:.4f}")
    print(f"  Iterations to convergence: {result.ranking.iterations}")

    print("\n" + result.summary())


if __name__ == "__main__":
    main()
