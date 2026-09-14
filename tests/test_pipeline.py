import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from molfuzzy import DecisionPipeline

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))
from lean_energy_case_study import ALTERNATIVES, CRITERIA, CRITERIA_MATRICES, DECISION_MATRICES


def test_full_pipeline_runs_and_produces_valid_ranking():
    pipeline = DecisionPipeline(
        criterion_labels=CRITERIA,
        alternative_labels=ALTERNATIVES,
    )
    result = pipeline.run(
        criteria_matrices=CRITERIA_MATRICES,
        decision_matrices=DECISION_MATRICES,
    )
    assert result.criteria.weights.sum() == pytest.approx(1.0)
    assert sorted(result.ranking.ranking_labels) == sorted(ALTERNATIVES)
    # The two most-recommended process alternatives reported in the source
    # case study (AI feedback loops and just-in-time energy use) should
    # both surface in the top half of the MolFuzzy ranking.
    top_half = result.ranking.ranking_labels[: len(ALTERNATIVES) // 2 + 1]
    assert "AI" in top_half
    assert "JTM" in top_half


def test_pipeline_is_reproducible_with_same_seed():
    pipeline = DecisionPipeline(
        criterion_labels=CRITERIA, alternative_labels=ALTERNATIVES, aco_seed=123
    )
    r1 = pipeline.run(CRITERIA_MATRICES, DECISION_MATRICES)
    r2 = pipeline.run(CRITERIA_MATRICES, DECISION_MATRICES)
    assert r1.ranking.ranking_labels == r2.ranking.ranking_labels
