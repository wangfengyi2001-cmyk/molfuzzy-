# MolFuzzy

**An open-source Python toolkit for molecular-geometry-based fuzzy multi-criteria decision support.**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.9%2B-blue)](pyproject.toml)
[![Tests](https://img.shields.io/badge/tests-pytest-green)](tests/)

MolFuzzy provides a reusable, four-stage pipeline for group multi-criteria
decision making (MCDM) under Molecular Fuzzy Set (MFS) uncertainty:

1. **Expert weighting** — a Linear Optimization Method (LOM) turns
   per-expert reliability dimensions into a normalized weight vector via
   goal-programming linear optimization.
2. **Evaluation balancing** — a Q-learning reward/penalty loop pulls every
   expert's evaluation matrix towards the LOM-selected benchmark expert
   until the matrices converge.
3. **Criterion weighting** — a cognitive map, built from geometry-normalized
   angles between fuzzy vectors, produces stabilized, interaction-aware
   criterion weights.
4. **Alternative ranking** — an Ant Colony Optimization (ACO) colony treats
   alternatives as cities and searches for the highest-fitness visiting
   sequence, yielding a full ranking rather than a single top pick.

Each stage is a standalone, independently testable module; the
`DecisionPipeline` class chains all four for the common case. The library
is domain-independent — the shipped example applies it to a manufacturing
investment-strategy problem, but any MCDM problem expressed as linguistic
expert judgements over criteria and alternatives can be solved the same
way.

## Why MolFuzzy

Existing fuzzy-MCDM tooling in the Python ecosystem typically implements a
single ranking method (e.g. TOPSIS or VIKOR) over a single fuzzy
environment. MolFuzzy instead packages a *complete pipeline* — uncertainty
representation, expert-reliability weighting, consensus-building, causal
criteria weighting and path-based alternative ranking — for the Molecular
Fuzzy Set environment, which represents uncertainty through both numerical
degrees and molecular-geometry angles (linear, trigonal planar,
tetrahedral, trigonal bipyramidal, octahedral). This lets users test how
sensitive their conclusions are to the choice of geometry and to the
Q-learning rate with a single function call (`molfuzzy.sensitivity`).

## Installation

```bash
pip install -e .
# or, without an editable install:
pip install .
```

Requires Python ≥ 3.9, NumPy ≥ 1.24 and SciPy ≥ 1.10 (installed
automatically). Optional: `matplotlib` for plotting sensitivity charts
(`pip install -e ".[plot]"`), `pytest` for the test suite
(`pip install -e ".[dev]"`).

## Quickstart

```python
from molfuzzy import DecisionPipeline

pipeline = DecisionPipeline(
    criterion_labels=["EE", "ADC", "EC", "PM"],
    alternative_labels=["MNT", "JTM", "SMRT", "SCH", "AI"],
)

result = pipeline.run(
    criteria_matrices=criteria_matrices,   # one linguistic matrix per expert
    decision_matrices=decision_matrices,   # one linguistic matrix per expert
)

print(result.summary())
# Criterion weights:
#   EE: 0.2465
#   ADC: 0.2546
#   EC: 0.2454
#   PM: 0.2535
#
# Alternative ranking (best first): JTM > AI > SMRT > MNT > SCH
# Path fitness: 1.3958
```

A complete, runnable version of the above — including the linguistic
input matrices — is in
[`examples/lean_energy_case_study.py`](examples/lean_energy_case_study.py).

### Command-line interface

```bash
molfuzzy run examples/lean_energy_scenario.json
molfuzzy run examples/lean_energy_scenario.json --geometry tetrahedral --learning-rate 0.5
molfuzzy sensitivity examples/lean_energy_scenario.json
```

### Sensitivity analysis

```python
from molfuzzy.sensitivity import run_sensitivity_analysis, rankings_are_stable

rows = run_sensitivity_analysis(
    criterion_labels, alternative_labels, criteria_matrices, decision_matrices,
    learning_rates=(0.1, 0.5, 1.0),
)
print(rankings_are_stable(rows))
```

## Package layout

```
molfuzzy/
  mfv.py             Molecular Fuzzy Value type + geometry angle utilities
  data.py            Linguistic scale (Negligible..High) <-> MFV conversions
  lom.py             Stage 1: Linear Optimization Method expert weighting
  qlearning.py       Stage 2: Q-learning evaluation-matrix balancing
  cognitive_map.py   Stage 3: cognitive-map criterion weighting
  aco.py             Stage 4: Ant Colony Optimization alternative ranking
  pipeline.py        DecisionPipeline: end-to-end orchestrator
  sensitivity.py      Multi-geometry / multi-learning-rate sweeps
  cli.py             `molfuzzy` command-line entry point
examples/
  lean_energy_case_study.py   Worked example (Python API)
  lean_energy_scenario.json   Same example (CLI / JSON input)
tests/                        pytest suite (17 tests, all four stages + pipeline)
```

## Implementation notes

Two aspects of the underlying methodology are expert-elicited /
model-design quantities that are not fully pinned down by closed-form
formulas: the per-expert reliability sub-dimensions feeding the LOM stage,
and the ACO exponents `alpha`/`beta`. MolFuzzy ships documented,
overridable defaults for both (`molfuzzy.lom.compute_reliability_dimensions`,
and `alpha=1.0, beta=2.0` in `aco_rank`) so results are reproducible
out of the box, while letting users substitute their own reliability
measures or exponents where they have domain-specific knowledge. See the
docstrings in `lom.py` and `aco.py` for details.

## Running the tests

```bash
pip install -e ".[dev]"
pytest tests/ -v
```

## License

MIT — see [LICENSE](LICENSE).

## Citation

If you use MolFuzzy in academic work, please cite the accompanying
SoftwareX paper (see [`CITATION.cff`](CITATION.cff)).
