"""Generate the figures used in the SoftwareX paper and repo docs."""
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "examples"))

OUT = Path(__file__).resolve().parent / "figures"
OUT.mkdir(exist_ok=True)

plt.rcParams.update(
    {
        "font.size": 11,
        "font.family": "DejaVu Sans",
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)

# ---------------------------------------------------------------------------
# Figure 1: software architecture / data flow
# ---------------------------------------------------------------------------

def fig_architecture():
    fig, ax = plt.subplots(figsize=(11, 4.4))
    ax.set_xlim(0, 11)
    ax.set_ylim(0, 4.4)
    ax.axis("off")

    stages = [
        ("Stage 1\nExpert weighting", "molfuzzy.lom", "#2b6cb0"),
        ("Stage 2\nEvaluation balancing", "molfuzzy.qlearning", "#2f855a"),
        ("Stage 3\nCriterion weighting", "molfuzzy.cognitive_map", "#805ad5"),
        ("Stage 4\nAlternative ranking", "molfuzzy.aco", "#c05621"),
    ]
    x0 = 0.4
    width = 2.15
    gap = 0.35
    y0 = 1.5
    height = 1.7

    for i, (title, module, color) in enumerate(stages):
        x = x0 + i * (width + gap)
        box = mpatches.FancyBboxPatch(
            (x, y0),
            width,
            height,
            boxstyle="round,pad=0.02,rounding_size=0.08",
            linewidth=1.5,
            edgecolor=color,
            facecolor=color,
            alpha=0.12,
        )
        ax.add_patch(box)
        ax.text(
            x + width / 2,
            y0 + height - 0.42,
            title,
            ha="center",
            va="center",
            fontsize=11,
            fontweight="bold",
            color=color,
        )
        ax.text(
            x + width / 2,
            y0 + 0.32,
            module,
            ha="center",
            va="center",
            fontsize=9.5,
            style="italic",
            color="#333333",
        )
        if i < len(stages) - 1:
            ax.annotate(
                "",
                xy=(x + width + gap - 0.05, y0 + height / 2),
                xytext=(x + width + 0.05, y0 + height / 2),
                arrowprops=dict(arrowstyle="-|>", color="#555555", lw=1.6),
            )

    # input / output labels
    ax.text(
        x0 + width / 2,
        y0 + height + 0.55,
        "Linguistic expert\nevaluations (JSON / Python)",
        ha="center",
        va="center",
        fontsize=9.5,
        color="#333333",
    )
    ax.annotate(
        "",
        xy=(x0 + width / 2, y0 + height + 0.05),
        xytext=(x0 + width / 2, y0 + height + 0.42),
        arrowprops=dict(arrowstyle="-|>", color="#555555", lw=1.4),
    )

    last_x = x0 + (len(stages) - 1) * (width + gap)
    ax.text(
        last_x + width / 2,
        y0 + height + 0.55,
        "Criterion weights +\nranked alternatives",
        ha="center",
        va="center",
        fontsize=9.5,
        color="#333333",
    )
    ax.annotate(
        "",
        xy=(last_x + width / 2, y0 + height + 0.42),
        xytext=(last_x + width / 2, y0 + height + 0.05),
        arrowprops=dict(arrowstyle="-|>", color="#555555", lw=1.4),
    )

    ax.text(
        x0 + 2 * (width + gap) + width / 2 - gap / 2,
        y0 - 0.55,
        "molfuzzy.mfv  (Molecular Fuzzy Value + molecular-geometry angle utilities, shared by stages 3-4)"
        "\nmolfuzzy.pipeline.DecisionPipeline orchestrates stages 1-4  |  molfuzzy.cli exposes them on the command line",
        ha="center",
        va="center",
        fontsize=9,
        color="#555555",
    )

    fig.tight_layout()
    fig.savefig(OUT / "fig1_architecture.png", dpi=300)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 2: illustrative example output (criterion weights + ranking)
# ---------------------------------------------------------------------------

def fig_example_output():
    from molfuzzy import DecisionPipeline
    from lean_energy_case_study import (
        ALTERNATIVES,
        CRITERIA,
        CRITERIA_MATRICES,
        DECISION_MATRICES,
    )

    pipeline = DecisionPipeline(criterion_labels=CRITERIA, alternative_labels=ALTERNATIVES)
    result = pipeline.run(CRITERIA_MATRICES, DECISION_MATRICES)

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))

    order = sorted(zip(CRITERIA, result.criteria.weights), key=lambda x: -x[1])
    labels, weights = zip(*order)
    colors = ["#805ad5" if i == 0 else "#b8a6dd" for i in range(len(labels))]
    axes[0].bar(labels, weights, color=colors)
    axes[0].set_title("Criterion weights (cognitive map)")
    axes[0].set_ylabel("Weight")
    for i, w in enumerate(weights):
        axes[0].text(i, w + 0.001, f"{w:.3f}", ha="center", fontsize=9)

    ranking = result.ranking.ranking_labels
    y_pos = np.arange(len(ranking))[::-1]
    colors2 = ["#c05621" if i == 0 else "#e8b894" for i in range(len(ranking))]
    axes[1].barh(y_pos, [len(ranking) - i for i in range(len(ranking))], color=colors2)
    axes[1].set_yticks(y_pos)
    axes[1].set_yticklabels(ranking)
    axes[1].set_xlabel("Rank position (best = longest bar)")
    axes[1].set_title("ACO alternative ranking")
    axes[1].set_xticks([])

    fig.tight_layout()
    fig.savefig(OUT / "fig2_example_output.png", dpi=300)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 3: sensitivity analysis across learning rates and geometries
# ---------------------------------------------------------------------------

def fig_sensitivity():
    from molfuzzy.sensitivity import run_sensitivity_analysis
    from lean_energy_case_study import (
        ALTERNATIVES,
        CRITERIA,
        CRITERIA_MATRICES,
        DECISION_MATRICES,
    )

    rows = run_sensitivity_analysis(
        CRITERIA, ALTERNATIVES, CRITERIA_MATRICES, DECISION_MATRICES
    )

    # Use lambda for the Q-learning rate: alpha is already taken by the ACO
    # pheromone exponent in the same paper/pipeline, so overloading it here
    # invites misreading the axis.
    scenario_labels = [f"{r.geometry}\nλ={r.learning_rate}" for r in rows]
    weights_by_criterion = {c: [] for c in CRITERIA}
    for r in rows:
        for c, w in zip(CRITERIA, r.weights):
            weights_by_criterion[c].append(w)

    fig, ax = plt.subplots(figsize=(11, 4.2))
    x = np.arange(len(rows))
    palette = {"EE": "#2b6cb0", "ADC": "#c05621", "EC": "#2f855a", "PM": "#805ad5"}
    for c in CRITERIA:
        ax.plot(x, weights_by_criterion[c], marker="o", label=c, color=palette.get(c))
    ax.set_xticks(x)
    ax.set_xticklabels(scenario_labels, fontsize=7.5, rotation=90)
    ax.set_ylabel("Criterion weight")
    ax.set_title("Criterion-weight sensitivity across molecular geometries and Q-learning rates")
    ax.legend(ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.32), frameon=False)
    fig.tight_layout()
    fig.savefig(OUT / "fig3_sensitivity.png", dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    fig_architecture()
    fig_example_output()
    fig_sensitivity()
    print("Figures written to", OUT)
