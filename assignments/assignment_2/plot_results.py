import pandas as pd
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

HERE = Path(__file__).parent
OUTPUTS = HERE / "outputs"

LABELS = {
    "uniform": "EA - uniform crossover",
    "neuron_block": "EA - neuron-block crossover",
    "mutation_only": "EA - mutation only",
    "random": "Random search (baseline)",
}

# Fixed colours ensure that a condition has the same colour in every figure.
COLORS = {
    "uniform": "tab:blue",
    "neuron_block": "tab:orange",
    "mutation_only": "tab:green",
    "random": "tab:red",
}

SHORT_LABELS = {
    "uniform": "Uniform\ncrossover",
    "neuron_block": "Neuron-block\ncrossover",
    "mutation_only": "Mutation\nonly",
    "random": "Random\nsearch",
}


def load_results() -> dict[str, pd.DataFrame]:
    results = {}
    for label in LABELS:
        files = sorted((OUTPUTS / label).glob("run_*.csv"))
        if files:
            results[label] = pd.concat(
                (pd.read_csv(f) for f in files),
                ignore_index=True,
            )
    return results


def create_plot() -> None:
    results = load_results()
    if not results:
        raise ValueError(f"no results found in {OUTPUTS}")

    # Use one shared x-axis, even when algorithms stop at different times.
    max_generation = max(int(df["generation"].max()) for df in results.values())

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for label, df in results.items():

        curves = df.pivot(
            index="generation",
            columns="run",
            values="best_so_far",
        )
        curves = curves.reindex(range(max_generation + 1)).ffill()

        mean = curves.mean(axis=1)
        std = curves.std(axis=1).fillna(0.0)
        n_runs = df["run"].nunique()
        stop_rows = df.loc[df.groupby("run")["generation"].idxmax()]
        median_stop = int(stop_rows["generation"].median())

        (line,) = ax.plot(
            mean.index,
            mean,
            label=f"{LABELS[label]} (n={n_runs}, median stop={median_stop})",
            color=COLORS[label],
        )
        ax.fill_between(
            mean.index,
            mean - std,
            mean + std,
            color=COLORS[label],
            alpha=0.2,
        )

        # Show the actual stopping generation of every individual run.
        ax.scatter(
            stop_rows["generation"],
            stop_rows["best_so_far"],
            color=line.get_color(),
            marker="x",
            s=30,
            zorder=3,
        )

    ax.set_xlabel("Generation")
    ax.set_ylabel("Best so far fitness")
    ax.set_title("Best so far fitness: mean ± std over runs")
    ax.legend()
    ax.grid(alpha=0.3)

    fig.text(
        0.5,
        0.01,
        "× = stopping generation; final best is carried forward after stopping.",
        ha="center",
        fontsize=8,
    )
    fig.tight_layout(rect=(0, 0.04, 1, 1))

    fig.savefig(OUTPUTS / "best_so_far.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def calculate_summary_values(
    results: dict[str, pd.DataFrame],
) -> tuple[int, dict[str, list[float]], dict[str, list[float]]]:
    """Calculate AOCC and terminal fitness for every run."""
    run_ids = sorted(results[next(iter(results))]["run"].astype(int).unique())

    # Every curve is compared over the longest interval completed by all runs.
    common_budget = min(
        int(df.groupby("run")["generation"].max().min())
        for df in results.values()
    )

    curves: dict[str, list[np.ndarray]] = {}
    terminal_values: dict[str, list[float]] = {}

    for condition, df in results.items():
        curves[condition] = []
        terminal_values[condition] = []

        for run_id in run_ids:
            run = df[df["run"] == run_id].set_index("generation")
            curve = run["best_so_far"].reindex(range(common_budget + 1))
            curves[condition].append(curve.to_numpy(dtype=float))
            terminal_values[condition].append(float(run["best_so_far"].iloc[-1]))

    all_fitness_values = np.concatenate(
        [curve for condition_curves in curves.values() for curve in condition_curves]
    )
    lower_bound = float(all_fitness_values.min())
    upper_bound = float(all_fitness_values.max())

    # Lower fitness is better, so the subtraction is reversed. This makes
    # higher normalized AOCC values represent better anytime performance.
    aocc_values = {
        condition: [
            float(np.mean((upper_bound - curve) / (upper_bound - lower_bound)))
            for curve in condition_curves
        ]
        for condition, condition_curves in curves.items()
    }

    return common_budget, aocc_values, terminal_values


def add_boxplots_and_points(
    ax: plt.Axes,
    values: dict[str, list[float]],
) -> None:
    """Show both the distribution summary and all individual runs."""
    conditions = list(LABELS)
    positions = np.arange(1, len(conditions) + 1)
    data = [values[condition] for condition in conditions]

    boxplot = ax.boxplot(
        data,
        positions=positions,
        widths=0.55,
        patch_artist=True,
        showfliers=False,
        medianprops={"color": "black", "linewidth": 1.5},
    )

    for box, condition in zip(boxplot["boxes"], conditions, strict=True):
        box.set_facecolor(COLORS[condition])
        box.set_alpha(0.3)
        box.set_edgecolor(COLORS[condition])

    # Small fixed offsets keep similar run values from being drawn on top of
    # each other while avoiding random movement between generated figures.
    for position, condition in zip(positions, conditions, strict=True):
        condition_values = values[condition]
        offsets = np.linspace(-0.13, 0.13, len(condition_values))
        ax.scatter(
            position + offsets,
            condition_values,
            color=COLORS[condition],
            edgecolor="white",
            linewidth=0.5,
            s=34,
            zorder=3,
        )

    ax.set_xticks(
        positions,
        [SHORT_LABELS[condition] for condition in conditions],
    )
    ax.grid(axis="y", alpha=0.3)


def create_aocc_plot(
    common_budget: int,
    aocc_values: dict[str, list[float]],
) -> None:
    """Plot equal-budget anytime performance as AOCC distributions."""
    fig, ax = plt.subplots(figsize=(6.4, 4.8))
    add_boxplots_and_points(ax, aocc_values)

    ax.set_title(
        f"Equal-budget anytime performance (generations 0--{common_budget})"
    )
    ax.set_ylabel("Normalized AOCC")
    ax.text(
        0.02,
        0.97,
        "Higher is better",
        transform=ax.transAxes,
        va="top",
        fontsize=9,
    )

    fig.tight_layout()
    fig.savefig(OUTPUTS / "aocc_distribution.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def create_terminal_fitness_plot(
    terminal_values: dict[str, list[float]],
) -> None:
    """Plot fitness measured at each run's plateau-based stopping point."""
    fig, ax = plt.subplots(figsize=(6.4, 4.8))
    add_boxplots_and_points(ax, terminal_values)

    ax.set_title("Terminal fitness at plateau (variable stopping generations)")
    ax.set_ylabel("Terminal best-so-far fitness")
    ax.text(
        0.02,
        0.97,
        "Lower is better",
        transform=ax.transAxes,
        va="top",
        fontsize=9,
    )

    fig.tight_layout()
    fig.savefig(
        OUTPUTS / "terminal_fitness_distribution.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)


def create_distribution_plots() -> None:
    """Create the two distribution figures from the saved experiment runs."""
    results = load_results()
    missing_conditions = set(LABELS) - set(results)
    if missing_conditions:
        missing = ", ".join(sorted(missing_conditions))
        raise ValueError(f"missing result files for: {missing}")

    common_budget, aocc_values, terminal_values = calculate_summary_values(results)
    create_aocc_plot(common_budget, aocc_values)
    create_terminal_fitness_plot(terminal_values)


if __name__ == "__main__":
    create_plot()
    create_distribution_plots()
