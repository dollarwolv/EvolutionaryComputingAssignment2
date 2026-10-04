import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

HERE = Path(__file__).parent
OUTPUTS = HERE / "outputs"

LABELS = {
    "uniform": "EA - uniform crossover",
    "neuron_block": "EA - neuron-block crossover",
    "random": "Random search (baseline)",
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
        )
        ax.fill_between(mean.index, mean - std, mean + std, alpha=0.2)

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


if __name__ == "__main__":
    create_plot()
