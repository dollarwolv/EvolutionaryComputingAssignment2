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
            results[label] = pd.concat(pd.read_csv(f) for f in files)
    return results


def create_plot() -> None:
    results = load_results()
    if not results:
        raise ValueError(f"no results found in {OUTPUTS}")

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for label, df in results.items():
        grouped = df.groupby("generation")["best_so_far"]
        mean, std = grouped.mean(), grouped.std().fillna(0.0)
        n_runs = df["run"].nunique()
        ax.plot(mean.index, mean, label=f"{LABELS[label]} (n={n_runs})")
        ax.fill_between(mean.index, mean - std, mean + std, alpha=0.2)

    ax.set_xlabel("Generation")
    ax.set_ylabel("Best so far fitness")
    ax.set_title("Best so far fitness: mean ± std over runs")
    ax.legend()
    ax.grid(alpha=0.3)

    fig.savefig(OUTPUTS / f"best_so_far.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    create_plot()
