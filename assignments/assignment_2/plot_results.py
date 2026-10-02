import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


HERE = Path(__file__).parent

def create_plot(plot_type: str):
    if not plot_type:
        raise ValueError("no plot type given")

    df = pd.read_csv(HERE / "outputs" / f"dataset_{plot_type}.csv")

    for run in df["run"].unique():
        run_data = df[df["run"] == run]
        plt.plot(run_data["generation"], run_data["best_so_far"], label=f"Run {run}")

    plt.xlabel("Generation")
    plt.ylabel("Best-so-far fitness")
    plt.title(f"{plot_type.capitalize()} Crossover: Best-so-far fitness over generations")

    plt.savefig(HERE / "outputs" / f"{plot_type}.png", dpi=300, bbox_inches="tight")
