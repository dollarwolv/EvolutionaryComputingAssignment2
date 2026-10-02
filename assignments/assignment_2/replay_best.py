from pathlib import Path
import numpy as np
from our_answer import run_experiment

def main() -> None:
    best_file = Path(__file__).parent / "outputs" / "best_phenotype.txt"
    best_genotype = np.loadtxt(best_file)
    run_experiment(best_genotype, mode="launcher")


if __name__ == "__main__":
    main()
