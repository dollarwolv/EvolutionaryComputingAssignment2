import argparse
from pathlib import Path
import numpy as np
from our_answer import run_experiment

def main() -> None:
    p = argparse.ArgumentParser(description="Replay a saved best genotype")
    p.add_argument("--label", default="uniform", help="uniform, neuron_block or random")
    p.add_argument("--run", type=int, default=1)
    p.add_argument("--mode", default="launcher", help="launcher or video")
    args = p.parse_args()

    best_file = Path(__file__).parent / "outputs" / args.label / f"best_genotype_run_{args.run}.txt"
    best_genotype = np.loadtxt(best_file)
    fitness = run_experiment(best_genotype, mode=args.mode)
    print(f"fitness: {fitness:.4f}")


if __name__ == "__main__":
    main()
