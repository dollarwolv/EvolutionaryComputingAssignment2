import numpy as np
import numpy.typing as npt
from parameters import *


def has_plateaued(
    best_fitnesses: list[float],
    patience: int = PLATEAU_PATIENCE,
    min_delta: float = PLATEAU_MIN_DELTA,
) -> bool:
    """Return True when a minimisation run has stopped improving.

    ``best_fitnesses`` must contain the best-so-far value after each
    generation. Lower values are better in this assignment.
    """
    if patience <= 0:
        raise ValueError("patience must be greater than zero")
    if min_delta < 0:
        raise ValueError("min_delta cannot be negative")

    # We need one value before the patience window and one at its end.
    if len(best_fitnesses) <= patience:
        return False

    old_best = best_fitnesses[-patience - 1]
    current_best = best_fitnesses[-1]
    improvement = old_best - current_best

    return improvement <= min_delta


def genotype_to_weights(
    genotype: npt.NDArray[np.float64],
    input_size: int,
    output_size: int,
    hidden_size: int = HIDDEN_SIZE,
) -> list[npt.NDArray[np.float64]]:

    num_w1 = input_size * hidden_size

    w1 = np.asarray(genotype[:num_w1]).reshape(input_size, hidden_size)
    w2 = np.asarray(genotype[num_w1:]).reshape(hidden_size, output_size)

    return [w1, w2]


def weights_to_genotype(weights: list[npt.NDArray[np.float64]]):
    w1, w2 = weights
    return np.concatenate([w1.ravel(), w2.ravel()]).tolist()
