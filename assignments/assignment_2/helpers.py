import numpy as np
import numpy.typing as npt
from parameters import *


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
