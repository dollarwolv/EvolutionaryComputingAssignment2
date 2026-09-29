import numpy as np
import numpy.typing as npt
from parameters import *


def mutate(
    weights: list[npt.NDArray[np.float64]],
    sigma: float = 0.2,
    mutation_prob: float = 0.2,
) -> list[npt.NDArray[np.float64]]:
    """
    Mutates weights by non-uniform mutation.

    Args:
        weights: The weights of the NN.
        sigma: the standard devitation of the normal distribution from which the perturbations are drawn.
        mutation_prob: the probability of a mutation occuring at all, per weight.
    """

    def perturb(w: np.float64, sigma: float):
        if RNG.random() < mutation_prob:
            perturbation = RNG.normal(scale=sigma)
            w = w + perturbation

        return w

    perturbed_weights = []

    for weight_arr in weights:
        vectorize_perturb = np.vectorize(perturb)
        perturbed_weights.append(vectorize_perturb(weight_arr, sigma))

    return perturbed_weights


def crossover(
    parents: tuple[list[npt.NDArray[np.float64]], list[npt.NDArray[np.float64]]],
    crossover_type: str = "uniform",
) -> tuple[list[npt.NDArray[np.float64]], list[npt.NDArray[np.float64]]]:
    """
    Crosses over two parents to create two children

    Args:
        parents: The parents that should be crossed over
        crossover_type: The type of crossover that should be performed. Can be either
    """

    parent_1, parent_2 = parents

    if len(parent_1) != len(parent_2):
        raise ValueError("Parents must have the same number of matrices")

    for layer_index, (matrix_1, matrix_2) in enumerate(zip(parent_1, parent_2)):
        if matrix_1.shape != matrix_2.shape:
            raise ValueError(
                f"Parent matrices at index {layer_index} must have the same shape"
            )

    child_1: list[npt.NDArray[np.float64]] = [
        np.empty(shape=array.shape, dtype=np.float64) for array in parent_1
    ]

    child_2: list[npt.NDArray[np.float64]] = [
        np.empty(shape=array.shape, dtype=np.float64) for array in parent_2
    ]

    if crossover_type == "layer_wise":

        # loop through the layers
        for layer_idx, (
            p_1_weights,
            p_2_weights,
            c_1_weights,
            c_2_weights,
        ) in enumerate(zip(parent_1, parent_2, child_1, child_2)):
            crossover_layer_type = RNG.choice(["row_wise", "column_wise"])
            if crossover_layer_type == "row_wise":

                # inside the layer, loop through rows
                for row in range(p_1_weights.shape[0]):

                    # pick the parent that c1 inherits from
                    picked_parent = RNG.choice(["p1", "p2"])

                    if picked_parent == "p1":
                        c_1_weights[row] = p_1_weights[row]
                        c_2_weights[row] = p_2_weights[row]
                    elif picked_parent == "p2":
                        c_1_weights[row] = p_2_weights[row]
                        c_2_weights[row] = p_1_weights[row]

            elif crossover_layer_type == "column_wise":

                # inside the layer, loop through rows
                for col in range(p_1_weights.shape[1]):

                    # pick the parent that c1 inherits from
                    picked_parent = RNG.choice(["p1", "p2"])

                    if picked_parent == "p1":
                        c_1_weights[:, col] = p_1_weights[:, col]
                        c_2_weights[:, col] = p_2_weights[:, col]
                    elif picked_parent == "p2":
                        c_1_weights[:, col] = p_2_weights[:, col]
                        c_2_weights[:, col] = p_1_weights[:, col]

            child_1[layer_idx] = c_1_weights
            child_2[layer_idx] = c_2_weights

    elif crossover_type == "uniform":

        # loop through layers
        for layer_idx, (
            p_1_weights,
            p_2_weights,
            c_1_weights,
            c_2_weights,
        ) in enumerate(zip(parent_1, parent_2, child_1, child_2)):

            # flatten matrices so that they're easier to loop through
            flattened_p_1, flattened_p_2, flattened_c_1, flattened_c_2 = (
                p_1_weights.flatten(),
                p_2_weights.flatten(),
                c_1_weights.flatten(),
                c_2_weights.flatten(),
            )

            # loop through weights of parents and perform uniform crossover
            for w_idx, (p_1_weight, p_2_weight) in enumerate(
                zip(flattened_p_1, flattened_p_2)
            ):
                picked_parent = RNG.choice(["p1", "p2"])

                if picked_parent == "p1":
                    flattened_c_1[w_idx] = p_1_weight
                    flattened_c_2[w_idx] = p_2_weight
                elif picked_parent == "p2":
                    flattened_c_1[w_idx] = p_2_weight
                    flattened_c_2[w_idx] = p_1_weight

            # turn children back into correct shape
            reshaped_c_1, reshaped_c_2 = (
                flattened_c_1.reshape(p_1_weights.shape[0], p_1_weights.shape[1]),
                flattened_c_2.reshape(p_1_weights.shape[0], p_1_weights.shape[1]),
            )

            child_1[layer_idx], child_2[layer_idx] = (
                reshaped_c_1,
                reshaped_c_2,
            )

        else:
            raise ValueError("crossover_type must be in ('uniform', 'layer_wise')")

    return child_1, child_2
