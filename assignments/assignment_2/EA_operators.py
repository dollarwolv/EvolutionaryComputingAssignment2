import numpy as np
import numpy.typing as npt
from parameters import *
from helpers import *
from ariel.ec import Population, Individual

def mutate_weights(
    weights: list[npt.NDArray[np.float64]],
    sigma: float = 0.2,
    mutation_prob: float = MUTATION_PROBABILITY,
) -> list[npt.NDArray[np.float64]]:
    """
    Mutates weights by gaussian mutation.

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

def mutate(
        population: Population,
        controller_output_size: int,
        sigma: float = 0.2,
        mutation_prob: float = MUTATION_PROBABILITY,
) -> Population:
    for ind in population.where(lambda ind: bool(ind.tags.get("mutate", False))):
        weights = genotype_to_weights(
            np.asarray(ind.genotype, dtype=np.float64),
            INPUT_SIZE,
            controller_output_size,
            HIDDEN_SIZE,
        )
        mutated_weights = mutate_weights(weights, sigma, mutation_prob)
        ind.genotype = weights_to_genotype(mutated_weights)
        ind.requires_eval = True
    return population

def tournament_selection(
    genotypes: list[list[float]],
    fitnesses: list[float],
    k: int = TOURNAMENT_SIZE,
) -> list[float]:
    """Pick one parent: sample k individuals at random, return the best.
    LOWER fitness is better, matching fitness_function.
    """
    candidate_idxs = RNG.choice(len(genotypes), size=k, replace=False)
    best_idx = min(candidate_idxs, key=lambda i: fitnesses[i])
    return genotypes[best_idx]

def parent_selection(population: Population) -> Population:
    shuffled = population.shuffle()
    for idx in range(0, len(shuffled) - 1, 2):
        ind_a = shuffled[idx]
        ind_b = shuffled[idx + 1]
        if ind_a.fitness_ is not None and ind_b.fitness_ is not None:
            if ind_a.fitness_ >= ind_b.fitness_:
                ind_a.tags = {"selected": True}
                ind_b.tags = {"selected": False}
            else:
                ind_a.tags = {"selected": False}
                ind_b.tags = {"selected": True}

    return shuffled

def crossover_weights(
    parents: tuple[list[npt.NDArray[np.float64]], list[npt.NDArray[np.float64]]],
    crossover_type: str = "uniform",
) -> tuple[list[npt.NDArray[np.float64]], list[npt.NDArray[np.float64]]]:
    """
    Crosses over two parents to create two children.
    Two types of crossover are supported: uniform crossover and hidden-neuron-block crossover.
    In uniform crossover, for each gene, there is a 50% chance that it comes from parent 1 or parent 2.
    In hidden-neuron-block crossover, Connections to and from neurons in the hidden layer are preserved. The connections
    may come from either parent.

    Args:
        parents: The parents that should be crossed over
        crossover_type: The type of crossover that should be performed. Can be either 'neuron_block' or 'uniform'.
    """

    parent_1, parent_2 = parents

    if len(parent_1) != len(parent_2):
        raise ValueError("Parents must have the same number of matrices")

    for layer_index, (matrix_1, matrix_2) in enumerate(zip(parent_1, parent_2)):
        if matrix_1.shape != matrix_2.shape:
            raise ValueError(
                f"Parent matrices at index {layer_index} must have the same shape"
            )

    child_1 = [np.empty_like(array) for array in parent_1]
    child_2 = [np.empty_like(array) for array in parent_2]

    if crossover_type == "neuron_block":

        p1_w1, p1_w2 = parent_1
        p2_w1, p2_w2 = parent_2

        c1_w1 = np.empty_like(p1_w1)
        c1_w2 = np.empty_like(p1_w2)
        c2_w1 = np.empty_like(p2_w1)
        c2_w2 = np.empty_like(p2_w2)

        # loop through neurons in hidden layer and copy its connections into children
        for hidden_idx in range(parent_1[0].shape[1]):
            picked_parent = RNG.choice(["p1", "p2"])

            if picked_parent == "p1":
                c1_w1[:, hidden_idx] = p1_w1[:, hidden_idx]
                c1_w2[hidden_idx, :] = p1_w2[hidden_idx, :]

                c2_w1[:, hidden_idx] = p2_w1[:, hidden_idx]
                c2_w2[hidden_idx, :] = p2_w2[hidden_idx, :]

            elif picked_parent == "p2":
                c1_w1[:, hidden_idx] = p2_w1[:, hidden_idx]
                c1_w2[hidden_idx, :] = p2_w2[hidden_idx, :]

                c2_w1[:, hidden_idx] = p1_w1[:, hidden_idx]
                c2_w2[hidden_idx, :] = p1_w2[hidden_idx, :]

        child_1 = [c1_w1, c1_w2]
        child_2 = [c2_w1, c2_w2]

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
        raise ValueError("crossover_type must be in ('uniform', 'neuron_block')")

    return child_1, child_2

def crossover(
        population: Population, 
        controller_output_size: int,
        crossover_type: str
) -> Population:
    parents = population.where(lambda ind: bool(ind.tags.get("selected", False)))
    for idx in range(0, len(parents) - 1, 2):
        p_a = parents[idx]
        p_b = parents[idx + 1]
        
        parent_1_weights = genotype_to_weights(
            np.asarray(p_a.genotype, dtype=np.float64),
            INPUT_SIZE,
            controller_output_size,
            HIDDEN_SIZE,
        )
        parent_2_weights = genotype_to_weights(
            np.asarray(p_b.genotype, dtype=np.float64),
            INPUT_SIZE,
            controller_output_size,
            HIDDEN_SIZE,
        )

        child_1_weights, child_2_weights = crossover_weights(
            (parent_1_weights, parent_2_weights),
            crossover_type=crossover_type,
        )

        child_1_genotype = weights_to_genotype(child_1_weights)
        child_2_genotype = weights_to_genotype(child_2_weights)

        child_a = Individual()
        child_a.genotype = child_1_genotype
        child_a.tags = {"mutate": True}

        child_b = Individual()
        child_b.genotype = child_2_genotype
        child_b.tags = {"mutate": True}

        population.extend([child_a, child_b])
    return population

def survivor_selection(population: Population) -> Population:
    shuffled = population.alive.shuffle()
    alive_count = len(shuffled)
    for idx in range(0, len(shuffled) - 1, 2):
        if alive_count <= POPULATION_SIZE:
            break
        ind_a = shuffled[idx]
        ind_b = shuffled[idx + 1]
        if (ind_a.fitness_ or 0.0) >= (ind_b.fitness_ or 0.0):
            ind_b.alive = False
        else:
            ind_a.alive = False
        alive_count -= 1
    return population