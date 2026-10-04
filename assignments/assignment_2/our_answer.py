# Third-party libraries
import mujoco as mj
import numpy as np
import numpy.typing as npt
from mujoco import viewer
import argparse
from pathlib import Path
import torch
import random
import pandas as pd

# Local libraries (ARIEL)
from ariel import console
from ariel.body_phenotypes.robogen_lite.modules.core import CoreModule
from ariel.ec import Population, Individual, EA, EAOperation
from ariel.ec import set_seed as ariel_set_seed
from ariel.simulation.environments import SimpleFlatWorld
from ariel.body_phenotypes.robogen_lite.prebuilt_robots.john_set import spider_8
from ariel.utils.renderers import single_frame_renderer, video_renderer
from ariel.utils.runners import simple_runner
from ariel.utils.video_recorder import VideoRecorder

from parameters import *
from helpers import genotype_to_weights, has_plateaued
from EA_operators import parent_selection, crossover, mutate, survivor_selection
from plot_results import create_plot


# ariel.ec's own generators/mutators/crossover draw from a separate,
# package-level RNG. Reseed it too if you build your EA on ariel.ec,
# or every one of your "multiple seeds" runs the same variation operators.
# function to set seed for reach run
def set_seed(seed: int) -> None:
    RNG.bit_generator.state = np.random.default_rng(seed).bit_generator.state
    random.seed(seed)
    torch.manual_seed(seed)
    ariel_set_seed(seed)


HERE = Path(__file__).parent


# ============================================================================ #
#  1. THE BODY AND THE WORLD
# ============================================================================ #
def build_world() -> SimpleFlatWorld:
    """Create the environment the robot lives in."""
    return SimpleFlatWorld()


def build_robot() -> CoreModule:
    """Create the robot body.

    YOU MAY CHANGE THIS. Options include the prebuilt bodies in
    `ariel.body_phenotypes.robogen_lite.prebuilt_robots` (gecko, spider, ...).

    Two consequences of this choice, and they matter:
      * The body determines `model.nu` (the number of hinges you must send
        commands to) - that is the OUTPUT size of your controller.
      * The body determines the size of `data.qpos` - if you feed qpos to your
        network, that is (part of) your INPUT size.
    Change the body and your genotype length changes with it. Keep the body
    FIXED within an experiment.
    """
    return spider_8()


# ============================================================================ #
#  2. THE CONTROLLER CONTRACT
# ============================================================================ #
def nn_controller(
    model: mj.MjModel,
    data: mj.MjData,
    weights: list[npt.NDArray[np.float64]],
) -> npt.NDArray[np.float64]:
    """Map robot state to hinge commands: in -> hidden -> actions.

    In this demo `weights` is drawn at RANDOM. In your assignment, `weights`
    is what the evolutionary algorithm produces: an individual's genotype,
    reshaped into these matrices. You are free to change the architecture
    itself (layers, activations, ...) - just keep input/output sizes correct.

    Parameters
    ----------
    model : mj.MjModel
        The MuJoCo model. Use `model.nu` for the number of hinges.
    data : mj.MjData
        The MuJoCo data. This is where you read the robot's state from.
    weights : list of ndarray
        [w1, w2] - the layer weight matrices.

    Returns
    -------
    npt.NDArray[np.float64]
        `model.nu` action values, already scaled to [-pi/2, pi/2].
    """
    w1, w2 = weights

    # --- INPUTS ---------------------------------------------------------- #
    # Bare qpos - the simplest choice, not necessarily a good one. See
    # YOUR JOB below

    position = data.qpos[0:3]
    orientation = data.qpos[3:7]

    linear_velocity = data.qvel[0:3]

    joint_positions = data.qpos[7:15]
    joint_velocities = data.qvel[6:14]

    target = np.asarray(TARGET_POSITION)

    relative_target = target - position
    distance = np.linalg.norm(target[:2] - position[:2])

    # Clock signal so the network has something to drive a rhythmic gait with,
    # plus a constant 1 that acts as the bias for the hidden layer.
    phase = 2 * np.pi * GAIT_FREQUENCY * data.time
    clock = [np.sin(phase), np.cos(phase), 1.0]

    inputs = np.concatenate(
        [
            position,
            orientation,
            linear_velocity,
            joint_positions,
            joint_velocities,
            relative_target,
            [distance],
            clock,
        ]
    )

    # --- FORWARD PASS ----------------------------------------------------- #
    layer1 = np.tanh(inputs @ w1)
    outputs = np.tanh(layer1 @ w2)  # in [-1, 1]

    # --- RESCALE TO THE HINGE RANGE --------------------------------------- #
    return outputs * (np.pi / 2)  # in [-pi/2, pi/2]


# ============================================================================ #
#  3. POSITION AND FITNESS
# ============================================================================ #
def get_core_position(data: mj.MjData) -> npt.NDArray[np.float64]:
    """Return the robot core's current (x, y, z) world position."""
    return np.asarray(data.qpos[0:3]).copy()


def fitness_function(
    initial_position: npt.NDArray[np.float64],
    final_position: npt.NDArray[np.float64],
    fell: bool,
    time_to_target: float | None,
) -> float:
    """Score one evaluation. LOWER IS BETTER.

    Three parts added together:
      1. distance change: negative if the robot got closer to the target
      2. speed: 0 to 1 if it reached the target (sooner = lower), 1 if it never did
      3. fall: +10 if the robot tipped over at any point
    """
    target = np.asarray(TARGET_POSITION)

    start_dist = np.linalg.norm(initial_position[:2] - target[:2])
    end_dist = np.linalg.norm(final_position[:2] - target[:2])
    score = end_dist - start_dist  # part 1

    if time_to_target is not None:  # part 2
        score += time_to_target / SIM_DURATION
    else:
        score += 1.0

    if fell:  # part 3
        score += 10.0

    return float(score)


# ============================================================================ #
#  4. RUNNING ONE EVALUATION
# ============================================================================ #
def run_experiment(
    genotype: npt.NDArray[np.float64],
    mode: ViewerTypes = MODE,
) -> float:
    """Set up the world, run one simulation, and return the fitness.

    This is the function your EA calls once per individual, with `mode` set
    to "simple" (headless).

    Returns
    -------
    float
        The fitness of this run. Lower is better.
    """
    # MuJoCo's control callback is a GLOBAL. Clear it. DO NOT REMOVE.
    mj.set_mjcb_control(None)

    # --- World and robot --------------------------------------------------- #
    world = build_world()
    robot = build_robot()

    world.spawn(
        robot.spec,
        position=SPAWN_POS,
        correct_collision_with_floor=True,
    )

    # Compile the world into a model. USE AS IS.
    model = world.spec.compile()
    data = mj.MjData(model)

    # Put the simulation in a clean, known state before reading anything.
    mj.mj_resetData(model, data)
    mj.mj_forward(model, data)

    # qx0, qy0 = data.qpos[4], data.qpos[5]
    # up_z0 = 1 - 2 * (qx0**2 + qy0**2)
    # print(f"starting up_z: {up_z0:.3f}")

    # --- Wire up the controller -------------------------------------------- #
    # Sizes are read from the compiled model, never hardcoded - they depend on
    # the body you chose in build_robot().
    input_size = INPUT_SIZE
    output_size = model.nu

    weights = genotype_to_weights(genotype, input_size, output_size)
    log = {"fell": False, "time_to_target": None}
    target = np.asarray(TARGET_POSITION)

    def control_callback(m: mj.MjModel, d: mj.MjData) -> None:
        """Compute and apply actions; MuJoCo calls this every physics step."""
        actions = nn_controller(m, d, weights)

        # Has the robot tipped over?
        qx, qy = d.qpos[4], d.qpos[5]
        up_z = 1 - 2 * (qx**2 + qy**2)
        if up_z < TILT_LIMIT:
            log["fell"] = True

        # Has the robot reached the target yet? (only record the first time)
        dist = np.linalg.norm(d.qpos[0:2] - target[0:2])
        if log["time_to_target"] is None and dist < ARRIVAL_RADIUS:
            log["time_to_target"] = d.time
        # DIRECT application (see the controller contract above).
        d.ctrl[:] = actions

        # DELTA application - comment out the line above and use these instead:
        # delta = 0.05
        # d.ctrl[:] += actions * delta
        # d.ctrl[:] = np.clip(d.ctrl, -np.pi / 2, np.pi / 2)

    # --- Record the starting point ----------------------------------------- #
    initial_position = get_core_position(data)

    # --- Run ---------------------------------------------------------------- #
    if mode != "no_control":
        mj.set_mjcb_control(control_callback)

    match mode:
        case "launcher":
            # Interactive window. Great for seeing what your robot does,
            # useless inside an evolutionary loop.
            viewer.launch(model=model, data=data)
        case "simple":
            # Headless. THIS is the one your EA uses.
            simple_runner(model, data, duration=SIM_DURATION)
        case "video":
            # Render to an mp4 - for the figures in your report.
            recorder = VideoRecorder(output_folder=str(DATA / "__videos__"))
            video_renderer(
                model,
                data,
                duration=SIM_DURATION,
                video_recorder=recorder,
            )
        case "frame":
            # A single image of the scene. Useful to check your spawn position
            # and that the robot is not clipping through the floor.
            single_frame_renderer(model, data, steps=1, show=True)
        case "no_control":
            # No controller attached: drag the hinges around by hand.
            viewer.launch(model=model, data=data)

    # Detach the callback again so the next run starts clean.
    mj.set_mjcb_control(None)

    # --- Score -------------------------------------------------------------- #
    final_position = get_core_position(data)

    fitness = fitness_function(
        initial_position, final_position, log["fell"], log["time_to_target"]
    )
    # console.log(f"start  : {np.round(initial_position, 3)}")
    # console.log(f"end    : {np.round(final_position, 3)}")
    # console.log(f"target : {np.round(TARGET_POSITION, 3)}")
    # console.log(f"fitness: {fitness:.4f}   (lower is better)")

    return fitness


def evaluate(population: Population) -> Population:
    for ind in population.unevaluated:
        ind.fitness = run_experiment(ind.genotype, mode="simple")
    return population


def controller_output_size() -> int:
    """Return the output_size of the controller."""
    mj.set_mjcb_control(None)
    world = build_world()
    robot = build_robot()
    world.spawn(
        robot.spec,
        position=SPAWN_POS,
        correct_collision_with_floor=True,
    )
    model = world.spec.compile()
    output_size = model.nu

    return output_size


# Each generation, binary tournaments pick POPULATION_SIZE // 2 parents, which
# are paired up and produce 2 children per pair - so this many evaluations/gen.
OFFSPRING_PER_GENERATION = 2 * ((POPULATION_SIZE // 2) // 2)


def new_individual(num_weights: int) -> Individual:
    ind = Individual()
    ind.genotype = RNG.normal(loc=0.0, scale=0.5, size=num_weights).tolist()
    return ind


# Calculate summary statistics for the current population.
def get_stats(population: Population) -> dict:
    fitnesses = [ind.fitness_ for ind in population.alive if ind.fitness_ is not None]

    return {
        "best_fitness": min(fitnesses),
        "mean_fitness": np.mean(fitnesses),
        "std_fitness": np.std(fitnesses),
    }


def make_record(
    run: int,
    generation: int,
    stats: dict,
    previous: dict | None,
) -> dict:
    best_so_far = stats["best_fitness"]
    if previous is not None:
        best_so_far = min(best_so_far, previous["best_so_far"])

    return {
        "run": run,
        "generation": generation,
        **stats,
        "best_so_far": best_so_far,
    }


def log_stats(
    population: Population,
    this_run: list,
    run: int,
) -> Population:
    previous = this_run[-1]
    this_run.append(
        make_record(run, previous["generation"] + 1, get_stats(population), previous)
    )
    return population


def run_ea(
    run: int,
    num_weights: int,
    nn_output_size: int,
    crossover_type: str,
    label: str,
) -> tuple[list, list]:
    initial = Population([new_individual(num_weights) for _ in range(POPULATION_SIZE)])
    initial = evaluate(initial)

    this_run = [make_record(run, 0, get_stats(initial), None)]

    ea = EA(
        initial,
        [
            EAOperation(parent_selection),
            EAOperation(
                crossover,
                controller_output_size=nn_output_size,
                crossover_type=crossover_type,
            ),
            EAOperation(mutate, controller_output_size=nn_output_size),
            EAOperation(evaluate),
            EAOperation(survivor_selection),
            EAOperation(log_stats, this_run=this_run, run=run),
        ],
        is_maximisation=False,
        # one database per run, so runs in parallel terminals don't clash
        db_file_path=HERE / "__data__" / f"{label}_run_{run}.db",
    )

    while not has_plateaued([record["best_so_far"] for record in this_run]):
        ea.step()

    console.log(
        f"Run {run} reached a plateau at generation " f"{this_run[-1]['generation']}."
    )

    best_individual = ea.get_solution("best", only_alive=True)
    return this_run, best_individual.genotype


def run_random_search(run: int, num_weights: int) -> tuple[list, list]:
    """Baseline: sample random controllers with the same evaluation budget as the EA.

    Generation 0 samples POPULATION_SIZE controllers, every later "generation"
    samples OFFSPRING_PER_GENERATION new ones, exactly like the EA.
    """
    batch = evaluate(
        Population([new_individual(num_weights) for _ in range(POPULATION_SIZE)])
    )
    this_run = [make_record(run, 0, get_stats(batch), None)]
    best = batch.best(sort="min", attribute="fitness_", n=1)[0]

    generation = 0
    while not has_plateaued([record["best_so_far"] for record in this_run]):
        generation += 1
        batch = evaluate(
            Population(
                [new_individual(num_weights) for _ in range(OFFSPRING_PER_GENERATION)]
            )
        )
        this_run.append(make_record(run, generation, get_stats(batch), this_run[-1]))
        batch_best = batch.best(sort="min", attribute="fitness_", n=1)[0]
        if batch_best.fitness_ < best.fitness_:
            best = batch_best

    console.log(
        f"Run {run} reached a plateau at generation " f"{this_run[-1]['generation']}."
    )

    return this_run, best.genotype


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Brain Evolution",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--algorithm", choices=["ea", "random"], default="ea")
    p.add_argument(
        "--crossover", choices=["uniform", "neuron_block"], default="uniform"
    )
    p.add_argument(
        "--runs",
        type=int,
        nargs="+",
        default=list(range(1, NUM_RUNS + 1)),
        help="Which runs to execute (1-based); run r uses seed BASE_SEED + r - 1. "
        "Split runs over several terminals to use more CPU cores.",
    )
    args = p.parse_args()
    return args


def main() -> None:
    args = parse_args()
    label = args.crossover if args.algorithm == "ea" else "random"

    output_dir = HERE / "outputs" / label
    output_dir.mkdir(parents=True, exist_ok=True)

    nn_output_size = controller_output_size()
    num_weights = INPUT_SIZE * HIDDEN_SIZE + HIDDEN_SIZE * nn_output_size

    for run in args.runs:
        seed = BASE_SEED + run - 1
        set_seed(seed)

        if args.algorithm == "random":
            this_run, best_genotype = run_random_search(run, num_weights)
        else:
            this_run, best_genotype = run_ea(
                run, num_weights, nn_output_size, args.crossover, label
            )

        pd.DataFrame(this_run).to_csv(output_dir / f"run_{run}.csv", index=False)
        np.savetxt(
            output_dir / f"best_genotype_run_{run}.txt", np.asarray(best_genotype)
        )

    create_plot()


if __name__ == "__main__":
    main()
