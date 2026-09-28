# Standard library
from pathlib import Path
from typing import Literal

# Third-party libraries
import mujoco as mj
import numpy as np
import numpy.typing as npt
from mujoco import viewer

# Local libraries (ARIEL)
from ariel import console
from ariel.body_phenotypes.robogen_lite.modules.core import CoreModule
from ariel.body_phenotypes.robogen_lite.prebuilt_robots.gecko import gecko
from ariel.body_phenotypes.robogen_lite.prebuilt_robots.spider import spider
from ariel.ec import set_seed
from ariel.simulation.environments import SimpleFlatWorld, OlympicArena
from ariel.utils.renderers import single_frame_renderer, video_renderer
from ariel.utils.runners import simple_runner
from ariel.utils.video_recorder import VideoRecorder

# Type aliases
type ViewerTypes = Literal["launcher", "video", "simple", "frame", "no_control"]

# --- RANDOM GENERATOR SETUP --- #
# Fix the seed while you are debugging.
# Report results over MULTIPLE seeds.
SEED = 42
RNG = np.random.default_rng(SEED)

# ariel.ec's own generators/mutators/crossover draw from a separate,
# package-level RNG. Reseed it too if you build your EA on ariel.ec,
# or every one of your "multiple seeds" runs the same variation operators.
set_seed(SEED)

# --- DATA SETUP --- #
SCRIPT_NAME = Path(__file__).stem
CWD = Path.cwd()
DATA = CWD / "__data__" / SCRIPT_NAME
DATA.mkdir(parents=True, exist_ok=True)

# --- EXPERIMENT CONSTANTS --- #
SPAWN_POS: list[float] = [-1.0, 0.0, 0.1]  # where the robot starts, i think this is the flat part of the olympic arena
TARGET_POSITION: list[float] = [2.0, 0.0, 0.1]  # where it should end up, might need to be lik 5 for the olympic arena
SIM_DURATION: float = 15.0  # seconds of simulated time per evaluation
MODE: ViewerTypes = "launcher"  # see run_experiment() for the options
TILT_LIMIT = 0.5        #tilt will count as tipped over
ARRIVAL_RADIUS = 0.30   # closer than this (metres) counts as that its reached the targer

HIDDEN_SIZE = 8 #For now setting the number of hidden nodes to 8
INPUT_SIZE = 30 #Input size 30 to include more stuff

def build_world() -> OlympicArena:
    """Create the environment the robot lives in.
    """
    return OlympicArena()

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
    return spider()

def genotype_to_weights(
    genotype: npt.NDArray[np.float64],
    input_size: int,
    output_size: int,
) -> list[npt.NDArray[np.float64]]:

    num_w1 = input_size * HIDDEN_SIZE

    w1 = genotype[:num_w1].reshape(input_size, HIDDEN_SIZE)
    w2 = genotype[num_w1:].reshape(HIDDEN_SIZE, output_size)

    return [w1, w2]


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
    distance = np.linalg.norm(
        target[:2] - position[:2]
    )

    inputs = np.concatenate([
        position,
        orientation,
        linear_velocity,
        joint_positions,
        joint_velocities,
        relative_target,
        [distance],
    ])

    # --- FORWARD PASS ----------------------------------------------------- #
    layer1 = np.tanh(inputs @ w1)
    outputs = np.tanh(layer1 @ w2)  # in [-1, 1]

    # --- RESCALE TO THE HINGE RANGE --------------------------------------- #
    return outputs * (np.pi / 2)  # in [-pi/2, pi/2]




def get_core_position(data: mj.MjData) -> npt.NDArray[np.float64]:
    """Return the robot core's current (x, y, z) world position."""
    return np.asarray(data.qpos[0:3]).copy()


def fitness_function(initial_position, final_position, fell, time_to_target):
    """Score one evaluation. LOWER IS BETTER.

    Three parts added together:
      1. distance change: negative if the robot got closer to the target
      2. speed: 0 to 1 if it reached the target (sooner = lower), 1 if it never did
      3. fall: +10 if the robot tipped over at any point
    """
    target = np.asarray(TARGET_POSITION)

    start_dist = np.linalg.norm(initial_position[:2] - target[:2])
    end_dist = np.linalg.norm(final_position[:2] - target[:2])
    score = end_dist - start_dist                      # part 1

    if time_to_target is not None:                     # part 2
        score += time_to_target / SIM_DURATION
    else:
        score += 1.0

    if fell:                                           # part 3
        score += 10.0

    return float(score)

##########################################################3

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

    qx0, qy0 = data.qpos[4], data.qpos[5]
    up_z0 = 1 - 2 * (qx0**2 + qy0**2)
    print(f"starting up_z: {up_z0:.3f}")

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
    console.log(f"start  : {np.round(initial_position, 3)}")
    console.log(f"end    : {np.round(final_position, 3)}")
    console.log(f"target : {np.round(TARGET_POSITION, 3)}")
    console.log(f"fitness: {fitness:.4f}   (lower is better)")

    return fitness


def main() -> None:
    """Run a single demo evaluation with a randomly-weighted controller."""
    # A quick look at the size of the problem you are about to search.
    mj.set_mjcb_control(None)
    world = build_world()
    robot = build_robot()
    world.spawn(
        robot.spec,
        position=SPAWN_POS,
        correct_collision_with_floor=True,
    )
    model = world.spec.compile()

    input_size = INPUT_SIZE
    output_size = model.nu
    num_weights = (
        input_size * HIDDEN_SIZE
        + HIDDEN_SIZE * output_size
    )
    console.log(f"controller inputs                  : {input_size}")
    console.log(f"controller outputs                 : {output_size}")
    console.log(f"genotype length (total weights)    : {num_weights}")
        
    genotype = RNG.normal(
    loc=0.0,
    scale=0.5,
    size=num_weights,
    )

    run_experiment(genotype, MODE)


if __name__ == "__main__":
    main()

