import numpy as np
from pathlib import Path

# Type aliases
type ViewerTypes = Literal["launcher", "video", "simple", "frame", "no_control"]
from typing import Literal

# --- RANDOM GENERATOR SETUP --- #
# Fix the seed while you are debugging.
# Report results over MULTIPLE seeds.
SEED = 42
RNG = np.random.default_rng(SEED)

# --- DATA SETUP --- #
SCRIPT_NAME = Path(__file__).stem
CWD = Path.cwd()
DATA = CWD / "__data__" / SCRIPT_NAME
DATA.mkdir(parents=True, exist_ok=True)

# --- EXPERIMENT CONSTANTS --- #
SPAWN_POS: list[float] = [
    -1.0,
    0.0,
    0.1,
]  # where the robot starts, i think this is the flat part of the olympic arena
TARGET_POSITION: list[float] = [
    5.42,
    0,
    0.2 + 0.106,
]  # where it should end up, might need to be lik 5 for the olympic arena
SIM_DURATION: float = 15.0  # seconds of simulated time per evaluation
MODE: ViewerTypes = "launcher"  # see run_experiment() for the options

HIDDEN_SIZE = 8  # For now setting the number of hidden nodes to 8
INPUT_SIZE = 30  # Input size 30 to include more stuff
MUTATION_PROBABILITY = 0.05
