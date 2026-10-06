import numpy as np
from pathlib import Path

# Type aliases
type ViewerTypes = Literal["launcher", "video", "simple", "frame", "no_control"]
from typing import Literal

# --- DATA SETUP --- #
SCRIPT_NAME = Path(__file__).stem
CWD = Path.cwd()
DATA = CWD / "__data__" / SCRIPT_NAME
DATA.mkdir(parents=True, exist_ok=True)


SPAWN_POS: list[float] = [0.0, 0.0, 0.1]  # where the robot starts for flat
TARGET_POSITION: list[float] = [2.0, 0.0, 0.1]  # where it should end up for flat

SIM_DURATION: float = 15.0  # seconds of simulated time per evaluation
MODE: ViewerTypes = "launcher"  # see run_experiment() for the options
TILT_LIMIT = 0.5  # tilt will count as tipped over
ARRIVAL_RADIUS = 0.30  # closer than this (metres) counts as that its reached the targer

HIDDEN_SIZE = 8  # For now setting the number of hidden nodes to 8
INPUT_SIZE = 33
GAIT_FREQUENCY = 1.0  # Hz of the sin/cos clock fed to the network
MUTATION_PROBABILITY = 0.05
MUTATION_SIGMA = 0.2
TOURNAMENT_SIZE = 3

RNG = np.random.default_rng()

# --- EVOLUTIONARY ALGORITHM CONSTANTS --- #
POPULATION_SIZE = 30
PLATEAU_PATIENCE = 200
PLATEAU_MIN_DELTA = 0.01
MAX_GENERATIONS = 500
NUM_RUNS = 5
BASE_SEED = 42
