# Bound A* Search. The Bound is the max time from the probe pickup to the lab
# Graph search with cost function and heuristic -> A*

# Notes:
## Underestimated minimum distance ist the sum of the minimum from all rows in the DRIVING_TIMES Matrix
## Best algorithm would be 2-Opt + Simulated annealing (i guess)
import numpy as np
np.set_printoptions(threshold=np.nan) # completely print big arrays

# Configure:
INSTANCE = "Testinstanzen/1_10.txt"
VEHICLE_COUNT = 1
###################################

NUM_EXCHANGE = 0
NUM_DOCS = 0
MAX_TRANSFER_TIME = 0
MAX_TIME = 0
DRIVING_TIMES = []

with open(INSTANCE) as infile:
    driving_times = False
    driving_times_line = 0
    for line in infile:
        if "NUM_EXCHANGE" in line:
            NUM_EXCHANGE = int(line[13:])
        elif "NUM_DOCS" in line:
            NUM_DOCS = int(line[9:])
        elif "MAX_TRANSFER_TIME" in line:
            MAX_TRANSFER_TIME = int(line[18:])
        elif "MAX_TIME" in line:
            MAX_TIME = int(line[9:])
        elif "DRIVING_TIMES" in line:
            driving_times = True
        elif driving_times:
            # if driving_times_line == 0:
            DRIVING_TIMES.append([int(n) for n in line.rstrip().split()])
            driving_times_line += 1
    DRIVING_TIMES = np.array(DRIVING_TIMES).astype(np.int16)

nodes = np.shape(DRIVING_TIMES)[0]