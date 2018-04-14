# Bound A* Search. The Bound is the max time from the probe pickup to the lab
# Graph search with cost function and heuristic -> A*

# Notes:
## Underestimated minimum distance ist the sum of the minimum from all rows in the DRIVING_TIMES Matrix
## Best algorithm would be 2-Opt + Simulated annealing (i guess)
import numpy as np
np.set_printoptions(threshold=np.nan) # completely print big arrays
from Opt2 import opt2
from Clustering import cluserting

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
    DRIVING_TIMES = np.array(DRIVING_TIMES).astype(np.int16)

# Remove the Exchange points:
DRIVING_TIMES_SLICED = DRIVING_TIMES[0:-NUM_EXCHANGE, 0:-NUM_EXCHANGE]
nodes = np.shape(DRIVING_TIMES_SLICED)[0]

CLUSTER_COUNT = VEHICLE_COUNT # this changes later
cluserting(DRIVING_TIMES_SLICED, CLUSTER_COUNT)
# cluster_Array =
# cluster_Array Array von Arrays mit Punkten die sich im jeweiligen Cluster befinden

# For the 2-opt alg. implementation just use all Nodes. Later there will be clusters
cluster_Array = np.linspace(0,nodes-1,nodes)
CLUSTER_COUNT = 1
sorted_Array, completion_time = opt2(cluster_Array, DRIVING_TIMES_SLICED, CLUSTER_COUNT)
# for cluster in cluster_Array: # later there will be more than one cluster

# VEHICLE_COUNT wird erhoeht von Mutterprogramm (abhaenglig von 2-opt)