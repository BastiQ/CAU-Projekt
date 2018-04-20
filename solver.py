# Bound A* Search. The Bound is the max time from the probe pickup to the lab
# Graph search with cost function and heuristic -> A*

# Notes:
## Underestimated minimum distance ist the sum of the minimum from all rows in the DRIVING_TIMES Matrix
## Best algorithm would be 2-Opt + Simulated annealing (i guess)
import numpy as np
import time
import copy
from Clustering import cluserting
from Opt2 import opt2
from Clustering import plot
np.set_printoptions(threshold=np.nan) # completely print big arrays

# Configure:
INSTANCE = "Testinstanzen/1_50.txt"
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
    DRIVING_TIMES = np.array(DRIVING_TIMES).astype(np.int32)

# Remove the Exchange points:
deleteExchangePoints = [p+1 for p in range(NUM_EXCHANGE)]
DRIVING_TIMES_SLICED = np.delete(DRIVING_TIMES, (deleteExchangePoints), axis=0)
DRIVING_TIMES_SLICED = np.delete(DRIVING_TIMES_SLICED, (deleteExchangePoints), axis=1)
nodes = np.shape(DRIVING_TIMES_SLICED)[0]

CLUSTER_COUNT = VEHICLE_COUNT # this changes later

points, clusterSets = cluserting(DRIVING_TIMES_SLICED, CLUSTER_COUNT)
# plot(points,clusterSets)

paths = []
for clusterNr, cluster in enumerate(clusterSets):
    B = copy.deepcopy(DRIVING_TIMES_SLICED) # B = DRIVING_TIMES des aktuellen Clusters
    delete = [d for d in range(B.shape[0]) if d not in cluster]
    B = np.delete(B, (delete), axis=0)
    B = np.delete(B, (delete), axis=1)

    # TODO: labor punkt muss noch mit rein
    print("search ideal path (2-opt)...")
    t0 = time.time()
    path, timeNeeded = opt2(B, True)
    t1 = time.time()
    print(t1 - t0)
    paths.append(path)
    print(path)

plot(points,clusterSets,None,paths)

# VEHICLE_COUNT wird erhoeht von Mutterprogramm (abhaenglig von 2-opt)
# optimize "startAngleRange" in Clustering, as well as the cluster count.
# try Angles 90, 120, 180, 220, 260, 300, 340 etc. if 2-opt alg is fast enought