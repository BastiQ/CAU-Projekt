# Notes:
## Underestimated minimum distance ist the sum of the minimum from all rows in the DRIVING_TIMES Matrix
## Best algorithm would be 2-Opt + Simulated annealing
# Wenn alter Merge pull verhindert:  git merge --abort  oder  git reset --merge
import numpy as np
import copy
import time
from Clustering import cluserting
from Opt2 import opt2
from Clustering import plot
np.set_printoptions(threshold=np.nan) # completely print big arrays

# Configure:
INSTANCE = "Testinstanzen/3_30.txt"
VEHICLE_COUNT = 2
###################################

NUM_EXCHANGE = 0
NUM_DOCS = 0
MAX_TRANSFER_TIME = 0
MAX_TIME = 0
DRIVING_TIMES = []
points = []

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
DRIVING_TIMES_SLICED = DRIVING_TIMES[0:-NUM_EXCHANGE, 0:-NUM_EXCHANGE]
nodes = np.shape(DRIVING_TIMES_SLICED)[0]


CLUSTER_COUNT = VEHICLE_COUNT # this changes later

''' Minimaze total Travel Time: '''
optimalClusterSets = []
optimalPaths = []
optimalTotalTime = 10000000000
for CLUSTER_COUNT in range(2, 6):
    print("Clusters: " + str(CLUSTER_COUNT))
    points, clusterSets = cluserting(DRIVING_TIMES_SLICED, CLUSTER_COUNT)

    paths = []
    times = []
    for clusterNr, cluster in enumerate(clusterSets):
        B = copy.deepcopy(DRIVING_TIMES_SLICED) # B = DRIVING_TIMES des aktuellen Clusters
        delete = [d for d in range(1,B.shape[0]) if d not in cluster]
        B = np.delete(B, (delete), axis=0)
        B = np.delete(B, (delete), axis=1)

        print("Search ideal path (2-opt + annealing)...")
        path, timeNeeded = opt2(B, True) # True = Fast

        print("Done.")
        # Path Points to original Points:
        if cluster[0] != 0:
            cluster = [0] + cluster
        newPath = [0]*path.shape[0]
        for i, c in enumerate(cluster):
            for changePos, p in enumerate(path):
                if p != 0:
                    if p == i:
                        newPath[changePos] = c
        paths.append(newPath)
        times.append(timeNeeded)

    plot(points, clusterSets, None, paths)
    print("Total Time: "+str(sum(times)))
    if sum(times) < optimalTotalTime:
        optimalClusterSets = clusterSets
        optimalPaths = paths
        optimalTotalTime = sum(times)
    # TODO: check if time one traveler is on its way extends the allowed time (time counts not from the start! But from the first doc!)

plot(points,optimalClusterSets,None,optimalPaths)