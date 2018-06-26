import numpy as np
import copy
from Clustering import cluserting
from Opt2 import opt2
from Clustering import plot
np.set_printoptions(threshold=np.nan) # print big arrays completely
'''
    File name: solver.py
    Author: Sebastian Nichtern
    Date created: 12.04.2018
    Date last modified: 26.06.2018
    Python Version: 3.6.3
'''

########### Configure: ############
INSTANCE = "Testinstanzen/2_80.txt" # 10,20,30,50 or 80 cities/doctors
PLOT_EVERY_CLUSTER_CONFIG = False
MAX_TRAVEL_TIME_PER_DRIVER = 50000 # with times like 40000,50000,60000 our algorithm works good with this data. For lower times new algorithms are necessary
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

MAX_TRANSFER_TIME = MAX_TRAVEL_TIME_PER_DRIVER
print("MAX_TRANSFER_TIME set to", MAX_TRANSFER_TIME)

# Remove the Exchange points:
DRIVING_TIMES_SLICED = DRIVING_TIMES[0:-NUM_EXCHANGE, 0:-NUM_EXCHANGE]
nodes = np.shape(DRIVING_TIMES_SLICED)[0]

CLUSTER_COUNT = 2 # this changes later

def pathPointsToOriginalPoints(path, cluster):
    # Path Points returned from 2Opt back to the original Points:
    newPath = [0]*path.shape[0]
    for j, c in enumerate(cluster):
        for changePos, p in enumerate(path):
            if p != 0:
                if p == j:
                    newPath[changePos] = c
    return newPath

''' Minimaze total travel time: '''
optimalClusterSets = []
optimalPaths = []
optimalTotalTime = 10000000000
for CLUSTER_COUNT in range(2, 25):
    print("Travelers (Clusters): " + str(CLUSTER_COUNT))
    points, clusterSets = cluserting(DRIVING_TIMES_SLICED, CLUSTER_COUNT)

    paths = []
    times = []
    print("Search ideal path (2-opt + annealing)...")
    for clusterNr, cluster in enumerate(clusterSets):
        B = copy.deepcopy(DRIVING_TIMES_SLICED) # B = DRIVING_TIMES des aktuellen Clusters
        delete = [d for d in range(1,B.shape[0]) if d not in cluster]
        B = np.delete(B, (delete), axis=0)
        B = np.delete(B, (delete), axis=1)
        if(B.shape[0] == 1):
            print("Cluster with 0 Points!")
            plot(points, clusterSets)
            continue
        if cluster[0] != 0:
            cluster = [0] + cluster # add strating point to first position

        ''' RUN 2-Opt Algorithm'''
        path, timeNeeded = opt2(B, False) # False = Fast
        timeNeeded = timeNeeded - B[0,path[1]] # time needed - time to first doctor/city
        paths.append(pathPointsToOriginalPoints(path, cluster))
        times.append(timeNeeded)

    if PLOT_EVERY_CLUSTER_CONFIG:
        plot(points, clusterSets, None, paths) # Plotting!!
    overTime = False
    for i,time in enumerate(times):
        print("path",i,"time:",time,"max:",MAX_TRANSFER_TIME)
        if time > MAX_TRANSFER_TIME:
            overTime = True
    if overTime:
        print("OVER TIME")

    print("Total Time: "+str(sum(times)))
    if sum(times) < optimalTotalTime and not overTime:
        optimalClusterSets = clusterSets
        optimalPaths = paths
        optimalTotalTime = sum(times)



''' Final Path optimization: '''

pathsFinal = []
timesFinal = []
print("FINAL OPTIMIZATION")
for clusterNr, cluster in enumerate(optimalClusterSets):
    B = copy.deepcopy(DRIVING_TIMES_SLICED) # B = DRIVING_TIMES des aktuellen Clusters
    delete = [d for d in range(1,B.shape[0]) if d not in cluster]
    B = np.delete(B, (delete), axis=0)
    B = np.delete(B, (delete), axis=1)
    if cluster[0] != 0:
        cluster = [0] + cluster  # add strating point to first position

    path, timeNeeded = opt2(B, True) # True = slow but more detailed
    timeNeeded = timeNeeded - B[0,path[1]] # time needed - time to first doctor/city
    pathsFinal.append(pathPointsToOriginalPoints(path, cluster))
    timesFinal.append(timeNeeded)
print("Done.")


'''  PRINT AND PLOT RESULTS  '''

print("Optimal number of clusters:",optimalClusterSets.shape[0])
print("optimalClusterSets:\n",optimalClusterSets)
print("optimalPath (for each Cluster):\n",pathsFinal)
print("optimalTotalTime:",sum(timesFinal),"min")
plot(points,optimalClusterSets,None,pathsFinal)
