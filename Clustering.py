# Some Infos and Ideas:
# The kartisian space calculated from the drivingTime Matrix is n-1 dimensional.
# I have tryed to write my own algorithems for getting the Problem down to 2 Dimensions: (__distanceMatrixToPoints_onlyFirstThree and __distanceMatrixToPoints_midEuclid)
# The mid Euclid dose not work really well (maybe because the first points set in 2d are completely wrong, or an error in the code (i havend completely checked its correctness))
# The onlyFirstThree Alg. works suprisingly better, although it only considered the distances to the first three points
# I found the Problem under the Name "Distance geometry problem with inaccurate distances":
# Maybe the Alg. "Solving the molecular distance geometry problem with inaccurate distance data" is better, but it looks pretty complicated: https://bit.ly/2H9bMGX (minimize Function with SciPy: https://bit.ly/2qHJ58P)
# Other possible techniques: Hierarchical method and  SOM and Gas NN (= Neural gas) (Algs. from https://bit.ly/2HKDS8j)
import numpy as np
import copy
import random
from sklearn.cluster import spectral_clustering
import matplotlib.pyplot as plt

########################################   DIMENSION REDUCTION   ########################################

def __distanceMatrixToPointsIfExact(A):
    # https://math.stackexchange.com/questions/156161/finding-the-coordinates-of-points-from-distance-matrix
    # this dose not work in this case because the matrix consists of Travletimes, not distances
    M = (np.square(A[0, :]) + np.square(np.transpose([A[:,0]])) - np.square(A)) / 2
    u, s, v = np.linalg.svd(M) # Singulärwertezerlegung
    rank = np.sum(s > 1e-10) # Alternative: np.linalg.matrix_rank(M).  1e-10 = Threshold where it counts as a zero
    # rank2 =
    points = []
    for i, singu in enumerate(s):
        if singu > 1e-10:
            points.append((singu*v[i]))
    points = np.array(points)
    return points

def __distanceMatrixToPoints_onlyFirstThree(A, farestPoint=False):
    # Using 2 initial Points and locate them in space, calculate all other point locations using there distances
    points = np.zeros((A.shape[1],2))
    # first fixed point is 0,0 (it's the Lab)
    # Second point as far away as possible from the first one: (https://bit.ly/2qHwZg4)
    secondPoint = np.argmax(A[0,1:])+1 # exclude first zero for argmin # argmax for index (armax for value)
    if farestPoint:
        points[1] = [0, A[0, secondPoint]] # second fixed point at (x=0, y=distance to doc whos farest away)
    else:
        secondPoint = 1
        points[1] = [0, A[0, 1]] # second fixed point at (x=0, y=distance to doc whos farest away)
    # print(secondPoint)
    for p in range(2,len(points)): # from third point to last point
        backup_p = p
        if p == secondPoint: # this Point has allready been used above
            p = 1
        d13 = A[0,p] # point 3 changes each loop -> used p instead of 3
        d23 = A[1,p]
        x1 = 0
        y1 = 0
        x2 = 0 # actually not needed
        y2 = A[0,1] # to the second Point
        # Calculate y3 and x3 (formulars generated with wolfram Alpha):
        y3 = np.divide((-np.square(d13)+np.square(d23)+np.square(y1)-np.square(y2)), 2*(y1-y2))
        x3 = x1+np.sqrt(-np.square(y3)+np.square(d13)+2*y3*y1-np.square(y1))

        # decide if x3 is negative or positive. It depends on the best Distance (+ or -) to the Distence in A:
        if p > 2: # only now the third point is defined
            dPositive = np.sqrt(np.square(x3 - points[2][0]) + np.square(y3 - points[2][1]))
            dNegative = np.sqrt(np.square(-x3 - points[2][0]) + np.square(y3 - points[2][1]))
            dPositive_DiffToA = np.abs(dPositive - A[3,p])
            dNagative_DiffToA = np.abs(dNegative - A[3,p])
            if dNagative_DiffToA < dPositive_DiffToA:
                x3 = -x3
        # and save the calculated location:
        if backup_p == secondPoint:
            p = secondPoint # put the point 2 on the place of the chosen secondPoint
        points[p] = [x3,y3]
    return points

def __distanceMatrixToPoints_midEuclid(A):
    # Places point after point in the 2D space at the location with mid best fitting Range to all places points
    # can use random points and optimization using the __pointsToDistanceMatrix() and __matrixDifference() functions

    # Problem mit diesem Alg: Punkte werden stueck fuer stueck hinzugefuegt, nur Abstaende zu bereits
    # gesetzten Punkten werden somit beruecksichtigt. Neu gesetzte Punkte beruecksichtigen dann zwar
    # die position der alten, wenn diese aber schon voellig falsch liegen nuetzt das auch nichts mehr

    points = np.zeros((A.shape[1], 2))
    order = list(range(1,A.shape[1]))
    random.shuffle(order) # random order in witch points are added
    points[1] = [0, A[0,order[-1]]] # first point (Lab) was (0,0), second point now placed

    for lastAddedPoint, newPoint in enumerate(order[1:]):
        # lastAddesPoint+2 is needed for the number of elements in points that are used for the approximation
        # 1. Suchraum bestimmen (Abstand * 1.5 von allen Punkten aus)
        xSearchSpace = [0,0] # min and max x
        ySearchSpace = [0,0] # min and max y
        ENLARGING_FACTOR = 1.5
        for oldPoint in range(lastAddedPoint+2):
            # distance to toPoint from element * 1,5
            xMin = (points[oldPoint][0] - A[newPoint,oldPoint]) * ENLARGING_FACTOR
            xMax = (points[oldPoint][0] + A[newPoint,oldPoint]) * ENLARGING_FACTOR
            yMin = (points[oldPoint][1] - A[newPoint,oldPoint]) * ENLARGING_FACTOR
            yMax = (points[oldPoint][1] + A[newPoint,oldPoint]) * ENLARGING_FACTOR
            if xSearchSpace[0] >= xMin:
                xSearchSpace[0] = xMin
            if xSearchSpace[1] <= xMax:
                xSearchSpace[1] = xMax
            if ySearchSpace[0] >= yMin:
                ySearchSpace[0] = yMin
            if ySearchSpace[1] <= yMax:
                ySearchSpace[1] = yMax

        # 2. Im suchraum minimum der Funktion (Abstaende zu allen nachbarn) Finden
        # Entweder brute force oder monte carlo oder Alternativer ansatz:
        # Diskretisierung in 100ter schritte und dort wo die besten ergebnisse gefunden
        # wurden anschließend nochmal mit kleineren schritten (z.b. 1ner) suchen

        # Discreted Brute Force (STEPSIZE):
        STEPSIZE = 200 # higher = faster
        bestPosition = [xSearchSpace[0], ySearchSpace[0]]
        lowermostDistanceSum = 10000000000
        for x in np.arange(xSearchSpace[0],xSearchSpace[1],STEPSIZE):
            for y in np.arange(ySearchSpace[0],ySearchSpace[1],STEPSIZE):
                distances = [] # distances to all already set points
                for oldPoint in range(lastAddedPoint + 2):
                    d = np.sqrt(np.square(x - points[oldPoint][0]) + np.square(y - points[oldPoint][1]))
                    if d < A[newPoint,oldPoint]: # if the distance is gone below the min distance from A
                        d += 1.5*(A[newPoint,oldPoint] - d) # then add the overstepped part to the distance. -> pseudodistance
                        # this way, going nearer to points dose not count as better solutions
                    distances.append(d)

                if sum(distances) < lowermostDistanceSum:
                    bestPosition = [x,y]
                    lowermostDistanceSum = sum(distances)
        print(str(lastAddedPoint+2) + ": " + str(bestPosition))
        points[lastAddedPoint+2] = bestPosition

        # Alternativer alg (Einfacher und ungenauer):
        # 2 zufaellige nachbarn auswaehlen und von diesen über den Abstand mit der formel aus
        # __distanceMatrixToPoints_onlyFirstTwo die Position bestimmen.

    return points

def __pointsToDistanceMatrix(p):
    dist = np.zeros((p.shape[0],p.shape[0])).astype(np.int)
    for i in range(dist.shape[0]):
        for j in range(dist.shape[1]):
            dist[i,j] = np.round(np.sqrt(np.square(p[j,0]-p[i,0])+np.square(p[j,1]-p[i,1])),0).astype(np.int)
    return dist

def __matrixDifference(A,B):
    # Only for 2D arrays!
    # Hint this module: works only if the points have the same order in both matreces
    err = 0
    if(A.shape == B.shape):
        for i in range(A.shape[0]):
            for j in range(A.shape[1]):
                err += np.abs(A[i,j]-B[i,j])
    else:
        print("A and B must have the same shape (in __matrixDifference(A,B))")
    return err

def __plot2D(A):
    points1 = __distanceMatrixToPoints_onlyFirstThree(A, False)        # n-1 Dimensions to 2D (err(10) = 232202/297272, err(30) = 3483812/3610502)
    points2 = __distanceMatrixToPoints_onlyFirstThree(A, True)        # n-1 Dimensions to 2D (err(10) = 232202/297272, err(30) = 3483812/3610502)
    # points = __distanceMatrixToPoints_midEuclid(A)  # err: 469522/613622/399714 -> this is bad

    D = __pointsToDistanceMatrix(points1)  # Distances in the model
    err = __matrixDifference(A, D)  # copare original distances and projected distances in 2D
    # print("err1: " + str(err))
    D      = __pointsToDistanceMatrix(points2)   # Distances in the model
    err    = __matrixDifference(A,D)            # copare original distances and projected distances in 2D
    # print("err2 (p2=farest away): "+str(err))

    plt.scatter(points2[:,0],points2[:,1], color="brown") # point 2 is farest away
    plt.scatter(points1[:,0],points1[:,1])

    # plt.scatter(points1[1, 0], points1[1, 1], color="green") # second point
    # plt.scatter(points2[1, 0], points2[1, 1], color="black") # second point

    plt.scatter(0,0,color="red")
    plt.annotate("Lab",(0,0))
    plt.show()

########################################   CLUSTERING   ########################################

def use_spectral_clustering(A, CLUSTER_COUNT):
    # numParts = int(round(np.shape(A)[0] / CLUSTER_COUNT,0))
    labels = spectral_clustering(A, CLUSTER_COUNT) # Produces a different result every time
    return labels

def labelsToSets(labels, CLUSTER_COUNT):
    clusterSets = [[] for _ in range(CLUSTER_COUNT)]
    for i, label in enumerate(labels):
        clusterSets[label].append(i)
    # DONT wunder about the "array" keword when you print out the return. Its because the Elements dont have the same size!!!
    return np.array([np.array(cluster) for cluster in clusterSets])

def approximateTimeNeededWithKNN(A, cluster):
    placesLeft = copy.deepcopy(list(cluster))
    placesLeft= np.trim_zeros(placesLeft) # if the start node is in this cluster, remove it
    path = [0] # first node is the Laboratory
    distanceSum = 0
    for _ in range(len(placesLeft)):
        lowestDist = 1000000000
        nearestNeighbor = 0
        for neighbor in placesLeft:
            dist = A[path[-1], neighbor]
            if dist != 0:
                if dist < lowestDist:
                    lowestDist = dist
                    nearestNeighbor = neighbor
        distanceSum += lowestDist
        path.append(nearestNeighbor)
        placesLeft = np.delete(placesLeft, np.argwhere(placesLeft == nearestNeighbor))
    # go back to the start:
    distanceSum += A[path[-1], 0]
    path.append(0)
    # print("path: "+str(path))
    return distanceSum

def cluserting(DRIVING_TIMES, CLUSTER_COUNT):
    __plot2D(DRIVING_TIMES)
    CLUSTER_COUNT = 4 # for testing
    # labels = use_spectral_clustering(DRIVING_TIMES.astype(np.float64), CLUSTER_COUNT)
        # How many points are in each cluster:
        # unique, counts = np.unique(labels, return_counts=True)
        # print(dict(zip(unique, counts)))
    # clusterSets = labelsToSets(labels, CLUSTER_COUNT)
    # for cluster in clusterSets:
    #     # print("__________________________________")
    #     # print("Cluster: "+str(cluster))
    #     time = approximateTimeNeededWithKNN(DRIVING_TIMES, cluster)
    #     # print(" time: "+str(time))




# Other Alorithems (don't fit this problem):
# 1. Random Projection to reduce Dimensions is only for measured data, with features (Variables). -> (Training set, target Values) (X : numpy array of shape [n_samples, n_features], y : numpy array of shape [n_samples])
# Use of that Alg.: http://scikit-learn.org/stable/modules/random_projection.html
