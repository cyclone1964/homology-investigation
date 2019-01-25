# This file contains an program for creating a "meandering" topology,
# which is a set of points constrained along a set of lines that split,
# join, vanish, and get modified.
#
# The way this is done is through independent "steps", each of which
# adds a fixed quantum to an existing endpoint. To start there is one
# endpoint, and it's direction is fixed to [1 0 0].
#
# At each step, the process is:
#
# Randomly select one of the existing endpoints
# 
# Randomly split that endpoint (probabilty of split is low) to create
# a new endpoint at the same place or randomly delete the endpoint by
# setting it's direction to empty.
#
# If the endpoint is not deleted, randomly perturb the direction of
# the current endpoint and adda length along the current direction to
# that endpoint, continuing the line.
#
# This continues for a fixed number of steps, so that at the end of
# the process, we have a fixed length of paths.
#
# Once we are done, we then randomly choose points from the paths and
# add clouds of points around them with random variances and random
# densities. At least that's the thought.
import sys
import math
import numpy as np

class Path:
    def __init__(self, direction=[1, 0, 0], stepSize = 1, origin = [0, 0, 0]):
        self.direction = np.array(direction)
        self.stepSize = stepSize
        self.points = [np.array(origin)]

# Just a function I use to write the lower distance matrix for a set
# of points.  This lower distance matrix does NOT include the
# diagonal.
def saveLowerDistanceMatrix(fileName, points):

    # Open the file
    file = open('../Output/Meander/'+fileName + '.lower_distance_matrix','w')
    print 'Save File: ',file.name

    # Now execute a double loop computing pairwise distances and
    # printing them out. The format could probably use some tweaking!
    size = points.shape
    numPoints = size[1]
    for firstIndex in xrange(1,numPoints):
        for secondIndex in xrange(firstIndex):
            range = np.linalg.norm(points[:,firstIndex] - points[:,secondIndex])
            file.write(repr(range) + ',')
    file.close()

# A function for saving points to a file so I can plot them in matlab
def savePoints(fileName, points):
    size = points.shape
    numPoints = size[1]
    file = open('../Output/Meander/' + fileName + ".dat",'w')
    print 'Save File: ',file.name
    for index in xrange(numPoints):
        file.write(repr(points[0,index])+' '+
                   repr(points[1,index])+' '+
                   repr(points[2,index])+'\n')
    file.close()

if __name__ == "__main__":

    # Set the number of steps
    numSteps = 200
    
    # probabilty of split and terminate
    randomDirection = 0.2
    splitProbability = 0.08
    terminateProbability = 0.01


    # First, let's create a single path and add it to the list of current paths
    # and then initialize an empty list of terminated paths
    currentPaths = [Path()]
    terminatedPaths = []

    for path in currentPaths:
        print 'Start: ', path.points, ':', path.direction
    
    # Now, for each step, do what we said in the header
    for stepIndex in range(numSteps):

        # Randomly pick one of the paths ...
        pathIndex = np.random.randint(len(currentPaths))

        # Is it to be split ?
        if (np.random.rand(1) < splitProbability):
            newPath = Path(direction=currentPath.direction, \
                           origin = currentPath.points[-1])
            print "Split Path: ", pathIndex, " @ ", \
                  newPath.points[-1], "<", newPath.direction,">"
            currentPaths.append(newPath)
            pathIndex = len(currentPaths) - 1
            continue;

        # Now, if we were doing this entirely mathematically correct, we
        # would define a random rotation here, and if we were doing that
        # really correct we would pick a uniformly random axis of rotation
        # normal to the current direction and then apply a gaussian
        # rotation around that axis. I'm way too lazy for that at this
        # point, so what we do instead is to add a guassian random value
        # to all three axes of the direciton and re-normalize it.
        currentPath = currentPaths[pathIndex]
        oldDirection = currentPath.direction
        currentPath.direction = currentPath.direction + \
                                randomDirection * np.random.randn(1,3)
        norm = np.linalg.norm(currentPath.direction)
        currentPath.direction = currentPath.direction/norm
        currentPath.points.append(currentPath.points[-1] +
                                  currentPath.direction * currentPath.stepSize)
        currentPaths[pathIndex] = currentPath

        # Is it to be terminated
        if (len(currentPaths) > 1 and np.random.rand(1) < terminateProbability):
            print "Terminate Path: ",pathIndex
            terminatedPaths.append(currentPaths[pathIndex])
            currentPaths.remove(currentPaths[pathIndex])
            continue

    # Now let's go through all the paths and get the points
    allPoints = np.ndarray((3,numSteps+1))

    pointIndex = 0
    for currentPath in currentPaths:
        for point in currentPath.points:
            allPoints[:,pointIndex] = point
            pointIndex = pointIndex+1
    for currentPath in terminatedPaths:
        for point in currentPath.points:
            allPoints[:,pointIndex] = point
            pointIndex = pointIndex+1

    savePoints('Path',np.array(allPoints))

    # Now, let's make clouds of points around that. The problem here
    # is that we need enough points to make the path make some sense
    # but not so many that ripser croaks. Ripser maxes out at about
    # 512 points on my machine and about 768 on the big server at URI.
    numPoints = 512

    randomPoints = np.random.randn(3,numPoints)
    for pointIndex in range(numPoints):
        pathIndex = np.random.randint(numSteps+1)
        randomPoints[:,pointIndex] = randomPoints[:,pointIndex] + allPoints[:,pathIndex]

    savePoints('ThickMeander',randomPoints)
    saveLowerDistanceMatrix('ThickMeander',randomPoints)

    # This one makes a thinner one
    numPoints = 512

    randomPoints = 0.2*np.random.randn(3,numPoints)
    for pointIndex in range(numPoints):
        pathIndex = np.random.randint(numSteps+1)
        randomPoints[:,pointIndex] = randomPoints[:,pointIndex] + allPoints[:,pathIndex]

    savePoints('ThinMeander',randomPoints)
    saveLowerDistanceMatrix('ThinMeander',randomPoints)
    
    
