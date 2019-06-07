# This python program generates a series of files used to assess the
# numerical performance of the ripser algorithm. It generates two
# distributions: a gaussian distribution in 3 space, a uniform
# distribution in 3-space, and a uniform distribtion around a line.

# For sys and path function
import sys

# For sqrt
import math

# for var and mean
import numpy as np

# Just a function I use to write the lower distance matrix for a set
# of points.  This lower distance matrix does NOT include the
# diagonal.
def saveLowerDistanceMatrix(fileName, points):

    # Open the file
    file = open('../Output/Performance/'+fileName+'.lower_distance_matrix','w')
    print 'Save File: ',file.name

    # Now execute a double loop computing pairwise distances and
    # printing them out. The format could probably use some tweaking!
    size = points.shape
    numPoints = size[1]
    for firstIndex in xrange(1,numPoints):
        for secondIndex in xrange(firstIndex):
            range = np.linalg.norm(points[:,firstIndex] - points[:,secondIndex])
            file.write(repr(range) + ',')
        file.write('\n')
    file.close()

# The main function that generates all the files
if __name__ == "__main__":

    # The line and helix have 512 points in them
    numPoints = 256

    points = np.random.normal(size=(3,numPoints))
    saveLowerDistanceMatrix('Gaussian',points)

    points = np.random.uniform(low= -1,high = 1, size = (3,numPoints))
    saveLowerDistanceMatrix('Uniform',points)

    points = np.random.normal(size=(3,numPoints))
    points[1,:] = np.random.uniform(low = -100, high = 100, size=(1,numPoints))
    saveLowerDistanceMatrix('Line',points);
