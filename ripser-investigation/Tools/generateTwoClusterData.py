# This file contains an program for creating lower diagonal distance
# matrices for two clusters of points normally distributed in 3 space
# at various distances from one another. The purpose is to explore how
# the statistics of the vietoris-rips barcodes change as the two
# cluters become intertwined.
import sys
import math
import numpy as np

# Just a function I use to write the lower distance matrix for a set
# of points.  This lower distance matrix does NOT include the
# diagonal.
def saveLowerDistanceMatrix(fileName, points):

    # Open the file
    file = open('../Output/TwoCluster'+fileName + '.lower_distance_matrix','w')
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
    file = open('../Output/' + fileName + ".dat",'w')
    print 'Save File: ',file.name
    for index in xrange(numPoints):
        file.write(repr(points[0,index])+' '+
                   repr(points[1,index])+' '+
                   repr(points[2,index])+'\n')
    file.close()

if __name__ == "__main__":

    for separation in range(0,17):
        cluster1 = np.random.normal(size=(3,300))
        cluster2 = np.random.normal(size=(3,300))
        cluster1[0,:] = cluster1[0,:] - separation/2
        cluster2[0,:] = cluster2[0,:] + separation/2
        points = np.concatenate((cluster1,cluster2),axis=1)
        fileName = 'TwoCluster-' + str(separation)
        print points.shape, fileName
        savePoints(fileName,points)
        saveLowerDistanceMatrix(fileName,points)

