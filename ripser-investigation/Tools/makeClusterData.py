# This python program generates a series of files used to assess the
# structure of bar codes as a function of "optimal cluster
# width". This is done with a series of tests
#
# The first is a set of points normally distributed about points
# uniformly distrubuted on a line.
#
# The second is a set of points normally distributed about a set of
# points uniformly distributed on a helix that runs around the x
# axis. The Helix has radius 5
#
# The third through sixth are a set of 9 balls separated by 1, 4, 16,
# and 64. These balls only have 32 points per ball.
#
# This file saves two files for each test. One is the
# lower_distance_matrix, which is stored in the Data directory
# ../Output. The other, stored locally, is a ".dat" file that can be
# read into matlab for plotting purposes. I cannot for the life of me
# get python to plot in 3 dimensions.

# For sys and path function
import sys

# For sqrt
import math

# for var and mean
import numpy as np

# And the plotting package
import matplotlib.pyplot as plt

# This class defines methods for finding points and normals along a helix
# The formula for this line in 3 space is:
#
# x = l
# y = radius * cos(rate * l)
# z = radius * sin(rate * l)
#
# Thus the tangent to the path at any point is given by the gradient
# of that function, which is converted to unit vector by dividing by
#
# sqrt(1 + 2 * (rate*radius)**2
#
# x = 1
# y = -rate * radius * sin(rate * l)
# z = rate * radius * cos(rate * l)
#
# And a normal is given by differentiating that yet again
#
# x = 0
# y = -cos(rate * l)
# z = -sin(rate * l)
#
# and the other normal is given by the cross product of those two
class HelixLine:
    def __init__(self,radius,rate):
        self.radius = radius
        self.rate = rate

    # Returns a 3 element column vector that is the [x,y,z] of the point
    def getPoint(self,l):
        point = [l,
                  self.radius*math.cos(self.rate * l),
                  self.radius*math.sin(self.rate * l)];
        point = np.transpose(np.matrix(point))
        return point

    # Returnss a 3x3 matrix, with columns that are the tangent, normal, binormal
    def getNormals(self,l):
        tangent = np.array([1,
                            -self.rate * self.radius * math.sin(self.rate * l),
                            self.rate * self.radius * math.cos(self.rate * l)])
        tangent = tangent / math.sqrt(1 + self.rate**2 * self.radius**2)
        normal = np.array([0,
                            -math.cos(self.rate * l),
                            -math.sin(self.rate * l)])
        binormal = np.cross(tangent,normal)
        normals = np.matrix([tangent[:],normal[:],binormal[:]]);
        return normals

# Just a function I use to write the lower distance matrix for a set
# of points.  This lower distance matrix does NOT include the
# diagonal.
def saveLowerDistanceMatrix(fileName, points):

    # Open the file
    file = open('../Output/'+fileName + '.lower_distance_matrix','w')
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
    file = open(fileName + ".dat",'w')
    print 'Save File: ',file.name
    for index in xrange(numPoints):
        file.write(repr(points[0,index])+' '+
                   repr(points[1,index])+' '+
                   repr(points[2,index])+'\n')
    file.close()

    
# The main function that generates all the files
if __name__ == "__main__":

    # The line and helix have 512 points in them
    numPoints = 512

    # Now lets make a line, which is to say a helix with 0 radius and
    # 0 rate. This we do by picking random points along the length
    # then setting random offsets around the line from those points.
    helix = HelixLine(0,0);

    # Uniformly distributed on the line. Note that for reasons I don't
    # get, xvalues is a list of a single list entry (rather than a
    # matrix), so in order to use it below we convert it to a vector.
    xvalues = np.random.uniform(low = -5,high = 5,size = (1,numPoints))
    xvalues = xvalues[0]

    # Initialize the points
    points = np.zeros((3, numPoints))
    for index in xrange(numPoints):

        # Extract teh x value ...
        x = xvalues[index]

        # now, compute the offset from the point by taking the dot product of a
        # gaussian random with each of the normals.
        offset = np.dot(helix.getNormals(x),
                        np.random.uniform(-1,1,size=(3,1)))

        # Get the point as the offset form the point on the helix
        point = helix.getPoint(x) + offset

        # And assign: I could not get this to work using a single line!!
        for i in range(len(offset)):
            points[i,index] = point[i]

    # Now save the points and the distance matrix
    savePoints('Line',points)
    saveLowerDistanceMatrix('Line',points)

    
    # Now lets make an actual helix, which will have a "tube" of
    # radius 1 around a helix of radius 5 that corkscrews 
    helix = HelixLine(5, 2*math.pi*5/20);

    # Again the the values and intialization
    xvalues = np.random.uniform(low = -5,high = 5,size = (1,numPoints))
    xvalues = xvalues[0]
    points = np.zeros((3, numPoints))

    # Now make points like we did before
    for index in xrange(numPoints):
        x = xvalues[index]
        offset = np.dot(helix.getNormals(x),
                        np.random.uniform(-1,1,size=(3,1)))
        point = helix.getPoint(x) + offset
        for i in range(len(point)):
            points[i,index] = point[i]

    # And save them to the correctly named file
    Base = 'Helix-'+str(helix.radius)
    savePoints(Base,points)
    saveLowerDistanceMatrix(Base,points)

    # Now we make a series of "ball clusters". Each is a set of 27
    # balls normally distributed in 3 dimensions. The centers of the
    # balls are separated by 1, 2, 4, and 8 stddev
    numPointsPerBall = 32
    for ballSeparationIndex in range(4):
        separation = 4 ** ballSeparationIndex
        print 'Separation: ',separation
        points = np.zeros((3,3*3*3*numPointsPerBall))

        # Now, in order to do the assignment, I make a vector of destination
        # indices into the points matrix
        indices = np.arange(numPointsPerBall)

        # Now 3 in each dimension
        for xIndex in range(-1,2):
            for yIndex in range(-1,2):
                for zIndex in range(-1,2):
                    temp = np.random.normal(size=(3,numPointsPerBall))
                    points[0,indices] = temp[0,:] + xIndex * separation
                    points[1,indices] = temp[1,:] + yIndex * separation
                    points[2,indices] = temp[2,:] + zIndex * separation
                    indices = indices + numPointsPerBall

        # Save to the right file.
        fileName = 'Balls-'+str(separation)
        savePoints(fileName,points)
        saveLowerDistanceMatrix(fileName,points)
