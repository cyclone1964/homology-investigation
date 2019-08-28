# This program is designed to generate .dat and .ldm files to test the
# application of persistent homology to the classification of 3-D
# objects. Specifically, it generates two classes of objects:
# spheres and cubes
#
# I am writing this in python so that it can be run by anybody but
# this is going to take a while since A) I am not familiar with
# shape-generation functions in python and B) The python 3-D rendering
# software doesn't work on this MAC for some reason.

# Import a few things
import sys
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# TO support future expansion, I design a few classes. The base class
# is termed a tesselated shape, specifically tesselated with
# triangles. The structure is taken from MATLAB's patch structure
# interface thing. The vertices are simply a 3xnumVertices matrix of
# points in 3-space, each column a new point. The faces are, in this
# case, 3xnumFaces array of column indices into the vertex
# matrix. Each column here defines a triangle of those three vertices.
class TesselatedShape:

    # The initializer
    def __init__(self, faces, vertices):

        # Do some error checking
        if (faces.shape[0] != 3 or vertices.shape[0] != 3):
            print("Bad Input Shape")
            
        # Store the input faces and vertices.
        self.faces = faces
        self.vertices = vertices
       
        # Now, let's compute the area of every face. We need to do
        # this so that we can randomly select faces weighted according
        # to their area later. We compute the area using the cross
        # product of two of the sides.
        areas = np.zeros(faces.shape[1])
        for faceIndex in range(faces.shape[1]):
            triangle = vertices[:,faces[:,faceIndex].astype(int)]
            areas[faceIndex] = 0.5 * \
                np.linalg.norm(np.cross(triangle[:,1] - triangle[:,0],
                                        triangle[:,2] - triangle[:,0]))
        
        # Let's remove those that have no area: this might happen for
        # some shape generating functions. I am very frustrated that
        # numpy does not have a simple analog to matlabs find funcion
        # here.
        indices = np.asarray(areas > 0).nonzero()
        indices = indices[0]
        self.faces = faces[:,indices].astype(int)
        self.areas = areas[indices]

        # Again with the screwy re-dimensioning of things. I cannot
        # for the life of me figure out numpys dimensioning system.
        self.faces.shape = (3,indices.size)

        # Now, we want to form a vector with entries increasing
        # monotonicially from 0 to 1 where each consecutive pair is a
        # range of the percentage of the face. We call this the selector.
        selector = np.cumsum(self.areas)
        selector = np.concatenate(([0], selector))
        self.selector = selector/selector[-1]

    # This method returns points randomly sampled along the tesslated
    # shape.  For each point, it randomly selects one of the faces,
    # and then it picks a point uniformly on that face. Because these
    # are triangles, the generation of a uniformly sampled point is a
    # little tricky.
    #
    # What we do is for each triangle, we form an orthogonal pair of
    # axes in the plane of the triangle. These axes consist of the
    # longest side of the triangle and an orthognalized and scaled
    # copy of the second longest side. These two axes form two sides
    # of the rectangle that holds the triangle. we can then randomly
    # select points along each of those two axes and check if they are
    # in the triangle by computing the barycentric coordinates.
    def getPoints(self,count):

        # This method goes through and selects points from the tesselated shape
        points = np.zeros((3,count))
        faceIndices = np.zeros((count,1))
        for pointIndex in range(count):

            # Get a uniform random number between 0 and 1 and find the
            # face that corresponds to that number by searching for
            # the first face index whose area contribution pushes the
            # total beyond the random number.
            temp = np.random.rand(1)
            faceIndex = 0
            while (self.selector[faceIndex] < temp and
                   faceIndex <= self.selector.size):
                faceIndex = faceIndex + 1
            faceIndex = faceIndex - 1
            faceIndices[pointIndex] = faceIndex
            vertexIndices = self.faces[:,faceIndex]

            # Get the triangle points and form a matrix of the offsets
            # of each side.
            trianglePoints = self.vertices[:,vertexIndices]
            sides = np.array([trianglePoints[:,1]-trianglePoints[:,0],
                              trianglePoints[:,2]-trianglePoints[:,1],
                              trianglePoints[:,0]-trianglePoints[:,2]])
            sides = np.transpose(sides)
            
            # Now let's sort the sides by their lengths
            lengths = np.sqrt(sum(sides * sides,0))
            indices = np.argsort(-lengths)
            
            # We need to find the common point, or "origin", of these
            # two sides and the two axes from the sides. Note we need
            # to invert the sign on one of the sides
            if (indices[0] + indices[1] == 1):
                origin = trianglePoints[:,1]
                if (indices[0] == 0):
                    axis0 = -sides[:,0]
                    axis1 = sides[:,1]
                else:
                    axis0 = sides[:,1]
                    axis1 = -sides[:,0]
            elif (indices[0] + indices[1] == 2):
                origin = trianglePoints[:,0]
                if (indices[0] == 0):
                    axis0 = sides[:,0]
                    axis1 = -sides[:,2]
                else:
                    axis0 = -sides[:,2]
                    axis1 = sides[:,0]
            else:
                origin = trianglePoints[:,2]
                if (indices[0] ==2):
                    axis0 = -sides[:,1]
                    axis1 = sides[:,2]
                else:
                    axis0 = sides[:,2]
                    axis1 = -sides[:,1]
            
            # Now do a graham schmidt orthoganlization of the
            # second onto the first
            baseAxis = axis0
            normalAxis = axis1 - axis0 * np.dot(axis1,axis0)/np.dot(axis0,axis0)

            # Now we have to re-scale the second axis so that it spans
            # the area of the triangle but no more
            area = np.linalg.norm(np.cross(axis0,axis1))/2
            scale = 2*area/np.linalg.norm(axis0);
            normalAxis = normalAxis * scale/np.linalg.norm(normalAxis)

            # Now in the computations below we are computing
            # barycentric coordinates using math from the paper
            # https://people.cs.clemson.edu/~dhouse/courses/
            # 404/notes/barycentric.pdf but fixing a sign problem in
            # the computation of the normal
            normal = -np.cross(sides[:,1],sides[:,0])
            area = np.linalg.norm(normal)
            normal = normal/area
            temp1 = np.transpose(np.array([origin, origin+normal]))

            # Now generate random points and check if they are in the
            # triangle. On average half of them will be.
            while(True):
                point = origin + np.random.rand(1) * baseAxis
                point = point + np.random.rand(1) * normalAxis

                # Thesea are the barycentric coordinates
                u = np.dot(np.cross(sides[:,1],
                                    point-trianglePoints[:,1]),
                           normal)/area
                v = np.dot(np.cross(sides[:,2],
                                    point-trianglePoints[:,2]),
                           normal)/area

                w = 1-u-v

                # IF they are all positive, we are in the triangle
                if (u >= 0 and v >= 0 and w >=0) :
                    break

            # Store this point and go to the next
            points[:,pointIndex] = point
            faceIndices[pointIndex] = faceIndex
        return [points, faceIndices]

# It is often useful to convert a tesselation of quadrilaterals into a
# tesselation of triangles. It is even more useful (for me) if
# the tesselation of quadrilaterals is represented as a matrix the way
# that MATLAB does it. This function does that

def tesselateMatrix(X, Y, Z):

    # The number of faces is twice the number of quadrilaterals
    numFaces = 2 * (X.shape[0]-1)*(X.shape[1]-1)
    faces = np.zeros([3,numFaces])
    vertices = np.array(np.zeros((3,X.size)))

    for index in range(X.shape[0]):
        vertexIndices = (index * X.shape[1] +
                         np.arange(X.shape[1])).astype(int)
        temp = np.concatenate((X[index,:],
                               Y[index,:],
                               Z[index,:]),0);
        temp.shape = (3,X.shape[1])

        for index2 in range(3):
            vertices[index2,vertexIndices] = temp[index2,:]

        # Now, we need to set up the triangulation
        if (index < X.shape[0]-1):

            # These are the upper triangles ...
            indices = (2 * index * (X.shape[1]-1) +
                       np.arange(X.shape[1]-1)).astype(int)
            faces[0,indices] = vertexIndices[0:-1]
            faces[1,indices] = vertexIndices[1:]
            faces[2,indices] = vertexIndices[0:-1] + X.shape[1]
            
            # And these the lower
            indices = ((2 * index + 1) * (X.shape[1]-1) +
                       np.arange(X.shape[1]-1)).astype(int)
            faces[0,indices] = vertexIndices[1:]
            faces[1,indices] = vertexIndices[0:-1] + X.shape[1]
            faces[2,indices] = vertexIndices[1:] + X.shape[1]    
    return [faces, vertices]

# Now this class inherits from the one above, but implements a
# cylinder similarly to the way that MATLAB does.  It starts by
# setting up a set of quadrilaterals then doing the triangulization
class Cylinder(TesselatedShape):
    def __init__(self, radius = np.ones((20,1)),xpositions = [],numAngles=20):

        # Now, these are matrices of facets just like MATLAB's
        # cylinder returns. So we need the number of points along the
        # cylinder and a vector of the angles
        numRadii = radius.shape[0]
        angles = np.linspace(0,2*np.pi,numAngles)

        # Now, if no positions were supplied, lay them equally along
        # the X axis, centered on the origin
        if (len(xpositions) == 0):
            xpositions = np.linspace(-1,1,len(radius))

        X = np.zeros((numRadii,numAngles));
        Y = np.zeros((numRadii,numAngles));
        Z = np.zeros((numRadii,numAngles));
        for index in range(numRadii):

            X[index,:] = radius[index] * np.cos(angles)
            Y[index,:] = radius[index] * np.sin(angles)
            Z[index,:] = index/(numRadii-1)
        
        [faces, vertices] = tesselateMatrix(X,Y,Z)
        
        # Invoke the initializer for the super class (I think)
        TesselatedShape.__init__(self,faces,vertices)

# This uses the Cylinder function with a radius that follows a cosine shape
# to generate a sphere. 
class Sphere(Cylinder):

    def __init__(self,radius=1,numPoints=20):

        # For a sphere, the radius of the cylinder varies as the length
        # along it. So we set the radius as if we are drawing a circle
        # centered on a point 1/2 way along the
        xpositions = np.linspace(-1,1,numPoints)
        radius = np.sqrt(1 - xpositions * xpositions)
        
        Cylinder.__init__(self,radius,xpositions)

# And this makes a cube using brute force
class Cube(TesselatedShape):

    def __init__(self,sideLengths=[2.,2.,2.]):

        # The vertices are simply the 8 points of the box
        x = sideLengths[0]/2
        y = sideLengths[1]/2
        z = sideLengths[2]/2
        vertices = np.array([[-x, x, -x, x, -x, x, -x, x],
                             [-y, -y, y, y, -y, -y, y, y],
                             [-z, -z, -z, -z, z, z, z, z]])
        faces = np.array([[0,1,2],
                          [1,2,3],
                          [4,5,6],
                          [5,6,7],
                          [0,2,4],
                          [2,4,6],
                          [1,3,5],
                          [3,5,7],
                          [0,1,4],
                          [1,4,5],
                          [2,3,6],
                          [3,6,7]]).astype(int)
        TesselatedShape.__init__(self,np.transpose(faces),vertices)

# This makes two nested spheres
class NestedSpheres(TesselatedShape):

    # Default is two nested spheres
    def __init__(self, radii=[1., 2.]):

        for index in range(sideLengths.size):
            sphere = Sphere(radii[index])
            if (index == 0):
                faces = sphere.faces
                vertices = sphere.vertices
            else:
                faces = numpy.concatenate((faces, sphere.faces))
                vertices = numpy.concatenate((vertices,
                                              sphere.vertices+vertices.shape[1]))
        TesselatedShape.__init__(self,faces, vertices)

# This makes a torus with tube radius and donut radius defined
class Torus(TesselatedShape):

    def __init__(self,
                 circleRadius = 0.1,
                 revolutionRadius=0.9,
                 numCircleAngles =16,
                 numRevolutionAngles = 32):

        X = np.zeros((numCircleAngles, numRevolutionAngles))
        Y = np.zeros((numCircleAngles, numRevolutionAngles))
        Z = np.zeros((numCircleAngles, numRevolutionAngles))

        circleAngles = np.linspace(0,2*np.pi,num=numCircleAngles)
        revolutionAngles = np.linspace(0,2*np.pi,num=numRevolutionAngles)
        revolutionRadii = revolutionRadius + circleRadius * np.cos(circleAngles)

        for index1 in range(numCircleAngles):
            for index2 in range(numRevolutionAngles):
                X[index1,index2] = revolutionRadii[index1]*np.cos(revolutionAngles[index2])
                Y[index1,index2] = revolutionRadii[index1]*np.sin(revolutionAngles[index2])
                Z[index1,index2] = circleRadius * np.sin(circleAngles[index1])

        [faces, vertices] = tesselateMatrix(X,Y,Z)
        TesselatedShape.__init__(self,faces, vertices)

    
        
# This is the main 
if __name__ == "__main__":

    # We make 1000 totoal shapes, each and 64 points on each
    # shape. WE choose 64 so that ripser can run well.
    numPoints = 64
    numShapes = 5000

    shapes = []
    shapes.append(Sphere())
    shapes.append(Cube())
    shapes.append(Torus())

    if (sys.argc > 1):
        numClasses = int(sys.argv[1])
        if (numClasses < 0 or numClasses > shapes.size):
            print('Unsupported number of classes: ', numClasses)
            sys.exit()
        shapes = shapes[:numClasses]

    # Open the label file
    outputDirectory = '../Output/MachineLearning/ThreeClass'
    labelFile = open(outputDirectory + '/Labels.dat','w') 
    
    # For each shape type ...
    for count in range(numShapes):
        typeIndex = np.random.randint(len(shapes))
        shape = shapes[typeIndex]
        print('Shape: ',count,' Type ',typeIndex)
        sys.stdout.flush()

        # .. by getting the points and writing them to the .dat file ...
        [points, Indices] = shape.getPoints(numPoints);
        shapeFile = open(outputDirectory + '/Shape' +
                         repr(count) + '.dat','w')
        for index in range(points.shape[1]):
            shapeFile.write(repr(points[0,index]) + ', ' +
                            repr(points[1,index]) + ', ' +
                            repr(points[2,index]) + '\n');
        shapeFile.close()

        # ... and then writing the LDM file ...
        ldmFile = open(outputDirectory + '/Shape' + repr(count) + '.ldm','w')
        ldmfFile = open(outputDirectory + '/Shape' + repr(count) + '.ldmf','w')
        for firstIndex in range(points.shape[1]):
            for secondIndex in range(firstIndex):
                distance = np.linalg.norm(points[:,firstIndex] -
                                          points[:,secondIndex])
                ldmFile.write(repr(distance) + ',')
                ldmfFile.write(repr(distance) + '\n')
                ldmFile.write('\n')
        ldmFile.close()
        ldmfFile.close()
            
        # Write the label
        labelFile.write(repr(typeIndex)+'\n')
    labelFile.close()
