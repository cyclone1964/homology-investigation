# This program is designed to generate .dat, .ldm, and .ldmf files to
# test the application of persistent homology to the classification of
# 3-D objects.
#
# It generates an ever-growing list of different 3D objects such as:
#
# cylinder, sphere, torus, torus-with-munchkin, interlocking torus, cube, etc
#
# I am writing this in python so that it can be run by anybody but
# this is going to take a while since I am not familiar with
# shape-generation functions in python. Thus, I wrote my own, and it's
# implemented as a class structure. Specifically we have:
#
# TesselatedShape - a base class that takes lists of faces and
# vertices and provides bookeeping and methods for selecting sets of
# points uniformly places along the surface area of the shape. The
# vertices/faces paradigm was stolen from matlab: the vertices is a
# 3xN list of points in 3 space, each column being an [X, Y, Z]
# triplet for a point. The faces are 3xM lists of indices into those
# points, where each column is a list of 3 points in a triangle.
#
# Cylinder - a Tesselated Shape that parameterizes the surface as
# MATLAB's cylinder function does. It takes a vector of radii and
# generates a cylinder around the X axis with radius channgin in
# uniform steps. This is remarkabley useful
#
# Sphere - a Cylinder with radius equal to the cosine of the X
# coordinate to generate a Sphere
#
# Torus - a torus, generated parametrically
#
# Others to follow
#
# Usage
#
# python makeTrainingSet.py OutputPathForFiles \
#       dataPath <NumberOfClasses> <NumPoints> <NumSamples>
#

# Import a few things
import sys
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# To support future expansion, I design a few classes. The base class
# is termed a tesselated shape, specifically tesselated with
# triangles. The structure is taken from MATLAB's patch structure
# interface thing. The vertices are simply a 3xnumVertices matrix of
# points in 3-space, each column a new point. The faces are, in this
# case, 3xnumFaces array of column indices into the vertex
# matrix. Each column here defines a triangle of those three vertices.
#
# It has methods for translations, rotation, centering, scaling, and
# "exporting". Exporting means writing a matlab function that one can
# invoke to render the shape in 3D since python's 3D plotting is
# substandard.
class TesselatedShape:

    # The initializer, which stores the faces and vertices and then
    # sets up the bookkeping to support randomly getting points
    # uniformly spaced along the surface.
    def __init__(self, faces, vertices, name = "Shape"):

        # Do some error checking
        if (faces.shape[0] != 3 or vertices.shape[0] != 3):
            print("Bad Input Shape")
            
        # Store the input faces and vertices and name
        self.faces = faces
        self.vertices = vertices
        self.name = name
        
        # Now, let's compute the area of every face. We need to do
        # this so that we can randomly select faces weighted according
        # to their area later. We compute the area using the cross
        # product of two of the sides.
        areas = np.zeros(faces.shape[1])
        for faceIndex in range(faces.shape[1]):
            triangle = vertices[:,faces[:,faceIndex].astype(int)]
            area = 0.5 * \
                np.linalg.norm(np.cross(triangle[:,1] - triangle[:,0],
                                        triangle[:,2] - triangle[:,0]))
            areas[faceIndex] = area

        # Let's remove those that have no area: this might happen for
        # some shape generating functions. I am very frustrated that
        # numpy does not have a simple analog to matlabs find function
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

    # This method returns points randomly sampled along the
    # tesslated shape.  For each point, it randomly selects one of the
    # faces, and then it picks a point uniformly on that face. Because
    # these are triangles, the generation of a uniformly sampled point
    # is a little tricky.
    #
    # What we do is for each triangle, we form an orthogonal pair of
    # axes in the plane of the triangle. These axes consist of the
    # longest side of the triangle and an orthognalized and scaled
    # copy of the second longest side. These two axes form two sides
    # of the rectangle that holds the triangle. We can then randomly
    # select points along each of those two axes and check if they are
    # in the triangle by computing the barycentric coordinates.
    def getPoints(self,count):

        # Initialize the points and indices
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
            
            # Now do a graham schmidt orthoganilization of the
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

                # These are the barycentric coordinates
                u = np.dot(np.cross(sides[:,1],
                                    point-trianglePoints[:,1]),
                           normal)/area
                v = np.dot(np.cross(sides[:,2],
                                    point-trianglePoints[:,2]),
                           normal)/area

                w = 1-u-v

                # If they are all positive, we are in the triangle
                if (u >= 0 and v >= 0 and w >=0) :
                    break

            # Store this point and go to the next
            points[:,pointIndex] = point
            faceIndices[pointIndex] = faceIndex
        return [points, faceIndices]

    # Now, in forming shapes, it is useful to be able to do a few
    # things.  The first is to catenate one shape into another. This
    # means do the bookkeeping to return a new shape with all the vertices
    # and faces correctly generated. Note that it is entirely possible
    # that some of the points will be replicated.
    #
    # This function can just concatenate them, in which case it is the
    # user's responsiblity to deal with intersections etc. However, if
    # invoked with a "Position" operation, this function will
    # concatenate the input onto the current object such that the x,
    # y, or z coordinates are distinct. For example, if called with
    # position=1, it will add the input to the self so that the
    # minimum X of the input is the maximum X of self. Similarly with
    # Y, and Z, using 2 and 3. Conversely, it will put it on the other
    # side using -1, -2, or -3.
    #
    # IMPORTANT SAFETY TIP: this function breaks the selector, so I
    # would only call it in the constructor for a super-class.
    def concatenate(self, shape, position = 0):
        temp = self.vertices.shape[1]
        if (position > 0):
            offset = self.vertices.max(axis=1) - shape.vertices.min(axis=1)
        elif (position < 0):
            offset = self.vertices.min(axis=1) - shape.vertices.max(axis=1)
        else:
            offset = np.zeros((3,1))

        if (abs(position) is 1):
            offset[1] = 0
            offset[2] = 0
        elif (abs(position) is 2):
            offset[0] = 0
            offset[2] = 0
        elif(abs(position) is 3):
            offset[0] = 0
            offset[1] = 0

        offset.shape = (3,1)
        self.vertices = np.concatenate((self.vertices,
                                        shape.vertices+offset),axis=1)
        self.faces = np.concatenate((self.faces,shape.faces+temp),axis=1)

    # And this centers the object on the origin, that is to say, it
    # puts the origin exactly 1/2 way between the X, Y, and Z limits.
    def center(self):
        offset = (self.vertices.min(axis=1) + self.vertices.max(axis=1))/2
        offset.shape = (3,1)
        self.vertices = self.vertices - offset

    # And this scales it. WARNING: THIS IS UNTESTED
    def scale(self,scale):
        self.vertices = self.vertices * scale

    # And this rotates a shape. The rotation is defined as a triplet of
    # right handed angles around the X, Y, and Z axes, applied in reverse order.
    #
    # This function does not break the selector
    def rotate(self, angles):
        rotationMatrix = np.eye(3)
        temp = np.array([[np.cos(angles[2]),-np.sin(angles[2]),0],
                         [np.sin(angles[2]),np.cos(angles[2]),0],
                         [0,0,1]])
        rotationMatrix = np.matmul(rotationMatrix,temp)

        temp = np.array([[np.cos(angles[1]),0,np.sin(angles[1])],
                         [0,1,0],
                         [-np.sin(angles[1]),0,np.cos(angles[1])]])
        rotationMatrix = np.matmul(rotationMatrix,temp)

        temp = np.array([[1, 0, 0], 
                         [0, np.cos(angles[0]), -np.sin(angles[0])],
                         [0, np.sin(angles[0]),np.cos(angles[0])]])
        rotationMatrix = np.matmul(rotationMatrix,temp)
    
        self.vertices = np.matmul(rotationMatrix, self.vertices)

    # And this translates the shape
    def translate(self, offset):
        for index in range(len(offset)):
            self.vertices[index,:] = self.vertices[index,:] + offset[index]

    # This function exports the shape to a MATLAB .m file so that it
    # can be fed to MATLABS surf function for display. THis is because
    # python does not render 3D very well.
    def export(self,fileName):
        fileId = open(fileName,'w')
        fileId.write('Shape.vertices = [...\n')
        for index in range(self.vertices.shape[1]):
            fileId.write(repr(self.vertices[0,index]) + ' ' +
                         repr(self.vertices[1,index]) + ' ' +
                         repr(self.vertices[2,index]) + '\n')
        fileId.write("]';\n")
        fileId.write('Shape.faces = [...\n')
        for index in range(self.faces.shape[1]):
            fileId.write(repr(self.faces[0,index]+1) + ' ' +
                         repr(self.faces[1,index]+1) + ' ' +
                         repr(self.faces[2,index]+1) + '\n')
        fileId.write("]';\n")
        fileId.write("figure();\n")
        fileId.write("patch('Vertices',Shape.vertices','Faces',Shape.faces');\n")
        fileId.write("xlabel('X'); ylabel('Y'); zlabel('Z'); title('")
        fileId.write(self.name);
        fileId.write("'); axis equal;\n")
                     
        fileId.close()

# It is often useful to convert a tesselation of quadrilaterals into a
# tesselation of triangles. It is even more useful (for me) if
# the tesselation of quadrilaterals is represented as a matrix the way
# that MATLAB does it. This function does that triangularization.
def tesselateMatrix(X, Y, Z):

    # The number of faces is twice the number of quadrilaterals
    numFaces = 2 * (X.shape[0]-1)*(X.shape[1]-1)
    faces = np.zeros([3,numFaces])
    vertices = np.array(np.zeros((3,X.size)))

    # Now, for each consecutive pair of rows in the matrix (done
    # circularly so the last is contiguous to the first ...
    for index in range(X.shape[0]):

        # ... copy in the vertices, which are separated into three matrices
        vertexIndices = (index * X.shape[1] +
                         np.arange(X.shape[1])).astype(int)
        temp = np.concatenate((X[index,:],
                               Y[index,:],
                               Z[index,:]),0)
        temp.shape = (3,X.shape[1])

        for index2 in range(3):
            vertices[index2,vertexIndices] = temp[index2,:]

        # ... and now set up the triangulation. This can actually be
        # done in one of two ways, depending upon which way we choose
        # to triangulate the squares that the matrix representation
        # implies. We randomly choose one of them as it doesnt' really
        # matter.
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
# setting up a set of quadrilaterals then doing the triangularization.
#
# The cylinder is aligned with the Z axis and has a radius as defined
# by the inputs at each height along the cylinder.
#
# Inputs: radii - vector of radii at each of the z positions
# zpositions - the locations of the radii
class Cylinder(TesselatedShape):
    def __init__(self,
                 radii = 0.5 * np.ones((2,1)),
                 zpositions = np.array([-0.5, 0.5]),
                 numAngles=20,
                 name="Cylinder"):

        # The number of radii must match the number of z positions
        if (len(zpositions) != len(radii)):
            print("Bad Input Vectors")

        # Now, these are matrices of facets just like MATLAB's
        # cylinder returns. So we need the number of points along the
        # cylinder and a vector of the angles
        numRadii = radii.shape[0]
        angles = np.linspace(0,2*np.pi,numAngles)
        
        X = np.zeros((numRadii,numAngles))
        Y = np.zeros((numRadii,numAngles))
        Z = np.zeros((numRadii,numAngles))

        for index in range(numRadii):
            X[index,:] = radii[index] * np.cos(angles) 
            Y[index,:] = radii[index] * np.sin(angles) 
            Z[index,:] = zpositions[index]
        
        [faces, vertices] = tesselateMatrix(X,Y,Z)
        
        # Invoke the initializer for the super class (I think)
        TesselatedShape.__init__(self,faces,vertices,name)
        self.center()

# This uses the Cylinder function with a radius that is chosen to
# generate a sphere.
class Sphere(Cylinder):

    def __init__(self,radius=1,numPoints = 20, name="Sphere"):

        # For a sphere, the radius of the cylinder varies as the length
        # along it. So we set the radius as if we are drawing a circle
        # centered on a point 1/2 way along the line
        zpositions = radius * np.linspace(-1,1,numPoints)
        radii = np.sqrt(radius*radius - zpositions * zpositions)
        Cylinder.__init__(self,radii, zpositions, name = name)

# This creates a line of spheres all touching of the given count. The
# spheres form a line along the X axis.
class StringOfSpheres(TesselatedShape):

    def __init__(self,
                 count=1,
                 numPoints=20,
                 name = "StringOfSpheres"):

        # For this we simply make a sphere, and catenate it to itself
        # multiple times, then center it
        stringOfSpheres = Sphere()
        sphere = Sphere()
        for index in range(count-1):
            stringOfSpheres.concatenate(sphere,position = 1)

        TesselatedShape.__init__(self,
                                 stringOfSpheres.faces,
                                 stringOfSpheres.vertices,
                                 name)
        self.center()

# This makes a triangular layer of spheres, similar to a rack of pool balls
class RackOfSpheres(TesselatedShape):
    def __init__(self,
                 numRows = 2,
                 name="RackOfSpheres"):

        # Start with a single sphere ...
        rack = Sphere()

        # .. and for every row we need to ...
        for rowIndex in range(1,numRows):
            # ... translate the current row along the Y axis and add another row
            rack.translate([0, np.sqrt(3), 0])
            rack.concatenate(StringOfSpheres(count = rowIndex+1))

        # Now initialize ourselves and center
        TesselatedShape.__init__(self,rack.faces, rack.vertices,name)
        self.center()

# This makes a tower of spheres with 1, then 3, then 6, then 10 etc
class Sphyrimid(TesselatedShape):
    def __init__(self,
                 numLayers = 2,
                 name="Sphyrimid"):


        # Start with one sphere ... 
        sphyrimid = Sphere()

        # ... and for every layer below that one ...
        for layerIndex in range(1,numLayers):
            # ... move the sphyrimid up ...
            sphyrimid.translate([0,0,2*np.sqrt(2/3)])

            # .. and add a rack below it 
            rack = RackOfSpheres(numRows = layerIndex+1)
            sphyrimid.concatenate(rack)

        # Initialize and then center
        TesselatedShape.__init__(self, sphyrimid.faces, sphyrimid.vertices,name)
        self.center()
        
# And this makes a cube using brute force
class Cube(TesselatedShape):

    def __init__(self,sideLengths=[0.5,0.5,0.5],name="Cube"):

        # The vertices are simply the 8 points of the box
        x = sideLengths[0]/2
        y = sideLengths[1]/2
        z = sideLengths[2]/2
        vertices = np.array([[-x, x, -x, x, -x, x, -x, x],
                             [-y, -y, y, y, -y, -y, y, y],
                             [-z, -z, -z, -z, z, z, z, z]])

        # The faces I typed in myself
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
        TesselatedShape.__init__(self,np.transpose(faces),vertices,name)
        self.center()

# This makes nested spheres of specific radii
class NestedSpheres(TesselatedShape):

    # Default is two nested spheres
    def __init__(self, radii=[1., 2.],name="NestedSpheres"):

        for index in range(radii.size):
            sphere = Sphere(radii[index])
            if (index == 0):
                faces = sphere.faces
                vertices = sphere.vertices
            else:
                faces = numpy.concatenate((faces, sphere.faces))
                vertices = numpy.concatenate((vertices,
                                              sphere.vertices+vertices.shape[1]))
        TesselatedShape.__init__(self,faces, vertices,name)

# This makes a torus. This is done by generating points around a line
# that forms a circle and then tesselating those points.
#
# Inputs:
# innerRadius - the inner radius of the torus (radius of the donut hole)
# outerRadius - the outer radius of the torus (raius of the donut)
class Torus(TesselatedShape):

    def __init__(self,
                 name = "Torus", 
                 innerRadius = 0.25,
                 outerRadius = 0.5,
                 numTubeAngles =16,
                 numWheelAngles = 32):

        X = np.zeros((numTubeAngles, numWheelAngles))
        Y = np.zeros((numTubeAngles, numWheelAngles))
        Z = np.zeros((numTubeAngles, numWheelAngles))

        # This is better done in terms of the radius of the center of the
        # tube and the radius of the tube itself
        wheelRadius = 0.5 * (innerRadius + outerRadius)
        tubeRadius = 0.5 * (outerRadius - innerRadius)
        tubeAngles = np.linspace(0,2*np.pi,num=numTubeAngles)
        wheelAngles = np.linspace(0,2*np.pi,num=numWheelAngles)
        wheelRadii = wheelRadius + tubeRadius * np.cos(tubeAngles)

        for index1 in range(numTubeAngles):
            for index2 in range(numWheelAngles):
                X[index1,index2] = wheelRadii[index1]*np.cos(wheelAngles[index2])
                Y[index1,index2] = wheelRadii[index1]*np.sin(wheelAngles[index2])
                Z[index1,index2] = tubeRadius * np.sin(tubeAngles[index1])

        [faces, vertices] = tesselateMatrix(X,Y,Z)
        TesselatedShape.__init__(self,faces, vertices,name)
        self.center();
        
# This creates a line of tori, stacked along the Z axis, all touching
# of the given count.
class StringOfTori(TesselatedShape):

    def __init__(self,
                 count = 2,
                 innerRadius = 0.25,
                 outerRadius = 0.5,
                 name="StringOfTori"):

        # For this we simply make a sphere, and catenate it to itself
        # multiple times with a specific offset
        torus = Torus(innerRadius = innerRadius,
                      outerRadius = outerRadius)
        stringOfTori = Torus(innerRadius = innerRadius,
                             outerRadius = outerRadius)
        for index in range(count-1):
            stringOfTori.concatenate(torus,position = 3)
        stringOfTori.center()
        TesselatedShape.__init__(self,
                                 stringOfTori.faces,
                                 stringOfTori.vertices,
                                 name)
    
# This makes a torus with a sphere in the hole by using the
# sub-functions and proper rotations and translations.
class TorusSphere(TesselatedShape):
    def __init__(self,
                 innerRadius = 0.25,
                 outerRadius = 0.5,
                 name = "TorusSphere"):
        torus = Torus(innerRadius = innerRadius,
                      outerRadius = outerRadius)
        torus.concatenate(Sphere(radius=innerRadius))
        TesselatedShape.__init__(self,torus.faces, torus.vertices,name)

# This makes two torii such that they are interlocking, so the hole in
# one is the tube of the other.
class InterTorus(TesselatedShape):
    def __init__(self,
                 radius = 0.25,
                 name="InterTorus"):
        torus1 = Torus(innerRadius = radius,
                       outerRadius = 3*radius);
        torus2 = Torus(innerRadius = radius,
                       outerRadius = 3*radius)
        torus2.rotate([np.pi/2, 0, 0])
        torus2.translate([2*radius, 0, 0])
        torus2.concatenate(torus1)
        TesselatedShape.__init__(self,torus2.faces, torus2.vertices,name)

# This makes a cylinder with torusSpheres on either end.
class CylinderWithTorusSphereEnds(TesselatedShape):
    def __init__(self,
                 radius=0.5,
                 length=1,
                 name="CylinderWithTorusSphereEnds"):
        thisShape = Cylinder(radii=np.array([radius, radius]), 
                             zpositions = 0.5*np.array([-length, length]))
        torusSphere = TorusSphere(outerRadius = radius,
                                  innerRadius = radius/2);
        torusSphere.translate([0,0,length/2])
        thisShape.concatenate(torusSphere)
        torusSphere.translate([0,0,-length])
        thisShape.concatenate(torusSphere)
        TesselatedShape.__init__(self,thisShape.faces, thisShape.vertices,name)
    

# This is the main function, with usage as described in the header
if __name__ == "__main__":

    
    # The default list of shapes we will make: we do this first so we
    # know how many shapes we have available when parsing the command
    # line arguments below
    print('Building Shape Catalog ....');
    shapes = []
    shapes.append(Sphere())
    shapes.append(Torus())
    shapes.append(TorusSphere())
    shapes.append(InterTorus())
    shapes.append(StringOfSpheres(count = 2))
    shapes.append(StringOfSpheres(count = 3))
    shapes.append(StringOfSpheres(count = 4))
    shapes.append(StringOfTori(count = 2))
    shapes.append(StringOfTori(count = 3))
    shapes.append(StringOfTori(count = 4))
    shapes.append(RackOfSpheres(numRows = 2))
    shapes.append(RackOfSpheres(numRows = 3))
    shapes.append(RackOfSpheres(numRows = 4))
    shapes.append(Sphyrimid(numLayers = 2))
    shapes.append(Sphyrimid(numLayers = 3))
    shapes.append(Sphyrimid(numLayers = 4))
    shapes.append(CylinderWithTorusSphereEnds())

    # Parse the command line arguments
    print('Parse Arguments ...')
    if (len(sys.argv) < 2):
        print('Usage: ', sys.argv[0],' PathToOutput <NumberOfClasses> ')
        print('          <NumPoints> <NumSamples>')
        sys.exit()

    # Set the output path and number of classes: number of classes
    # limited to the number above
    outputPath = sys.argv[1]
    if (len(sys.argv) > 2):
        numClasses = int(sys.argv[2])
    else:
        numClasses = len(shapes)
    if (numClasses > len(shapes)):
        print('Warning; Number of classes limited to ',repr(len(shapes)))
        numClasses = len(shapes)

    # Set the number of points to sample them by
    if (len(sys.argv) > 3):
        numPoints = int(sys.argv[3])
    else:
        numPoints = 64

    # Set the number of samples to generate
    if (len(sys.argv) > 4):
        numSamples = int(sys.argv[4])
    else:
        numSamples = 5000

    # Now go through and center and scale them all and then also export them so
    # I can look at them in matlab.
    for index in range(len(shapes)):
        shapes[index].center()
        extents = (np.amax(shapes[index].vertices,1) -
                   np.amin(shapes[index].vertices,1))
        scale = 1/np.amax(extents)
        shapes[index].scale(scale)
        #fileName = shapes[index].name + repr(index) + '.m'
        #shapes[index].export(fileName)

    if (numClasses < 0 or numClasses > len(shapes)):
        print('Unsupported number of classes: ', numClasses)
        sys.exit()
    shapes = shapes[:numClasses]

    # Open the label file: we will write the labels after each shape
    # so that the data files are there for each line.
    labelFile = open(outputPath + '/Labels.dat','w') 
    
    # For each shape type ...
    for count in range(numSamples):
        typeIndex = np.random.randint(len(shapes))
        shape = shapes[typeIndex]
        print('Shape: ',count,' Type ',typeIndex)
        sys.stdout.flush()

        # .. by getting the points and writing them to the .dat file ...
        [points, Indices] = shape.getPoints(numPoints)
        shapeFile = open(outputPath + '/Shape' +
                         repr(count) + '.dat','w')
        for index in range(points.shape[1]):
            shapeFile.write(repr(points[0,index]) + ', ' +
                            repr(points[1,index]) + ', ' +
                            repr(points[2,index]) + '\n')
        shapeFile.close()

        # ... and then writing the LDM and LDMF files ...
        ldmFile = open(outputPath + '/Shape' + repr(count) + '.ldm','w')
        ldmfFile = open(outputPath + '/Shape' + repr(count) + '.ldmf','w')
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
