# This is a set of classes that support generating points sampled from
# various 3D surfaces. Specifically, there is a base class
# "TesselatedShape" that creates surfaces tesselated by
# triangles. This is inherited by various other canonical shapes:
# spheres, cubes, torii, kein bottles, and also combinations of them
# put together.
import os
import sys
import random
import math
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# The base class is termed a tesselated shape, specifically tesselated
# with triangles. The structure is taken from MATLAB's patch structure
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

    # This method returns points uniformly randomly sampled along the
    # tesslated shape.  For each point, it randomly selects one of the
    # faces, and then it picks a point uniformly on that face. Because
    # these are triangles, the generation of a uniformly sampled point
    # is a little tricky.

    # What we do is for each triangle, we form an orthogonal pair of
    # axes in the plane of the triangle. These axes consist of the
    # longest side of the triangle and an orthognalized and scaled
    # copy of the second longest side. These two axes form two sides
    # of the rectangle that holds the triangle. We can then randomly
    # select points along each of those two axes and check if they are
    # in the triangle by computing the barycentric coordinates.

    # If occlusion is enabled, this routine will also remove a certain
    # percentage of the shape such that all the randomly removed
    # points lay on one side of a randomly oriented plane. This is
    # done by#
    #
    #   1) Randomly rotating the points sampled 
    #   2) Sorting the remaining points by z component
    #   3) Return the indicated number of points that have the highest Z value
    def getPoints(self,count,occlusion=0):

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

        # Now, if it's required, we implement occlusion. This is done
        # with a random rotation followed by a sorting in Z and
        # keeping the indicated percentage of points.

        # The randome rotation is executed by first generating a
        # random quarternion. This avoids biases inherent in using
        # random angles.
        if (occlusion != 0):
            # For those not familiar, a quarternion is a representation of
            # a rotation as an axis of rotation and an angle around that
            # axis. Generating a random direction is easily done by
            # choosing a random point on a sphere.
            temp = Sphere()
            direction, index = temp.getPoints(1)
            direction = direction/np.linalg.norm(direction,2)
            
            # NOTE: This is only from 0 to pi because the formulation of
            # the quarternion uses theta/2
            angle = np.pi * np.random.rand(1)
            angle.shape = (1,1)
            q = np.concatenate((np.sin(angle), 
                                np.cos(angle)*np.array(direction)))

            # Al this math taken from Wikipedia for computing the
            # rotation matrix associated with a quaternion.

            # Extract the values from Q
            q0 = q[0]
            q1 = q[1]
            q2 = q[2]
            q3 = q[3]
    
            # First row of the rotation matrix
            r00 = 2 * (q0 * q0 + q1 * q1) - 1
            r01 = 2 * (q1 * q2 - q0 * q3)
            r02 = 2 * (q1 * q3 + q0 * q2)
            
            # Second row of the rotation matrix
            r10 = 2 * (q1 * q2 + q0 * q3)
            r11 = 2 * (q0 * q0 + q2 * q2) - 1
            r12 = 2 * (q2 * q3 - q0 * q1)
            
            # Third row of the rotation matrix
            r20 = 2 * (q1 * q3 - q0 * q2)
            r21 = 2 * (q2 * q3 + q0 * q1)
            r22 = 2 * (q0 * q0 + q3 * q3) - 1
            
            # 3x3 rotation matrix
            rot_matrix = np.array([[r00, r01, r02],
                                   [r10, r11, r12],
                                   [r20, r21, r22]])
            rot_matrix.shape = (3,3)

            # Now rotate the points
            points = np.matmul(rot_matrix,points)

            # Now, let's sort by the Z axis
            indices = np.argsort(points[2,:])

            # We want to remove the first part of them (the ones with
            # the lowest Z values) as if they were occluded by a
            # plane.
            if occlusion  > 0:
                num_occluded = math.floor(occlusion * len(indices))
            else:
                min = abs(occlusion)
                max = 1 + occlusion
                temp = (max-min) * np.random.random(1) + min
                num_occluded = math.floor(temp * len(indices))

            indices = indices[num_occluded:]
            points = points[:,indices]
            faceIndices = faceIndices[indices]

        # Now form the rotation matrix
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

        if (abs(position) == 1):
            offset[1] = 0
            offset[2] = 0
        elif (abs(position) == 2):
            offset[0] = 0
            offset[2] = 0
        elif(abs(position) == 3):
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
    # can be fed to MATLABS surf function for display. This is because
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
        self.center()
        
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
    
# This generates the "figure 8" immersion of a Klien bottle in three
# dimensions. This math taken from wikipedia but augmented for
# different scalings.
#
# This embedding basically renders a torus except that the
# cross-section of the torus is a "figure 8", hence the name. It is
# parameterized by two angles: The angle around the Z axis, which runs
# through the hole in the torus, and the angle around the center of
# the tube. 
#
# The properties are:
# holeRadius = radius of the center hole of the torus
# tubeRadius = the radius of the tube of the torus
# 
class Figure8KleinBottle(TesselatedShape):
    def __init__(self,
                 name = "Figure8KleinBottle", 
                 holeRadius: float = 1.0,
                 tubeRadius: float = 1.0,
                 numTubeAngles: int = 32,
                 numWheelAngles: int = 32):

        X = np.zeros((numTubeAngles,numWheelAngles))
        Y = np.zeros((numTubeAngles,numWheelAngles))
        Z = np.zeros((numTubeAngles,numWheelAngles))

        u = 2 * np.pi * np.arange(0,1,1.0/numWheelAngles)
        v = 2 * np.pi * np.arange(0,1,1.0/numTubeAngles)

        # The input is in terms of hole and tube width. We need the
        # radius of the center of the tube
        centerRadius = holeRadius + tubeRadius
        for uIndex in range(len(u)):
            cu = np.cos(u[uIndex])
            su = np.sin(u[uIndex])
            cuo2 = np.cos(u[uIndex]/2)
            suo2 = np.sin(u[uIndex]/2)
            for vIndex in range(len(v)):
                sv = np.sin(v[vIndex])
                s2v = np.sin(2 * v[vIndex])
                
                X[uIndex,vIndex] = cu * (centerRadius +
                                         tubeRadius * (cuo2 * sv -
                                                        suo2 * s2v))
                
                Y[uIndex,vIndex] = su * (centerRadius +
                                         tubeRadius * (cuo2 * sv -
                                                        suo2 * s2v))
                Z[uIndex,vIndex] = tubeRadius * (suo2 * sv + cuo2*s2v)
        [faces, vertices] = tesselateMatrix(X,Y,Z);
        TesselatedShape.__init__(self,faces,vertices,name)
        self.center()

class StandardKleinBottle(TesselatedShape):
    def __init__(self,
                 name = "StandardKleinBottle", 
                 radius: float = 5.0,
                 num_points: int = 128):

        X = np.zeros((num_points,num_points))
        Y = np.zeros((num_points,num_points))
        Z = np.zeros((num_points,num_points))

        u =    np.pi * np.arange(0,1,1.0/num_points)
        v = 2* np.pi * np.arange(0,1,1.0/num_points)


        for uIndex in range(num_points):
            su = np.sin(u[uIndex])
            cu = np.cos(u[uIndex])
            for vIndex in range(num_points):
                sv = np.sin(v[vIndex])
                cv = np.cos(v[vIndex])
                X[uIndex,vIndex] = ((0 - 2 / 15) * cu *
                                    (3 * cv -
                                     30 * su +
                                     90 * (cu ** 4) * su - 
                                     60 * (cu ** 6) * su +
                                     5 * cu * cv * su))
                Y[uIndex,vIndex] = ((0 - 1 / 15) * su *
                                    (3 * cu -
                                     3 * (cu ** 2) * cv -
                                     48 * (cu ** 4) * cv +
                                     48 * (cu ** 6) * cv -
                                     60 * su +
                                     5 * cu * cv * su -
                                     5 * (cu ** 3) * cv * su - 
                                     80 * (cu ** 5) * cv * su +
                                    80 * (cu ** 7) * cv * su))
                Z[uIndex,vIndex] = ((2 / 15) * (3 + 5 * cu * su) * sv)
        X, Y, Z = radius * X, radius * Y, radius*Z
        [faces, vertices] = tesselateMatrix(X,Y,Z);
        TesselatedShape.__init__(self,faces,vertices,name)
        self.center()

# This function executes the generation of all the files for a given
# set of shapes. Put another way, it creates a training/evaluation
# training set. The inputs are:
#
# shapes - the list of shapes to sample from
# outputPath - where to put all the shape data files and labels file
# numPoints - the number of points to sample for each shape sample
# numSamples - how many total training samples to generate
# occlusion - what the occlusion is for each shape
#
# It generates the following data files
#
# ShapeX.dat - a file containing numPoints lines, each with X/Y/Z of point
# ShapeX.ldm - the lower diginal matrix: numPoints lines with 1 - numPoints-1
# ShapeX.ldmf - a flattened ldm
# ShapeX.bc - the barcodes file, each line with a birth/death pair and a dim
# ShapeX_Y.sli - sorted barcodes for dim Y
def generateShapeData(shapes,
                      outputPath,
                      numPoints=64,
                      numSamples=5000,
                      occlusion = 0):

    # Check if the output path exists
    if (not os.path.exists(outputPath)):
        print('Directory Does Not Exist ... Create:  ',outputPath)
        os.makedirs(outputPath)
        
    # The path to the ripser excecutable
    ripserPath = "../../ripser-investigation/ripser"

    # Set the output path and number of classes: number of classes
    # limited to the number above
    numClasses = len(shapes)

    # Now go through and center and scale them all and then also export them so
    # I can look at them in matlab.
    for index in range(len(shapes)):
        shapes[index].center()
        extents = (np.amax(shapes[index].vertices,1) -
                   np.amin(shapes[index].vertices,1))
        scale = 1/np.amax(extents)
        shapes[index].scale(scale)
        fileName = outputPath + "/" + shapes[index].name + repr(index) + '.m'
        shapes[index].export(fileName)

    # Open the label file: we will write the labels after each shape
    # so that the data files are there for each line.
    labelFile = open(outputPath + '/Labels.dat','w') 
    
    # For each shape type ...
    for count in range(numSamples):
        typeIndex = np.random.randint(len(shapes))
        shape = shapes[typeIndex]
        sys.stdout.flush()

        # .. by getting the points and writing them to the .dat file ...
        points, _ = shape.getPoints(numPoints,occlusion=occlusion)

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

        # Now run ripser on them
        inputFile = outputPath + "/Shape{}.ldm".format(count)
        outputFile = outputPath + "/Shape{}.bc".format(count)
        command = "rm -f " + outputFile
        command = command + "; " + ripserPath + "/ripser --dim 2 " + inputFile + "| /usr/bin/awk -f parseRipserFile.awk > " + outputFile

        os.system(command)

        # Now load the output of ripser and make the sli files
        inputFile = outputPath + "/Shape{}.bc".format(count)
        bc = np.loadtxt(inputFile).astype(float)
    
        persistence = bc[:,1] - bc[:,0]
        dims = bc[:,2].astype(int)
    
        for dim in np.unique(dims):
            indices = np.nonzero(dims == dim)
            indices = indices[0]

            i = np.argsort(persistence[indices])
            indices = indices[i]
            np.savetxt("{}/Shape{}_dim_{}.sli".format(outputPath,count,dim),
                       bc[indices,0:2])
    labelFile.close()
