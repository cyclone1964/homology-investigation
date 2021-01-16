# Generate the samples for the occluded data set. This is the same as the
# baseline data set but with occlusion.

import canonicalShapeGenerator as shape

shapes = []
shapes.append(shape.Sphere())
shapes.append(shape.Torus())
shapes.append(shape.TorusSphere())
shapes.append(shape.InterTorus())
shapes.append(shape.StringOfSpheres(count = 2))
shapes.append(shape.StringOfSpheres(count = 3))
shapes.append(shape.StringOfSpheres(count = 4))
shapes.append(shape.StringOfTori(count = 2))
shapes.append(shape.StringOfTori(count = 3))
shapes.append(shape.StringOfTori(count = 4))
shapes.append(shape.RackOfSpheres(numRows = 2))
shapes.append(shape.RackOfSpheres(numRows = 3))
shapes.append(shape.RackOfSpheres(numRows = 4))
shapes.append(shape.Sphyrimid(numLayers = 2))
shapes.append(shape.Sphyrimid(numLayers = 3))
shapes.append(shape.Sphyrimid(numLayers = 4))


for numClasses in [2,4,8,12,16]:
    for numPoints in [16, 32, 48, 64, 96, 128, 192, 256]:
        outputPath = ('../Data/Occluded/' +
                      str(numClasses) + 'Class' +
                      str(numPoints) + 'Points')
        print('Populate ',outputPath)
        shape.generateShapeData(shapes[:numClasses],
                                outputPath,
                                numPoints, 
                                occlusion = 0.5)

        
