# Generate the samples for the baseline data set. 
import os
import sys
import canonicalShapeGenerator as shape
import generateClamBarcodes as clam
import generateRipserBarcodes as ripser

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

# This is a list of the data sets and the parameters for each of them
dataSets = {'Baseline-Ripser' : {'numSamples'  : 500,
                                 'shapePath'   : '../Data/Baseline',
                                 'occlusion' : 0, 
                                 'barcodeSource' : ripser,
                                 'barcodePath' : 'Ripser',
                                 'numClasses': [2,8,16],
                                 'numPoints' : [32, 64, 128]},
            'Baseline-Clam': {'numSamples'  : 5000,
                              'shapePath'   : '../Data/Baseline', 
                              'occlusion' : 0, 
                              'barCodeSource' : clam,
                              'barcodePath' : 'Clam',
                              'numClasses': [2,4,8,12,16],
                              'numPoints' : [32, 48, 64, 96, 128, 192, 256]},
            'Occluded-Ripser': {'numSamples'  : 5000,
                              'shapePath'   : '../Data/Occluded', 
                              'occlusion' : 0.5, 
                              'barCodeSource' : ripser,
                              'barcodePath' : 'Ripser',
                              'numClasses': [2,4,8,12,16],
                              'numPoints' : [32, 48, 64, 96, 128, 192, 256]},
            'Occluded-Clam': {'numSamples'  : 5000,
                              'shapePath'   : '../Data/Occluded', 
                              'occlusion' : 0.5, 
                              'barCodeSource' : clam,
                              'barcodePath' : 'Clam',
                              'numClasses': [2,4,8,12,16],
                              'numPoints' : [32, 48, 64, 96, 128, 192, 256]},
            'Random-Ripser': {'numSamples'  : 5000,
                              'shapePath'   : '../Data/RandomOccluded', 
                              'occlusion' : -0.25, 
                              'barCodeSource' : ripser,
                              'barcodePath' : 'Ripser',
                              'numClasses': [2,4,8,12,16],
                              'numPoints' : [32, 48, 64, 96, 128, 192, 256]},
            'Random-Clam':   {'numSamples'  : 5000,
                              'shapePath'   : '../Data/RandomOccluded', 
                              'occlusion' : -0.25, 
                              'barCodeSource' : clam,
                              'barcodePath' : 'Clam',
                              'numClasses': [2,4,8,12,16],
                              'numPoints' : [32, 48, 64, 96, 128, 192, 256]},
            }
                      
runData = dataSets[sys.argv[1]]

for numClasses in runData['numClasses'] :
    for numPoints in runData['numPoints'] :
    
        shapePath = os.path.join(runData['shapePath'],
                                 str(numClasses) + 'Class' +
                                 str(numPoints) + 'Points')
        if (not os.path.exists(shapePath)):
            print('Create ',shapePath)
            os.makedirs(shapePath)
            print('Generate Shapes for ',shapePath)
            shape.generateShapeData(shapes[:numClasses],
                                    shapePath,
                                    numPoints,
                                    occlusion=runData['occlusion'],
                                    numSamples=runData['numSamples'])
        barcodePath = os.path.join(runData['shapePath'],
                                   str(numClasses) + 'Class' +
                                   str(numPoints) + 'Points',
                                   runData['barcodePath'])

        if (not os.path.exists(barcodePath)):
            print('Create barcode Directory: ',barcodePath);
            os.makedirs(barcodePath)
            print('Generate Barcodes: ',barcodePath)
            runData['barcodeSource'].generateBarcodes(shapePath,barcodePath)
