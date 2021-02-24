import os
import sys
import numpy as np

def generateBarcodes(inputPath,
                     outputPath):

    labels = np.loadtxt(os.path.join(inputPath,'Labels.dat'))

    # The path to the ripser excecutable
    ripserPath = "../../ripser-investigation/ripser"

    np.savetxt(os.path.join(outputPath,'Labels.txt'),labels);

    for count in range(len(labels)):
        # Now run ripser on them
        inputFile = inputPath + "/Shape{}.ldm".format(count)
        outputFile = outputPath + "/Shape{}.bc".format(count)
        if (count%100 == 0):
            print('  Create Ripser Barcode: ',outputFile)
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
        
