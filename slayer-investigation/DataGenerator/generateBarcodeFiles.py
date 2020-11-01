# This file generates the barcode files. It does this by invoking the ripser program on all the
# shape files found in the source directory
import os
index = 0
dataDir = "../Data/TwoClass64"
ripserPath = "../../ripser-investigation/ripser"

while(1):
    inputFileName = dataDir + "/Shape{}.ldm".format(index)
    outputFileName = dataDir + "/Shape{}.bc".format(index)
    if (os.path.isfile(outputFileName)):
        print(outputFileName,": Exists")
        index = index + 1
        continue

    if (os.path.isfile(inputFileName)):
        command = "rm -f outputFileName;" + ripserPath + "/ripser --dim 2 " + inputFileName
        command = command + " | /usr/bin/awk -f parseRipserFile.awk > " + outputFileName
        print(outputFileName)
        os.system(command)
    else:
        break
    index = index + 1

    
