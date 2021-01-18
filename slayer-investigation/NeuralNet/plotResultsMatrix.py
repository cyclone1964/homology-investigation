import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patch
from matplotlib.collections import PatchCollection

# Read the data
matrixData = np.loadtxt(sys.argv[1]).astype(float)

# Now, the first column contains the number of classes, the second the
# number of points, and the third the weight. We wish to convert this
# to a matrix we can plot.

numClasses = np.unique(matrixData[:,0])
numPoints = np.unique(matrixData[:,1])

matrix = np.zeros((len(numClasses),len(numPoints)))

[f, ax] = plt.subplots(1)
for classIndex in range(len(numClasses)):
    for pointIndex in range(len(numPoints)):
        indices = np.logical_and(matrixData[:,0] == numClasses[classIndex],
                         matrixData[:,1] == numPoints[pointIndex])
        matrix[classIndex,pointIndex] =  matrixData[indices,2]
plt.pcolor(matrix)
for classIndex in range(len(numClasses)):
    for pointIndex in range(len(numPoints)):

        color = "black"
        if (matrix[classIndex,pointIndex] < 0.6):
            color = "white"
        plt.text(pointIndex+0.5,classIndex+0.5,
                 "{:.2f}".format(matrix[classIndex,pointIndex]),
                 ha="center",va="center",
                 color=color)
            
ax.set_xticks([0.5 + i for i in range(len(numPoints))])
ax.set_xticklabels([str(i) for i in numPoints])
ax.set_yticks([0.5 + i for i in range(len(numClasses))])
ax.set_yticklabels([str(i) for i in numClasses])
ax.set_xlabel("Number Of Points")
ax.set_ylabel("Number Of Classes")
if (len(sys.argv) > 2) :
    plt.title(sys.argv[2])
else:    
    plt.title("Slayer Object Recognition Performance")
plt.show()
