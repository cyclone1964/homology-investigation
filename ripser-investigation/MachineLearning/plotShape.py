# This script plots the points in a shape file in 3 space so that I can
# visualize them and make sure that they are right. Maybe
import sys
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

fileName = sys.argv[1]
X = np.loadtxt(fileName,delimiter=',')

# Make a figure and a 3D axis
fig = plt.figure()
ax = fig.add_subplot(111,projection = '3d')
ax.scatter(X[:,0],X[:,1],X[:,2],'b');
plt.show()
