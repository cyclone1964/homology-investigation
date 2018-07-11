# This function makes a simple 2D set of points, from those generates
# the lower_distance_matrix, and plots them so we can discuss them.

# For sqrt
import math

# for var and mean
import numpy

# And the plotting package
import matplotlib.pyplot as plt

# The main function that opens the input file, which is it's only
# argument, and is assumed to be a lower_distance_matrix.
if __name__ == "__main__":

    # This is a list of points that we will plot and then feed
    # generate the LDM for
    xvalues = [-1,1,0,0,5]
    yvalues = [0,0,2,5,5]
    labels = ['A', 'B', 'C', 'D', 'E']

    xvalues = [100*x for x in xvalues]
    yvalues = [100*y for y in yvalues]
    fig, ax = plt.subplots()

    ax.plot(xvalues,yvalues,'k.')
    for index in range(len(xvalues)):
        ax.text(xvalues[index], yvalues[index], labels[index],va='bottom')
    ax.axis('equal')
    ax.axis([-200, 600, -100, 600])
    ax.set_title('Ripser Test Case')
    plt.xlabel('X')
    plt.ylabel('Y')

    file = open('RipserTest.lower_distance_matrix','w')
    
    for firstIndex in range(len(xvalues)):
        for secondIndex in range(firstIndex):
            x = numpy.mean([xvalues[firstIndex], xvalues[secondIndex]])
            y = numpy.mean([yvalues[firstIndex], yvalues[secondIndex]])
            dx = xvalues[firstIndex] - xvalues[secondIndex]
            dy = yvalues[firstIndex] - yvalues[secondIndex]
            l = str(round(math.sqrt(dy*dy+dx*dx)))
            angle = math.atan2(yvalues[secondIndex] - yvalues[firstIndex],
                               xvalues[secondIndex] - xvalues[firstIndex])
            angle = angle * 180/math.pi
            if (angle > 180):
                angle = angle - 390
            if (angle > 90 or angle < -90):
                angle = angle +180
            ax.text(x,y,l,ha='center',rotation=angle)
            file.write(l + ',')
        if (firstIndex > 0):
            file.write('\n')

    file.close()

    plt.draw()
    plt.savefig("RipserTest.png")
    
    
    
