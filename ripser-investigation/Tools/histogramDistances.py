# This simple routine plots this histogram of the distances in a lower
# distance matrix.
#
# python histogramDistances.py FileName.lower_distance_matrix
#
# I may add more options as time goes on

# This for getting at argv
import sys

# For sqrt
import math

# for var and mean
import numpy

# And the plotting package
import matplotlib.pyplot as plt

# The main function that opens the input file, which is it's only
# argument, and is assumed to be a lower_distance_matrix.
if __name__ == "__main__":

    file = open(sys.argv[1],'r');
    lines = file.readlines()
    file.close()

    distances = []
    for line in lines:
        words = line.split(',')
        words = words[0:-1]
        temp = [int(word) for word in words]
        distances = distances + temp

    print "Read ",len(distances)," distances"
    fig, ax = plt.subplots()
    plt.hist(distances,256)
    ax.set_title("Distance Histogram: " + sys.argv[1])
    plt.xlabel('Distance')
    plt.ylabel('Count')
    plt.draw()
    plt.savefig(sys.argv[1] + "-distance-hist.png")


    var = math.sqrt(numpy.var(distances))
    mean = numpy.mean(distances)

    print "1 sigma: ", mean-var, " - ", mean+var
    print "2 sigma: ", mean-2*var, " - ", mean+2*var
