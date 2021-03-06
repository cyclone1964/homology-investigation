# THis simple routine plots barcodes from the output files of
# ripser. It takes the name of the file as an argument:
#
# python showBarcodes.py ripser.out
#
# I may add more options as time goes on

# Used to split lines
import os

# We use re for parsing of the lines
import re

# This for getting at argv
import sys

# For sqrt
import math

# This for doing KS tests on the endpoints to see if they are normal
from scipy import stats
import numpy

# And the plotting package
import matplotlib.pyplot as plt

# plotBarCodes - do what it says
#
# plotBarCodes(name, dimension, barcodes) does what it says where:
#
# name - is a name used to identify the input barcode source
# dimension - is a string representation of the current dimension
# barcodes is a list of 2-tuples or a list of lists of 2 values.
#
# It plots the barcodes and then saves the plot to a file. 
def plotBarcodes(name,barcodes,dimension):

    print("Plot ",len(barcodes), " barcodes")
    fig, ax = plt.subplots()
    for index in range(len(barcodes)):
        ax.plot(barcodes[index],[index, index])
    ax.set_title(name + " Dimension: " + dimension)
    plt.savefig(name + "-barcodes-" + dimension + ".png")

# This function plots a histogram of the starting and ending distance for
# the current dimension.
def plotEndpoints(name,barCodes,dimension):
    print("Plot", len(barCodes), "histograms")

    startPoints = [barCodes[i][0] for i in range(len(barCodes))]
    endPoints = [barCodes[i][1] for i in range(len(barCodes))]
    lengths = [(barCodes[i][1] - barCodes[i][0]) for i in range(len(barCodes))]

    fig, ax = plt.subplots()
    ax.plot(lengths, endPoints,'.')
    ax.set_title(name + " Dimension: " + dimension)
    plt.xlabel('Barcode Length')
    plt.ylabel('Barcode Endpoint')
    plt.savefig(name +"-endpoints-" + dimension + ".png")

    # Now, let's do an Anderson-Wilke test against a normal distribution
    temp = (endPoints - numpy.mean(endPoints))/math.sqrt(numpy.var(endPoints))

    # Plot the histogram as a bar plot
    fix, ax = plt.subplots()
    plt.hist(endPoints,128)
    
    ax.set_title(name + " Dimension:" + dimension)
    plt.xlabel('Barcode Endpoint')
    plt.ylabel('Count')
    plt.draw()
    plt.savefig(name +"-endpoint-histogram-" + dimension + ".png")

    # Now, let's do a histogram of the lengths
    ax.clear()
    ax.hist(lengths,128)
    ax.set_title(name + " Dimension: " + dimension)
    plt.xlabel('Barcode Length')
    plt.ylabel('Count')
    plt.draw()
    plt.savefig(name + "-length-histogram-" + dimension + ".png")
    
    # Now let's plot the lengths .vs. the starting point
    ax.clear()
    plt.plot(endPoints, lengths, 'k.')
    plt.xlabel('Barcode Endpoint');
    plt.ylabel('Barcode Length')
    ax.set_title(name + " Dimension: " + dimension)
    plt.draw()
    plt.savefig(name + "-end-length-" + dimension + ".png")
    
# The main function that opens the input file, which is it's only
# argument, and is assumed to be the catted output of a ripser run. It
# parses the lines looking for dimension prints (which define new
# dimensions) or barcode lines.
if __name__ == "__main__":

    # First, open the file, read it, and close it again
    file = open(sys.argv[1],'r');
    lines = file.readlines()
    file.close()

    # Get the root of the filename as the annotation for the plots and
    # their output files
    name = os.path.splitext(sys.argv[1])
    name = name[0]
    
    # Initialize the list of barcodes to empty
    barcodes = []
    allBarCodes = []

    # The dimension lines have the word "persistence" in them and the last word
    # is a representation of the dimension.
    
    # The barcodes are presumed to exist on lines that have a "[" as
    # their first non-space character.
    for line in lines:

        # If this is a a new "dimension", we need to make a new plot for it.
        if (re.match('persistence',line)):

            # If we already have barcodes, then we need to make a new
            # axes, plot them, and show the axes
            if (len(barcodes) > 0):
                plotBarcodes(name,barcodes,dimension)
                plotEndpoints(name,barcodes,dimension)
                allBarCodes.append(barcodes)

            # In any event, initialize new barcodes and starting point
            words = line.split(' ')
            dimension = words[-1][0]
            barcodes = []
            continue

        # Otherwise, get rid of any spaces at the beginning of the line,
        # then check to makes sure the next character is a "["
        temp = re.sub('\ ','',line)
        if (re.match('^\[',temp)):

            # It is!! Remove all the punctuation except the comma
            barcode = re.sub('[[\)\n]','',temp)

            # Split on the comma
            barcode = barcode.split(',')

            # Convert this to a list. (Should I use a tuple instead?) and
            # check that it is the right length. There are occasional bad
            # lines in there!
            barcode = [float(y) for y in barcode if len(y) > 0]
            if (len(barcode) is not 2):
                print("Bad Line: ",line)
                continue;
        
            # Add this to the end of the barcodes list
            barcodes.append(barcode)


    allBarCodes.append(barcodes)
    plotBarcodes(name,barcodes,dimension)
    plotEndpoints(name,barcodes,dimension)
    

