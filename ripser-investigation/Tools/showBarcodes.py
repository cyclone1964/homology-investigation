# THis simple routine plots barcodes from the output files of
# ripser. It takes the name of the file as an argument:
#
# python showBarcodes.py ripser.out
#
# I may add more options as time goes on

# We use re for parsing of the lines
import re
import sys

# And the plotting package
import matplotlib.pyplot as plt

if __name__ == "__main__":

    # First, open the file, read it, and close it again
    file = open(sys.argv[1],'r');
    lines = file.readlines()
    file.close()

    # Now, the barcodes are presumed to exist on lines that have a "[" as
    # their first non-space character. so let's find those and then
    # convert them. While we are at it, we find the lowest non-zero
    # starting point. This makes plotting a little nicer.
    barcodes = []
    startingPoint = 0

    for line in lines:

        # First, get rid of any spaces at the beginning of the line,
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
        
            barcode = [int(y) for y in barcode if y.isdigit()]
            if (len(barcode) is not 2):
                print "Bad Line: ",line
                continue;
        
            # Add this to the end of the barcodes list
            barcodes.append(barcode)

            if (barcode[0] > 0 and barcode[0] > startingPoint):
                startingPoint = barcode[0]

    print "Read ",len(barcodes), " barcodes @ ", startingPoint

    # Let's sort them by starting point (?)

    # Now plot them
    for index in range(len(barcodes)):
        if (barcodes[index][0] > 0):
            plt.plot(barcodes[index],[index, index])

    plt.show()
