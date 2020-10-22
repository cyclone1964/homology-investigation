This directory contains the code I use to generate point clouds around academic shapes. Specifically, spheres, cubes, cylinders, torii, and various combinations thereof.

The files are:
_____________________________________________________________

makeTrainingSet.py

A python script that makes the point clouds. It supports lots of shapes by default but the number can be reduced with command line arguments. Creates the following "file types":

 Shape###.dat each line of which defines the X, Y, and Z of a point uniformly sampled along the surface of the object. Read the comments for descriptions
 
 Shape###.ldm which has the lower-diagonal distance matrix for the given shape
 Shape###.ldmf which has the "flattened" lower diagonal distance matrix

It puts them in a directory specified on the command line.

One can read the code to make different combinations than are provided, or to generate different sub-sets of them.
_____________________________________________________________

generateBarcodeFiles.csh, parseRipserFile.awk

A c shell script and supporting awk script that runs ripser for each of the Shape###.ldm files and creates a formatted barcode Shape###.bc where each line is a bar code, defined by three numbers: the birth, the death, and the dimension
_____________________________________________________________

makeSlayerFiles.py

A program that breaks the barcode files into "slayer intput" files of the name
Shape###_X.sli where X is the dimension and the lines have two numbers: the birth and the persistence. The lines in this file are sorted by persistence. This supports the Dataset used by my slayer interface
_____________________________________________________________

plotShape.py

A program that will read in and plot a shape from the .dat file in python.
_____________________________________________________________

generatePointsFromStl.m

A MATLAB file that can read in an Stl file (a form of 3D specification file one can get from, say, turobosquid) and generate uniformly sampled points from it.

