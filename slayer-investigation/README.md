This directory contains the files I developed to attempt to train simple ANNs to identify various shapes using the input layer described in "Deep Learning with Topological Signatures" by Hofer, Kwitt, Niethammer and Uhl. There are three files here at this time

There are three directories:

DataGenerater - the files that generate the data. Requires an installation of ripser to run, the path to which is defined in the file "generateBarcodeFiles.csh"

Data - where the data is stored. Each directory containes the output of "generateTrianingData" and "generateRipserFiles.csh" The types of files are:

Labels .dat - the labes for each of the shapes in the directory
ShapeXXX.dat - the X/Y/Z locations of the points on the surface of the shape
ShapeXXX.ldm - the lower distance matrix for the points in the shape
ShapeXXX.ldmf - a "flattened" ldm
ShapeXXX.bc - barcode file

NeuralNet - the code that imeplments the ANN in pytorch. 
slayer.py - contains the definition of the input layer as a pytorch module.

dataLoader.py - the data loader module. This module reads bar codes directly, "rotates" them, and then implements the scale change portion of the activiation function. It is here that the variable "nu" is set. 

trainSlayer - executes the training

