#!/usr/bin/bash
# A script I use for running ML on all the shape data sets
for numClass in 2 4 8 12 16
do
    for numPoints in 96 128
    do
	directory=../Data/Occluded/${numClass}Class${numPoints}Points
	outputFile=../Data/Occluded/${numClass}Class${numPoints}Points/slayerStats.txt
	python3 simpleShape.py ${directory} | tee ${outputFile}
    done
done
