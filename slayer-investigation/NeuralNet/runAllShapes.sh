#!/usr/bin/bash
# A script I use for running ML on all the shape data sets
for numClass in 2 4 8 12 16
do
    for numPoints in 32 48 64 96 128 192 256
    do
	directory=../Data/RandomOccluded/${numClass}Class${numPoints}Points
	outputFile=../Data/RandomOccluded/${numClass}Class${numPoints}Points/slayerStats.txt
	rm -f ${outputFile}
	python3 simpleShape.py ${directory} | tee ${outputFile}
    done
done
