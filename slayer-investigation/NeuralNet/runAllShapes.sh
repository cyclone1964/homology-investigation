#!/usr/bin/bash
# A script I use for running ML on all the shape data sets
path=$1
type=$2
for numClass in 2 4 8 12 16
do
    for numPoints in 16 32 48 64 96 128 192 256
    do
	directory=${path}/${numClass}Class${numPoints}Points/$2
	outputFile=${directory}/slayerStats.txt
	rm -f ${outputFile}
	python3 simpleShape.py ${directory} | tee ${outputFile}
    done
done
chmod
