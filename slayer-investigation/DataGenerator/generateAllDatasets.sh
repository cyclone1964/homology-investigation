#!/usr/bin/bash
for numClass in 2 4 8 12 16
do
    for numPoints in 16 32 48 64
    do
	directory=../Data/${numClass}Class${numPoints}
	mkdir ${directory}
	rm -f ${directory}/*
	python3 makeTrainingSet.py ${directory} ${numClass} ${numPoints}
    done
done
