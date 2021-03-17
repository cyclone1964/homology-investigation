#!/usr/bin/bash
# A script I use for running ML on all the shape data sets
path=$1
type=$2
for numClass in 2 4 8 12 16
do
    for numPoints in 16 32 48 64 96 128 192 256
    do
	directory=${path}/${numClass}Class${numPoints}Points/$2
	if [ -d "${directory}" ]; then
	    outputFile=${directory}/slayerStats.txt
	    if [ ! -f "${outputFile}" ]; then
		echo Train in ${directory}
		python3 simpleShape.py ${directory} | tee ${outputFile}
	    else
		echo ${outputFile} exists
	    fi
	else
	    echo No Such Directory ${directory}
	fi
    done
done

