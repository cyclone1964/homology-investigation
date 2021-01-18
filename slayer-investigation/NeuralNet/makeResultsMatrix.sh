#!/bin/sh
# A script i use to generate results matrices for plotting
directory=$1
outputFile=$2

rm -f ${outputFile}
touch ${outputFile}
for numClass in 2 4 8 12 16
do
    for numPoints in 16 32 48 64 96 128 192 256
    do
	file=${directory}/${numClass}Class${numPoints}Points/slayerStats.txt
	if (test -f ${file}); then
	    echo -n $numClass $numPoints " " >> ${outputFile}
	    grep Learned  ${file} | awk '{print $5,$7;}' >> ${outputFile}
	fi
    done
done
	    
		  
