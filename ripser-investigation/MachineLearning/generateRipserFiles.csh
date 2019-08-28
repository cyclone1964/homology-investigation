#!/bin/tcsh -f
#
# This script takes as it's only input the path to the training set
# and makes the .ripser output file for all the ldms it finds there

# Check that a path was provided
if ($#argv < 1) then
    echo "Usage: makeTrainingData.csh path_to_files"
    exit(1)
endif

# Now process files until there are no more
@ index = 0
set path = $argv[1]
while (1)
    set inputFile = "$path/Shape$index.ldm"
    set outputFile = "$path/Shape$index.ripser"
    echo "Process Input File: " $inputFile
    if (-f $inputFile) then
	../ripser/ripser --dim 4 $inputFile | \
	/usr/bin/awk -f parseRipserFile.awk >! $outputFile
    else
        echo "No Such File: " $inputFile " Exiting"
	break
    endif
    @ index = $index + 1
end

