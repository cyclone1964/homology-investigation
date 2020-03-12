#!/bin/tcsh
#
# This script takes as it's only input the path to the training set
# and makes the .bc output files for all the ldms it finds there
set ripserPath = "../../ripser-investigation/ripser/"

# Check that a path was provided
if ($#argv < 1) then
    echo "Usage: generateBarcodeFiles.csh path_to_files"
    exit(1)
endif

# Now process files until there are no more
@ index = 0
@ maxPoints = 0
set path = $argv[1]
while (1)
    set inputFile = "$path/Shape$index.ldm"
    set outputFile = "$path/Shape$index.bc"
    echo "Process Input File: " $inputFile
    if (-f $inputFile) then
	$ripserPath/ripser --dim 4 $inputFile | \
	/usr/bin/awk -f parseRipserFile.awk >! $outputFile
	set numPoints = `/usr/bin/wc -l $outputFile | /usr/bin/awk '{print $1;}'`
	if ($numPoints > $maxPoints) then
	    @ maxPoints = $numPoints
	    echo $maxPoints >! "$path/maxPoints.dat"
	endif
    else
        echo "No Such File: " $inputFile " Exiting"
	break
    endif
    @ index = $index + 1
end
echo $maxPoints > "$path/maxPoints.dat"

