#!/bin/tcsh
#
# This script makes the input files for the performance measurements
# and then executes the performance measurements on them, writing the
# output to "performance.dat". We hope.

# First, the GreenGenes Input
#echo "Create the academic performance input files"
#python createPerformanceDistributions.py

#echo "Create the GreenGenes performance input file"
#../GreenGenes/createLowerDistanceMatrix --stride 512 \
#   ../GreenGenes/isolated_named_strains_gg16s_aligned.fasta \
#   ../Output/Performance/GreenGenes.lower_distance_matrix
foreach type("GreenGenes" "Gaussian" "Uniform" "Line")

    # First, we make the full-dimension timing files
    set outputFile = ../Output/Performance/$type-Full.dat

    if (!(-e $outputFile)) then
	@ size = 24
	while ($size < 32)
	    echo Running $type Size $size
	    rm -f Temp.dat
	    head -n $size ../Output/Performance/$type.lower_distance_matrix > Temp.dat
	    @ dim = $size + 1
	    set command = "../ripser/ripser --dim $dim Temp.dat"
	    set result = `/usr/bin/time -p $command |& awk '/sys/{print $2}'`
	    echo $size $result >> $outputFile
	    @ size = $size + 1
	end

	rm -f Temp.dat
    else
	echo $outputFile Already Exists

    # Now, the fixed-dimension files
    set outputFile = ../Output/Performance/$type-Fixed.dat
    if (!(-e $outputFile)) then
	@ size = 64
	while ($size <= 729)
	    echo Running $type Size $size
	    rm -f Temp.dat
	    head -n $size ../Output/Performance/$type.lower_distance_matrix > Temp.dat
	    set command = "../ripser/ripser --dim 3 Temp.dat"
	    set result = `/usr/bin/time -p $command |& awk '/sys/{print $2}'`
	    echo $size $result >> $outputFile
	    @ size = $size + $size / 2
	end

	rm -f Temp.dat
    endif
end
