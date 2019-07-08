#!/bin/tcsh -f
@ index = 0

while ($index < 500)
    echo Cube $index
    set inputFile = "../Output/MachineLearning/Cube$index.ldm"
    set outputFile = "../Output/MachineLearning/Cube$index.ripser"
    ../ripser/ripser --dim 4 $inputFile | \
    awk -f parseRipserFile.awk >! $outputFile

    echo Sphere $index
    set inputFile = "../Output/MachineLearning/Sphere$index.ldm"
    set outputFile = "../Output/MachineLearning/Sphere$index.ripser"
    ../ripser/ripser --dim 4 $inputFile | \
    awk -f parseRipserFile.awk >! $outputFile
    @ index = $index + 1
end

