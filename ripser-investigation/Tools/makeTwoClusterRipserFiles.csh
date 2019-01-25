#!/bin/tcsh

@ d = 1

while (${d} < 16) 
      set input = ../Output/TwoCluster/TwoCluster-$d.lower_distance_matrix
      set output = ../Output/TwoCluster/TwoCluster-$d.ripser
      echo ripser $input $output

      ../ripser/ripser --dim 2 $input > $output
      python histogramDistances.py $input
      python showBarcodes.py $output
      @ d = ${d} + 1
end
