#!/bin/tcsh

python generateMeander.py

rm -f ../Output/Meander/*.ripser
../ripser/ripser ../Output/Meander/ThickMeander.lower_distance_matrix > ../Output/Meander/ThickMeander.ripser
../ripser/ripser ../Output/Meander/ThinMeander.lower_distance_matrix > ../Output/Meander/ThinMeander.ripser

python showBarcodes.py ../Output/Meander/ThickMeander.ripser
python showBarcodes.py ../Output/Meander/ThinMeander.ripser

