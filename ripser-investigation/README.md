This directory contains the ripser-specific support functions that we
are using to explore the application of the ripser program to large
data sets. Right now this is limited to the "GreenGenes" data set.

Directory Structure


ripser:

Contains a heavily cleaned up and commented version of the ripser
program. We are looking into ways to speed it up even more, both
algorithmically and computationally.

GreenGenes:

Contains the data base downloadable from :

http://greengenes.lbl.gov/Download/Sequence_Data/Fasta_data_files/).

It also contains a c program computeLowerDistanceMatrix.c that parses the
.fasta file and computes hamming distances for each of the alignments
and stores them in lower distance matrix format in a text file.

Tools:

Contains a single python tool showBarcodes.py that plots the barcodes
for each of the simplex dimensions.

Output:

Where we stage output of the different programs, specifically
createLowerDistanceMatrix and ripser itself.

WE PRESUME THAT THE FOLLOWING CONVENTIONS ARE ADHERED TO

To prevent loading data files into the repo, the following suffixes
have been .gitignored. Please use them for all data files

.fasta - the files provided by GreenGenes
.lower_distance_matrix - the files generated by createLowerDistanceMatrix
.ripser - the output of ripser

Papers:

Contains relevant papers about topological homology

Notes:

Contains notes taken at various meetings

Running the thing

From the shell

If one was going to do the whole data base, something like ...

createLowerDistanceMatrix \
GreenGenes/isolated_named_strains_gg16S_aligned.fasta \
Output/isolated_named_strains_gg16S_aligned.lower_distance_matrix

ripser/ripser \
Output/isolated_named_strains_gg16S_aligned.lower_distance_matrix > \
Output/isolated_named_strains_gg16S_aligned.ripser

python \
Tools/showBarCodes.py \
Output/isolated_named_strains_gg16S_aligned.ripser

However as currently written, createLowerDistanceMatrix would take on
the order of a week to run on the whole data set, ripser would crash
for memory reasons, and I can't imagine what showBarCodes would do. So
I usually ^C the createLowerDistanceMatrix after (a while), then use
head to create a smaller lower_distance_matrix then use
ripser/showBarCodes on that.

head -n 1024 \
Output/isolated_named_strains_gg16S_aligned.lower_distance_matrix \
Output/isolated_named_strains_gg16S_aligned.1024.lower_distance_matrix

ripser/ripser \
Output/isolated_named_strains_gg16S_aligned.1024.lower_distance_matrix > \
Output/isolated_named_strains_gg16S_aligned.1024.ripser

python Tools/showBarCodes.py \
Output/isolated_named_strains_gg16S_aligned.1024.ripser
