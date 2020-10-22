# This function reads in the bar codes and from that makes files named
# ShapeXXX_Y.sli. This is a file, sorted by barcode lifetime, with two
# columsn: start and persistence, and Y is the dimension.
import os
import sys
import numpy as np
import os.path as pth

class TempClass:
    def __init__(self, this, that):
        self.this = this
        self.that = that

if __name__ == "__main__":


    print('Parse Arguments ...')
    if (len(sys.argv) < 2):
        print('Usage: ', sys.argv[0],' PathToOutput')
        sys.exit()

    print(" Path", sys.argv[1])
    root_dir = sys.argv[1]

    labels = np.loadtxt(root_dir + '/Labels.dat').astype(int)

    for index in range(len(labels)):
        print("Shape", str(index))
        bc = np.loadtxt(root_dir +
                        '/Shape' + str(index) + '.bc', 
                        dtype='float').astype(float)
    
        persistence = bc[:,1] - bc[:,0]
        dims = bc[:,2].astype(int)
    
        for dim in np.unique(dims):
            indices = np.nonzero(dims == dim)
            indices = indices[0]
            print("    D", dim, ": ", len(indices)) 

            i = np.argsort(persistence[indices])
            indices = indices[i]
            np.savetxt(root_dir + '/Shape' + repr(index) +
                       '_' + repr(dim) + '.sli',
                       bc[indices,0:2])

