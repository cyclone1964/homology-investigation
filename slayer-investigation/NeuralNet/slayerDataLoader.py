import math
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

# Now, we write our own data set class so that we can leverage all
# loader capabilities of pytorch. The structure here is that the
# generated data is stored in a directory in a series of files with a
# specific naming convention.
#
# Labels.dat - list of all the labels (0-N)
# Shape-<ID>.bc - file containing the barcode for the shape
# MaxPoints.dat - the maximum number of barcodes in the 
class SlayerDataset(Dataset):

    """ SlayerDataset: for reading my generated Object data """
    def __init__(self,
                 root_dir,
                 partition = 'train',
                 nu=0.01,
                 limit = 0.001,
                 maxPoints = 256):
        """
        Args:
             root_dir (string): path to the data directory
        
        This routine reads in the vector of IDs from the Directory
        and sets up index lists for the various partitions
        """
        self.nu = nu
        self.limit = limit
        self.root_dir = root_dir
        self.maxPoints = maxPoints

        # This loads the labels, which is a list of all viable IDs
        self.labels = np.loadtxt(root_dir + '/Labels.dat').astype(int)

        # Now split that into training, validation, and test. Training
        # is the first 1/2, validation the next 1/4, and test the rest
        numTrain = math.floor(len(self.labels)/2)
        numVal = math.floor(len(self.labels)/4)
        self.partitions = {'train': range(0,numTrain),
                           'validate': range(numTrain,numTrain+numVal),
                           'test': range(numTrain+numVal,len(self.labels))}

        # Now for this instantiation, we choose one of the partitions
        self.indices = self.partitions[partition];
        print('Partition ',partition,' has length ',len(self.indices))
        
    """
    Return the length of the current partition
    """
    def __len__(self):
        return len(self.indices)

    """ 
    Get an instance of the current partition. The input is taken as an
    index into the currently selected index and the associated
    files are read in.

    In this case the input data is a set of barcodes which we 
    """
    def __getitem__(self,index):

        # Now load the barcodes
        bc = np.loadtxt(self.root_dir + '/Shape' +
                         str(index) + '.bc',
                         dtype='float').astype(float)
        
        # Now do the math that does the rotation
        extent = bc[:,1] + bc[:,0]
        persistence = bc[:,1] - bc[:,0]
        
        # Now do the limitation on the persistence for the given
        # activation function
        persistence = np.where(persistence < self.nu,
                                np.log(persistence/self.nu) + self.nu,
                                persistence)

        # Append the third row of the input data, the one that
        # defines if each column is a "real" data point or a dummy one
        # used to make them all the same size.
        X = np.transpose(np.stack([extent,
                                   persistence,
                                   np.ones(persistence.shape)]))

        # Now let's make it the same size as the rest of them
        Temp = np.zeros((self.maxPoints - X.shape[0],3)).astype(float)

        X = np.concatenate([X, Temp],axis=0)
        
        # Now let's convert it to a torch float
        X = torch.from_numpy(X).float()
        
        y = self.labels[index]
        
        return X, y
