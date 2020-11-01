# This file contains my implementation of a slayer ML set for the
# "SimpleShape" data set. It is loosely based upon the animal.py
# implementatino with a greatly simplified DataSet and provider and a
# much simpler collater.
import os
import math
import torch
import numpy as np
import sklearn.cluster
from collections import defaultdict
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedShuffleSplit

# Data flow and it's architeture
#
# Our dataset returns a list of num_elements x 2 numpy arrays, with one entry for each
# dimension in the list. 
# [(num_elements x 2) (num_elements x 2) ....] for each dimension
#
# The batching breturns a list of lists from the dataset
#
# 
# Inside of the forward function, each slayer layer needs to be fed with a tuple
# ([batchsize x num_elements x 2], dummy_points[batchsize x num_elements], 2, batch size)
#
# The collator thus returns a list of these tuples, again one for each dimension in the set

#torch.cuda.current_device()
import random
random.seed(123)

import torch.nn as nn

# These are the slayer support functions, which need to be installed
# into our python libraries.
from torchph.nn.slayer import \
     SLayerExponential, SLayerRational, \
     LinearRationalStretchedBirthLifeTimeCoordinateTransform, \
     SLayerRationalHat

from torch.nn.parameter import Parameter
from torch.utils.data import DataLoader, SubsetRandomSampler, Dataset
from torch.autograd import Variable

# This is the training parameters, classed up for us
class train_env:
    cuda = False
    nu = 0.01
    n_epochs = 200
    lr_initial = 0.01
    momentum = 0.9
    lr_epoch_step = 40
    batch_size = 20
    train_size = 0.9

# This vector determins the number of dimensions we will be using to
# train as well as the number of slayer elements for each
# dimension. This is set up to be variable as, for Vietoris-Rips
# complexes, the number of simplices in each dimension reduces
# substantially as the dimension increases
num_elements = [64, 12, 4]

# Now, we write our own data set class so that we can leverage all
# loader capabilities of pytorch. The structure here is that the
# generated data is stored in a directory in a series of files with a
# specific naming convention.
#
# Labels.dat - list of all the labels (0-N)
# Shape-<ID>.#.bc - file containing the barcode for the shape ID in dimension #
#
# Thus the number of classes can be retrieved from the number of unique labels. 
class SimpleShapeDataset(Dataset):

    """ SlayerDataset: for reading my generated Object data """
    def __init__(self,
                 root_dir,
                 num_elements = [64]):
        """
        Args:
             root_dir (string): path to the data directory
             num_elements: a list of element counts for each of the supported dimension
        
        This routine reads in the vector of IDs from the Directory
        and sets up index lists for the various partitions
        """
        self.root_dir = root_dir
        self.num_elements = num_elements
        
        # This loads the labels, which is a list of all viable IDs
        self.labels = np.loadtxt(root_dir + '/Labels.dat').astype(float)
        self.labels = self.labels[:100]
        
        # Set the number of classes from the number of unique labels
        self.num_classes = len(np.unique(self.labels))

        
    """
    Return the length of the data set
    """
    def __len__(self):
        return len(self.labels)

    """ 
    Get an instance of the current partition. The input is taken as an
    index into the label set and the associatedfiles are read in.

    In this case the input data is a set of barcodes which we 
    """
    def __getitem__(self,index):

        # The item is returned as a list of num_elements[dim]x2 arrays and the label
        X = []

        # For each dimension ....
        for dim in range(len(self.num_elements)):

            filename = \
              self.root_dir+ '/Shape' + str(index) + '_' + str(dim) + '.sli'
            # ... if it exists, read in the file as float type,
            # limited to the number of elements for this dimension ...
            if (os.path.isfile(filename)):                      
                points = np.loadtxt(filename,
                                    max_rows = self.num_elements[dim],
                                    ndmin = 2).astype(float)
                # ... and add zeros at the end to fill it out if the read was short ...
                if (points.shape[0] < self.num_elements[dim]):
                    points = np.append(points,
                                          -1*np.ones((self.num_elements[dim] - points.shape[0],
                                                    points.shape[1])).astype(float),
                                           axis=0)
            # ... but if the file doesn't exist, just laod zeros ...
            else:
                points = -1 * np.ones((self.num_elements[dim],2)).astype(float)
            # .. and append to the list
            X.append(points)

        y = self.labels[index]
        
        return X, y

# This function collates a batch so that we can feed the slayer
# modules directly (as it were) To do this, it needs to know the
# number of elements in each list that is consistent with the dataset.
class SimpleCollate:
    def __init__(self,num_elements, cuda=False):
        self.cuda = cuda
        self.num_elements = num_elements

    # Now, this takes a batch as returned by the DataLoader and turns
    # it into data usable directly by the algorithm. The Batch is a
    # list, with one entry per sample in the batck, where each entry
    # is itself a list of the data from the
    def __call__(self, batch):

        # Let's initialize the bar codes and the flags that indicate
        # their validity to have the correct number of rows for each of the dimensions.`q
        barcodes = [np.zeros((0, s ,2)) for s in self.num_elements]
        validity = [np.zeros((0, s)) for s in self.num_elements]
        labels = [];

        # Now for every entry in the batch ... 
        for entry in batch:
            # ... extract the data and the labels and append the labes ...
            X, y = entry
            labels.append(y)
            
            # ... and now, for every dimension, append the input data
            # to the barcodes and use the presence of zeros to indiate
            # which ones are valid.
            for d in range(len(self.num_elements)):
                # Note the pre-pending of a singleton dimension to the
                # shape of the input data to make it appendable to the
                # batched barcode data
                barcodes[d] = np.append(barcodes[d],np.reshape(X[d],(1,)+X[d].shape), axis = 0)

                # Now set the validity flags from the existence of zeros in the array
                v = np.where(X[d][:,0] >= 0, 1, 0)
                validity[d] = np.append(validity[d], np.reshape(v, (1,)+v.shape), axis=0)

        # Now convert into torch tensors, pushing to the cuda if that's enabled
        barcodes = [torch.tensor(barcodes[d],dtype=torch.float32) for d in range(len(barcodes))]
        validity = [torch.tensor(validity[d], dtype=torch.float32) for d in range(len(barcodes))]
        y = torch.tensor(np.array(labels), dtype = torch.long)

        # And then into tuples
        if (self.cuda):
            X = [(barcodes[d].cuda(),
                  validity[d].cuda(),
                  self.num_elements[d],
                  len(batch))
                for d in range(len(self.num_elements))]
            y.cuda()
        else:
            X = [(barcodes[d],
                  validity[d],
                  self.num_elements[d],
                  len(batch))
                for d in range(len(self.num_elements))]

        return X, y

dataset = SimpleShapeDataset(root_dir='../Data/TwoClass64',
                                 num_elements = num_elements)

# The transforms on the input data:
#
# the first selects only the dimensions we want to use (from all the
# ones in the file)
#
# The second acts to convert the dictionary into a flattened set of parameters
# 
# The third applies the selected coordinate transform. Note that the
# predicate here is already satisfied by the second list.
def Slayer(num_elements):
    return SLayerRationalHat(num_elements, radius_init=0.25, exponent=1)

def LinearCell(n_in, n_out):
    m = nn.Sequential(nn.Linear(n_in, n_out), 
                      nn.BatchNorm1d(n_out), 
                      nn.ReLU(),
                     )
    m.out_features = m[0].out_features
    return m


# This is a model for simple shapes wherein the slayer layers are
# allowed to be of different sizes before they are catenated
# together. The number of elements per layer (in this case per barcode
# dimension) is held in a list, and the dimensions run from 0 to
# length(num_elements)-1
class SimpleModel(nn.Module):

    def __init__(self, num_elements, num_classes):

        super().__init__()   
        self.num_elements = num_elements

        self.slayers = []
        cls_in_dim = 0
        for dim in range(len(num_elements)):
            s = Slayer(num_elements[dim])
            self.slayers.append(nn.Sequential(s))
            cls_in_dim += num_elements[dim]

        self.cls = nn.Sequential(nn.Dropout(0.3),
                                 LinearCell(cls_in_dim, 16),    
                                 nn.Dropout(0.2),
                                 LinearCell(16, num_classes))
        
    def forward(self, input):

        x = []
        for dim in range(len(self.slayers)):
            t = self.slayers[dim](input[dim])
            x.append(t)
        x = torch.cat(x, dim=1)
        x = self.cls(x)       

        return x
    
    def center_init(self, dataset):

        allPoints = []
        for index in range(len(dataset)):
            X,_ = dataset[index]
            for dim in range(len(X)):
                points = X[dim]
                indices = np.nonzero(points[:,1])
                points = points[indices[0],:]
                if (index == 0):
                    allPoints.append(points)
                else:
                    allPoints[dim] = np.append(allPoints[dim], points,axis=0)

        for dim in range(len(allPoints)):
            kmeans = sklearn.cluster.KMeans(n_clusters = self.num_elements[dim],
                                            random_state=0).fit(allPoints[dim])
            self.slayers[dim].centers = torch.tensor(kmeans.cluster_centers_)

def experiment(train_slayer):    

    stats_of_runs = []

    splitter = StratifiedShuffleSplit(n_splits=10, 
                                      train_size=train_env.train_size, 
                                      test_size=1-train_env.train_size, 
                                      random_state=123)

    train_test_splits = list(splitter.split(X=dataset.labels, y=dataset.labels))
    train_test_splits = [(train_i.tolist(), test_i.tolist()) for train_i, test_i in train_test_splits]

    # This thing gives run numbers and indices for the two sets
    for run_i, (train_i, test_i) in enumerate(train_test_splits):

        model = SimpleModel(num_elements,dataset.num_classes)
        model.center_init([dataset[i] for i in train_i])

        if (train_env.cuda):
            model.cuda()
            for d in range(len(model.slayers)):
                model.slayers[d].centers.cuda()
                model.slayers[d].sharpness.cuda()
            
        for d in range(len(model.slayers)):
            if (model.slayers[d].centers.is_cuda):
                print("Centers cuda")
            else:
                print("Centers NOT cuda")

        collate_fn = SimpleCollate(num_elements, cuda = train_env.cuda)

        stats = defaultdict(list)
        stats_of_runs.append(stats)
        
        opt = torch.optim.SGD(model.parameters() if train_slayer else model.cls.parameters(), 
                              lr=train_env.lr_initial, 
                              momentum=train_env.momentum)

        for i_epoch in range(1, train_env.n_epochs+1):      

            model.train()

            
            
            dl_train = DataLoader(dataset,
                                  batch_size=train_env.batch_size,
                                  collate_fn = collate_fn,
                                  sampler=SubsetRandomSampler(train_i))

            dl_test = DataLoader(dataset,
                                 batch_size=train_env.batch_size, 
                                 collate_fn = collate_fn,
                                 sampler=SubsetRandomSampler(test_i))

            epoch_loss = 0

            # Every once in a while we update the learning rate
            if i_epoch % train_env.lr_epoch_step == 0:
                for para_group in opt.param_groups:
                    para_group['lr'] = para_group['lr'] * 0.5

            # This is where the actual training happens using the given optimizer.
            for i_batch, (x, y) in enumerate(dl_train, 1):

                y = torch.autograd.Variable(y)

                def closure():
                    opt.zero_grad()
                    y_hat = model(x)
                    loss = nn.functional.cross_entropy(y_hat, y)   
                    loss.backward()
                    return loss

                loss = opt.step(closure)

                epoch_loss += float(loss)
                stats['loss_by_batch'].append(float(loss))
                stats['centers'].append(model.slayers[1][0].centers.data.cpu().numpy())
                
                print("Epoch {}/{}, Batch {}/{}".format(i_epoch, train_env.n_epochs, i_batch, len(dl_train)), end="       \r")

            stats['train_loss_by_epoch'].append(epoch_loss/len(dl_train))            
                     
            model.eval()    
            true_samples = 0
            seen_samples = 0
            epoch_test_loss = 0
            
            for i_batch, (x, y) in enumerate(dl_test):

                y_hat = model(x)
                epoch_test_loss += float(nn.functional.cross_entropy(y_hat, torch.autograd.Variable(y)).data)

                y_hat = y_hat.max(dim=1)[1].data.long()

                true_samples += (y_hat == y).sum()
                seen_samples += y.size(0)  

            test_acc = true_samples.item()/seen_samples
            stats['test_accuracy'].append(test_acc)
            stats['test_loss_by_epoch'].append(epoch_test_loss/len(dl_test))
            
        print('')
        print('acc.', np.mean(stats['test_accuracy'][-10:]))
        
    return stats_of_runs
    
res_learned_slayer = experiment(True)
accs = [np.mean(s['test_accuracy'][-10:]) for s in res_learned_slayer]
print(accs)
print(np.mean(accs))
print(np.std(accs))

res_rigid_slayer = experiment(False)
accs = [np.mean(s['test_accuracy'][-10:]) for s in res_rigid_slayer]
print(accs)
print(np.mean(accs))
print(np.std(accs))

