import os
import sys
import torch
import torch.nn as nn
import sklearn.cluster
import numpy as np

from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedShuffleSplit

import itertools
import matplotlib.pyplot as plt

from torchph.nn.slayer import SLayerExponential, \
    SLayerRational, \
    LinearRationalStretchedBirthLifeTimeCoordinateTransform, \
    prepare_batch, \
    SLayerRationalHat

from sklearn.model_selection import ShuffleSplit
from collections import Counter, defaultdict
from torch.utils.data import DataLoader, SubsetRandomSampler

from collections import OrderedDict

from torch.autograd import Variable

from sklearn.model_selection import StratifiedShuffleSplit

# These are the parameters used for training, including the maximum
# number of elements of each of the dimensions we use.
dims_to_use = ["dim_{}".format(k) for k in range(3)]
class train_env:
    plot_enable = False
    n_splits = 8
    num_elements = 64
    nu = 0.01
    n_epochs = 30
    lr_initial = 0.01
    momentum = 0.9
    lr_epoch_step = 40
    batch_size = 100
    train_size = 0.9
    

coordinate_transform = \
    LinearRationalStretchedBirthLifeTimeCoordinateTransform(nu=train_env.nu)

# This is the data set used to provide access to the output of ripser
# as we haev them maintained. Specifically, in the data directory, we
# find files of the following name structures
#
# Labels.dat - list of all the labels (0-N)
# Shape#.bc - file containing the barcode for the shape ID in dimension 
# Shape#.dat - file containint the locations of all the points on the shape
# Shape#.ldm - the lower diagonal matrix
# Shape#.ldmf - a flattened version thereof
# Shape#_dim_#.sli - the barcode broken out by dimension
#
# Thus the number of classes can be retrieved from the number of unique labels. 
class SimpleDataset(Dataset):

    """ SlayerDataset: for reading my generated Object data """
    def __init__(self,
                 root_dir,
                 dims_to_use):
        """
        Args:
             root_dir (string): path to the data directory
             max_elements: a list of element counts for each of the supported dimension
        
        This routine reads in the vector of IDs from the Directory
        and sets up index lists for the various partitions
        """
        self.root_dir = root_dir
        self.dims_to_use = dims_to_use
        
        # This loads the labels, which is a list of all viable IDs
        self.labels = np.loadtxt(root_dir + '/Labels.dat').astype(int)
        
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

        # The item is returned as a list of max_elements[dim]x2 arrays
        # and the label
        

        # For each dimension ....
        X = {}
        for dim in self.dims_to_use:

            filename = \
              self.root_dir+ '/Shape' + str(index) + '_' + dim + '.sli'

            # ... if it exists, read in the file as float type,
            # limited to the number of elements for this dimension ...
            if (os.path.isfile(filename)):                      
                points = np.loadtxt(filename,
                                    max_rows = train_env.num_elements,
                                    ndmin = 2).astype(np.float32)
            # .. and append to the list
            X[dim] = torch.from_numpy(points)

        y = self.labels[index]

        return X, y



class PHTCollate:   
    def __init__(self, nu, cuda=True):
        self.cuda = cuda
        
    def __call__(self, sample_target_iter):
        
        augmented_samples = []
        x, y = dict_sample_target_iter_concat(sample_target_iter)
                                              
        for k in x.keys():
            batch_view = x[k]
            x[k] = prepare_batch(batch_view, 2)                  

        y = torch.LongTensor(y)    

        if self.cuda:
            # Shifting the necessary parts of the prepared batch to the cuda
            x = {k: collection_cascade(v,
                                       lambda x: isinstance(x, tuple),
                                       lambda x: (x[0].cuda(), x[1].cuda(), x[2], x[3]))
                 for k, v in x.items()}

            y = y.cuda()

        return x, y                       
    
collate_fn = PHTCollate(train_env.nu, cuda=True)

def Slayer(n_elements):
    return SLayerRationalHat(n_elements, radius_init=0.25, exponent=1)

def LinearCell(n_in, n_out):
    m = nn.Sequential(nn.Linear(n_in, n_out), 
                      nn.BatchNorm1d(n_out), 
                      nn.ReLU(),
                     )
    m.out_features = m[0].out_features
    return m

class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()   
        
        self.slayers = ModuleDict()
        for k in dims_to_use:
            s = Slayer(train_env.num_elements)
            self.slayers[k] = nn.Sequential(s)
        cls_in_dim = len(dims_to_use)*train_env.num_elements
            
        self.cls = nn.Sequential(
                                nn.Dropout(0.3),
                                LinearCell(cls_in_dim, int(cls_in_dim/4)),    
                                nn.Dropout(0.2),
                                LinearCell(int(cls_in_dim/4), int(cls_in_dim/16)),  
                                nn.Dropout(0.1),
                                nn.Linear(int(cls_in_dim/16), 20))
        
    def forward(self, input):
        x = []
        for k in dims_to_use:
            xx = self.slayers[k](input[k])
            x.append(xx)

        x = torch.cat(x, dim=1)          
        x = self.cls(x)       
                                              
        return x
    
    def center_init(self, sample_target_iter):
        centers = k_means_center_init(sample_target_iter, train_env.num_elements)
        
        for k, v in centers.items():
            self.slayers._modules[k][0].centers.data = v
            
class ModuleDict(nn.Module):
    def __init__(self):
        super().__init__()
        
    def __setitem__(self, key, item):
        setattr(self, key, item)
        
    def __getitem__(self, key):
        return getattr(self, key)

def k_means_center_init(sample_target_iter: dict, n_centers: int):
    samples_by_view, _ = dict_sample_target_iter_concat(sample_target_iter)
    
    points_by_view = {}
    for k, v in samples_by_view.items():
        points_by_view[k] = torch.cat(v, dim=0).numpy()
    
    k_means = {k: sklearn.cluster.KMeans(n_clusters=n_centers, init='k-means++', n_init=10, random_state=123)
               for k in points_by_view.keys()}
    
    center_inits_by_view = {}
    for k in points_by_view.keys():
        centers = k_means[k].fit(points_by_view[k]).cluster_centers_
        centers = torch.from_numpy(centers)
        center_inits_by_view[k] = centers
        
    return center_inits_by_view  

def dict_sample_target_iter_concat(sample_target_iter: iter):
    """
    Gets an sample target iterator of dict samples. Returns
    a concatenation of the samples based on each key and the
    target list.

    Example:
    ```
    sample_target_iter = [({'a': 'a1', 'b': 'b1'}, 0), ({'a': 'a2', 'b': 'b2'}, 1)]
    x = dict_sample_iter_concat([({'a': 'a1', 'b': 'b1'}, 0), ({'a': 'a2', 'b': 'b2'}, 1)])
    print(x)
    ({'a': ['a1', 'a2'], 'b': ['b1', 'b2']}, [0, 1])
    ```

    :param sample_target_iter:
    :return:
    """

    samples = defaultdict(list)
    targets = []

    for sample_dict, y in sample_target_iter:
        for k, v in sample_dict.items():
            samples[k].append(v)

        targets.append(y)

    samples = dict(samples)

    length = len(samples[next(iter(samples))])
    assert all(len(samples[k]) == length for k in samples)

    return samples, targets

def numpy_to_torch_cascade(input):
    def numpy_to_torch(array):
        return_value = None
        try:
            return_value = torch.from_numpy(array)
        except Exception as ex:
            if len(array) == 0:
                return_value = torch.Tensor()
            else:
                raise ex

        return return_value.float()

    return collection_cascade(input,
                              stop_predicate=lambda x: isinstance(x, numpy.ndarray),
                              function_to_apply=numpy_to_torch)

def collection_cascade(input, stop_predicate: callable, function_to_apply: callable):
    if stop_predicate(input):
        return function_to_apply(input)
    elif isinstance(input, list or tuple):
        return [collection_cascade(x,
                                   stop_predicate=stop_predicate,
                                   function_to_apply=function_to_apply) for x in input]
    elif isinstance(input, dict):
        return {k: collection_cascade(v,
                                      stop_predicate=stop_predicate,
                                      function_to_apply=function_to_apply) for k, v in input.items()}
    else:
        raise ValueError('Unknown type collection type. Expected list, tuple, dict but got {}'
                         .format(type(input)))

def experiment(train_slayer,dataset):
    
    stats_of_runs = []
    
    splitter = StratifiedShuffleSplit(n_splits=train_env.n_splits, 
                                      train_size=train_env.train_size, 
                                      test_size=1-train_env.train_size, 
                                      random_state=123)
    train_test_splits = list(splitter.split(X=dataset.labels, y=dataset.labels))
    train_test_splits = [(train_i.tolist(), test_i.tolist()) for train_i, test_i in train_test_splits]
    
    for run_i, (train_i, test_i) in enumerate(train_test_splits):
        print('Run {}: '.format(run_i),end='',flush=True)

        model = SimpleModel()
        model.center_init([dataset[i] for i in train_i])
        model.cuda()

        stats = defaultdict(list)
        stats_of_runs.append(stats)
        
        opt = torch.optim.SGD(model.parameters() if train_slayer else model.cls.parameters(), 
                              lr=train_env.lr_initial, 
                              momentum=train_env.momentum)

        for i_epoch in range(1, train_env.n_epochs+1):
            if ((i_epoch-1)%10 == 0):
                print('{:02d} '.format(i_epoch-1),end='',flush=True)
            model.train()
            
            dl_train = DataLoader(dataset,
                                  batch_size=train_env.batch_size, 
                                  collate_fn=collate_fn,
                                  sampler=SubsetRandomSampler(train_i))

            dl_test = DataLoader(dataset,
                                 batch_size=train_env.batch_size, 
                                 collate_fn=collate_fn, 
                                 sampler=SubsetRandomSampler(test_i))

            epoch_loss = 0    

            if i_epoch % train_env.lr_epoch_step == 0:
                adapt_lr(opt, lambda lr: lr*0.5)

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
                for dim in dims_to_use:
                    stats['centers'].append(model.slayers[dim][0].centers.data.cpu().numpy())

#                print("Epoch {}/{}, Batch {}/{}".format(i_epoch, train_env.n_epochs, i_batch, len(dl_train)), end="       \r")

            stats['train_loss_by_epoch'].append(epoch_loss/len(dl_train))            
                     
            model.eval()    
            true_samples = 0
            seen_samples = 0
            epoch_test_loss = 0
            
            for i_batch, (x, y) in enumerate(dl_test):

                y_hat = model(x)
                epoch_test_loss += float(nn.functional.cross_entropy(y_hat, torch.autograd.Variable(y.cuda())).data)

                y_hat = y_hat.max(dim=1)[1].data.long()

                true_samples += (y_hat == y).sum()
                seen_samples += y.size(0)  

            test_acc = true_samples.item()/seen_samples
            stats['test_accuracy'].append(test_acc)
            stats['test_loss_by_epoch'].append(epoch_test_loss/len(dl_test))
        print(' Mean acc: ', np.mean(stats['test_accuracy'][-train_env.n_splits:]))
        
    return stats_of_runs


# Run the experienent with slayer doing the learning, which is to say
# the elements actually adapt
train_env.root_dir = sys.argv[1]
dataset = SimpleDataset(train_env.root_dir, dims_to_use)
dataset.data_transforms = [
                           lambda x: {k: x[k] for k in dims_to_use}, 
                           numpy_to_torch_cascade,
                           lambda x: collection_cascade(x, 
                                                        lambda x: isinstance(x, torch.Tensor), 
                                                        lambda x: coordinate_transform(x))
                           ]
res_learned_slayer = experiment(True,dataset)
accs = [np.mean(s['test_accuracy'][-10:]) for s in res_learned_slayer]
print("Learned Performance ({}): Mean {} Var {}".format(train_env.root_dir,
                                                     np.mean(accs),
                                                     np.std(accs)))
if (False):
    # Now run it with the centers unlearned to see the different
    res_rigid_slayer = experiment(False)
    accs = [np.mean(s['test_accuracy'][-10:]) for s in res_rigid_slayer]
    print("Rigid Performance ({}): Mean {} Var {}".format(train_env.root_dir,
                                                          np.mean(accs),
                                                          np.std(accs)))

if (train_env.plot_enable):
    stats = res_learned_slayer[-1]
    plt.figure()
    if 'centers' in stats:
        c_start = stats['centers'][0]
        c_end = stats['centers'][-1]

        plt.plot(c_start[:,0], c_start[:, 1], 'bo', label='center initialization')
        plt.plot(c_end[:,0], c_end[:, 1], 'ro', label='center learned')

        all_centers = numpy.stack(stats['centers'], axis=0)
        for i in range(all_centers.shape[1]):
            points = all_centers[:,i, :]
            plt.plot(points[:, 0], points[:, 1], '-k', alpha=0.25)
        

        plt.legend()
    
        plt.figure()
        plt.plot(stats['train_loss_by_epoch'], label='train_loss')
        plt.plot(stats['test_loss_by_epoch'], label='test_loss')
        plt.plot(stats['test_accuracy'], label='test_accuracy')

        plt.legend()
        plt.show()
