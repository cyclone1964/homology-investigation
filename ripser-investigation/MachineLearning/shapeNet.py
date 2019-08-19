# This version implements a simple neural net to do the classification
# problem of sphere/cube based upon points randomly sampled from their
# surfaces.

# The first thing is to import all the things we need
#
# os, re, and math for file manipulation
# pytorch of course
# numpy for data input manipulations
# matplotlib for plotting intermediates
# shorthands for nn and model_zoo
# The data set and data loader support
import os
import re
import sys
import math
import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# First, let's see if we have a cuda device
if torch.cuda.is_available():
    print("Run on GPU!!");
    device = torch.device("cuda:0")
else:
    print("Run on CPU :(");
    device = torch.device("cpu")

# This defines the number of points (in 3 space) for these shapes.
# This is hard coded into the data sets.
numPoints = 64
numClasses = 2
numDimensions = 3
numDistances = 2016

# Now, we write our own data set class so that we can leverage all
# loader capabilities of pytorch. The structure here is that the
# generated data is in two types of files, named "CubeXXX" and
# "SphereXXX" where XXX runs from 0 to 499. We break this into two
# sets. The first is the training set, which is 800 examples, the
# second is the eval set, which is 200 samples.
class ShapeDataset(Dataset):

    """ Sonar Dataset: for reading my generated Sonar data """
    def __init__(self, root_dir,partition = 'train'):
        """
        Args:
             root_dir (string): path to the data directory
        
        This routine reads in the vector of IDs from the Directory
        and sets up index lists for the various partitions
        """
        self.root_dir = root_dir


        # The way that the files are ordered is that I created a "dictionary",
        # which is a random permutation of all the file indices. If this file
        # exists, we use it. If not, we create it and then use it.
        fileName = root_dir + '/Labels.dat'
        self.labels = np.loadtxt(fileName).astype(int)
        
        # Now split that into training, validation, and test. Training
        # is the first 1/2, validation the next 1/4, and test the rest
        numTrain = math.floor(len(self.labels)/2)
        numVal = math.floor(len(self.labels)/4)
        numTest = numVal
        self.partitions = {'train': range(0,numTrain),
                           'validate': range(numTrain,numTrain+numVal),
                           'test': range(numTrain+numVal,len(self.labels))}

        # Now for this instantiation, we choose one of the partitions
        self.indices = self.partitions[partition]; 

    """
    Return the length of the current partition
    """
    def __len__(self):
        return len(self.indices)

    """ 
    Get an instance of the current partition. The input is taken as an
    index into the currently selected index and the associated
    files are read in.
    """
    def __getitem__(self,idx):

        # Get the file name for this partition location
        index = self.indices[idx]
        fileName = self.root_dir + '/Shape'+ str(index) + '.ldmf'

        # Now load the image, reshape as necessary, and convert to a
        # torch tensor of floats.
        X = np.loadtxt(fileName,delimiter=',')
        X.shape = (1,numDistances)
        X = torch.from_numpy(X).float()
        y = self.labels[index]
        return X, y

# A simple network for classifying points. It is currently two lienar
# lears with a Relu between and a softmax layer
class ShapeNet(nn.Module):

    def __init__(self):

        # Initialize the parent object
        super(ShapeNet, self).__init__()

        self.model = torch.nn.Sequential(
            torch.nn.Linear(numDistances,
                            numDistances),
            torch.nn.ReLU(),
            torch.nn.Linear(numDistances,
                            numClasses),
        )
            
    # Now the forward propagation. This runs the front end, then feeds
    # the output of that to the range and frequency nets, concatenates
    # them, and makes the output.
    def forward(self, x):
        
        # First, lets do the three sets of convolutions
        y = self.model(x)
        return y


# Set up the data set
dataDir = '../Output/MachineLearning'
trainingSet = ShapeDataset(dataDir,partition = 'train')
validationSet = ShapeDataset(dataDir,partition = 'validate')

# Now, let's try to train a network!! First, set up the data loader
# from the training set above with a batch size of 20 for now.
trainingLoader = DataLoader(trainingSet,batch_size = 20)
validationLoader = DataLoader(validationSet,batch_size = 20)


# Now, if there is a checkpoint file provided, load that and re-start.
# Otherwise, reinitialize the model and statistics from scratch
model = ShapeNet().float()
model = model.to(device)
lossFunction = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr = 1e-2)
optimizer.zero_grad()

if len(sys.argv) > 1:
    fileName = sys.argv[1]
    print(' Load Checkpoint: ',fileName)
    if (torch.cuda.is_available()):
        checkpoint = torch.load(fileName)
    else:
        checkpoint = torch.load(fileName,map_location = "cpu")
        
    # Now extract the things from the dicionary
    epoch = checkpoint['epoch']
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    losses = checkpoint['losses']
    if ('valPerformance' in checkpoint):
        valPerformance = checkpoint['valPerformance']
    else:
        valPerformance = []

    if ('trainPerformance' in checkpoint):
        trainPerformance = checkpoint['trainPerformance']
    else:
        trainPerformance = []
else:

    # Create the simpleNet, insuring that it is implemented in floats not
    # doubles since it runs faster.
    print(' Initialize New Run: ',fileName)
    epoch = 0
    losses = []
    valPerformance = []
    trainPerformance = []

    
# Use simple SGD to start

# Now do up to 2000 epochs
while (epoch < 100):

    # For all the batches
    
    numTotal = 0
    numCorrect = 0
    print(' Epoch :',epoch)
    if (epoch == 40):
        print ('New LR 1e-3')
        optimizer = torch.optim.SGD(model.parameters(), lr = 1e-3)
    elif (epoch == 60):
        print ('New LR 1e-4')
        optimizer = torch.optim.SGD(model.parameters(), lr = 1e-4)
    elif (epoch == 80):
        print ('New LR 1e-5')
        optimizer = torch.optim.SGD(model.parameters(), lr = 1e-5)
    elif (epoch == 90):
        print ('New LR 1e-6')
        optimizer = torch.optim.SGD(model.parameters(), lr = 1e-6)

        
    
    for X_batch, y_batch in trainingLoader:

        # Send them to the cuda device if we are using one
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)
        
        # Zero the parameter gradients
        optimizer.zero_grad()
        
        # Do the prediction and add the loss
        y_pred = torch.squeeze(model.forward(X_batch))
        predClass = torch.argmax(y_pred,dim=1)
        loss = lossFunction(y_pred, y_batch)

        # Append the loss for this to the list
        losses.append(loss.item())
        
        # Back propagate
        loss.backward()
        optimizer.step()

        #  The mode is correct iff the bin values match
        states = (torch.eq(predClass,y_batch)).cpu().numpy()
        numTotal += states.shape[0]
        numCorrect += np.sum(np.where(states,1,0))
    trainPerformance.append(float(numCorrect)/float(numTotal))
    
    # Now let us do the validation
    numTotal = 0
    numCorrect = 0
    for X_val, y_val in validationLoader:

        X_val = X_val.to(device)
        y_val = y_val.to(device)
        y_pred = torch.squeeze(model.forward(X_val))
        predClass = torch.argmax(y_pred,dim=1)

        #  The mode is correct iff the bin values match
        states = (torch.eq(predClass,y_val)).cpu().numpy()
        numTotal += states.shape[0]
        numCorrect += np.sum(np.where(states,1,0))
    valPerformance.append(float(numCorrect)/float(numTotal))

    print('Tested Performance: ',
          ' train: ',trainPerformance[-1],
          ' val:',valPerformance[-1],
          ' ',numCorrect,'/',numTotal)

    # Now, every so many, we save a checkpoint so that we can
    # restart from there.
    if (epoch % 10 == 9):
        fileName = 'CheckPoint-' + str(epoch) +'.pth'
        state = {'epoch': epoch,
                 'losses': losses,
                 'trainPerformance':trainPerformance, 
                 'valPerformance':valPerformance, 
                 'state_dict' : model.state_dict(),
                 'optimizer': optimizer.state_dict()}
        torch.save(state,fileName)
        print('Saved Checkpoint ',fileName)

    # Next epoch please
    epoch = epoch + 1

