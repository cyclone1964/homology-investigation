# This version implements a conv/deconv layer to do the detection

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

# This implements the training of a neural net for object recognition
# using the slayerModule as an input layer. After that it is a simple
# FC net with no convolution.
from slayerModule import slayerInput
from slayerDataLoader import SlayerDataset

class ObjectNet(nn.Module):

    def __init__(self,numClasses: int=2):

        # Initialize the parent object
        super(ObjectNet, self).__init__()

        # The input layer takes up to 256 points in 3 space. The data
        # loader laods data directly into this. The output of this is
        # 256 x 1
        self.inputLayer = slayerInput(256,maxPoints = 256)

        # Now the fully connected layerse
        self.fc = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256,128),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(128,64),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(64,16),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(16,numClasses),
            nn.ReLU(inplace=True)
        )

    # Now the forward propagation. This runs the front end, then feeds
    # the output of that to the range and frequency nets, concatenates
    # them, and makes the output.
    def forward(self, x):
        
        # First, lets do the three sets of convolutions
        x = self.inputLayer(x)
        x = self.fc(x)
        return(x)

dataDir = '../Data/TwoClass64'
model = ObjectNet().float()

trainingSet = SlayerDataset(dataDir,partition = 'train')
validationSet = SlayerDataset(dataDir,partition = 'validate')
        
trainingLoader = DataLoader(trainingSet,batch_size = 20)
validationLoader = DataLoader(validationSet,batch_size = 20)

optimizer = torch.optim.SGD(model.parameters(),
                                lr = 0.1,
                                momenteum=0.7)
# Now, we do some training. First, we have the option of loading from
# a checkpoint if we find one: start from the highest numbered one
lastCheckpoint = 0
for r,d,f in os.walk('.'):
    for file in f:
        if 'CheckPoint' in file:
            parts = re.split("[-\.]",file)
            lastCheckpoint = max(lastCheckpoint,int(parts[1]))
            print("Found Checkpoint",lastCheckpoint)
# If we found a checkpoint, load it, otherwise, start from scratch
if (lastCheckpoint > 0):
    fileName = "CheckPoint-" + str(lastCheckpoint) + ".pth"
    print("Loading Checkpoint: ",fileName)
    if (torch.cuda.is_available()):
        checkpoint = torch.load(fileName)
    else:
        checkpoint = torch.load(fileName,map_location = "cpu")

    # Now extract the things from the dicionary
    epoch = checkpoint['epoch']
    model.load_state_dict(checkpoint['state_dict'])
    optim.load_state_dict(checkpoint['optimizer'])
    losses = checkpoint['losses']
    if ('valPerformance' in checkpoint):
        valPerformance = checkpoint['valPerformance']
    else:
        valPerformance = []

else:
    print("Checkpoint Not Found");
    epoch = 0
    losses = []
    valPerformance = []


while (epoch < 5000):
    for X_batch, y_batch in trainingLoader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        y_pred = model.forward(X_batch)
        loss = torch.nn.functional.cross_entropy(y_pred,y_batch)
        losses.append(loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    numTotal = 0
    numCorrect = 0
    for X_val, y_val in validationLoader:
        X_val = X_val.to(device)
        y_val = y_val.to(device)
        y_pred = model.forward(X_val)

        predProbability, predIndices = torch.max(y_pred,dim=1)
        
        states = (torch.eq(predIndices,y_val)).cpu().numpy()
        numCorrect += np.sum(np.where(states,1,0))
        numTotal += states.shape[0]

    print('Loss in Epoch ',epoch,' was ', numCorrect/numTotal)
    # Add this to the performance numbers
    valPerformance.append(float(numCorrect)/float(numTotal))

    # Now, every 1000 or so, we save a checkpoint so that we can
    # restart from there.
    fileName = 'CheckPoint-' + str(epoch) +'.pth'
    state = {'epoch': epoch,
             'losses': losses,
             'valPerformance':valPerformance, 
             'state_dict' : model.state_dict(),
             'optimizer': optim.state_dict()}
    torch.save(state,fileName)
    print('Saved Checkpoint ',fileName,
          ' train: ',trainPerformance[-1],
          ' val:',valPerformance[-1])

    # Next epoch please
    epoch = epoch + 1
    
