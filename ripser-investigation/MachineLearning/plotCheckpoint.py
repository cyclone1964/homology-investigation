# A simple script that reads in a checkpoint as I have defined it and
# plots the losses in that checkpoint.
import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt

fileName = sys.argv[1]
checkpoint = torch.load(fileName,map_location="cpu")

losses = checkpoint['losses']
valPerformance = checkpoint['valPerformance']
trainPerformance = checkpoint['trainPerformance']
epoch = checkpoint['epoch']

plt.figure()
plt.plot(np.sqrt(losses))
plt.title(f'{fileName}: shapeNet Loss ({epoch} Epochs)')
plt.xlabel('MiniBatch')
plt.ylabel('CrossEntopyLoss')

plt.figure()
plt.plot(valPerformance,'b')
plt.plot(trainPerformance,'k')
plt.title('shapeNet Probability Correct Classification')
plt.xlabel('Epoch')
plt.ylabel('Probability')
plt.show()

