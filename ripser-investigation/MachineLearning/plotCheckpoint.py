# A simple script that reads in a checkpoint as I have defined it and
# plots the losses in that checkpoint.
import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt

fileName = 'CheckPoint-' + sys.argv[1] + '.pth'
checkpoint = torch.load(fileName,map_location="cpu")

losses = checkpoint['losses']
valPerformance = checkpoint['valPerformance']
trainPerformance = checkpoint['trainPerformance']
epoch = checkpoint['epoch']

for index in range(len(trainPerformance)):
    trainPerformance[index] = 10*trainPerformance[index]
    valPerformance[index] = 10*valPerformance[index]

plt.figure()
plt.plot(np.sqrt(losses))
plt.title('SonarNet MSE (75 Epochs x 25000 Samples)')
plt.xlabel('MiniBatch')
plt.ylabel('Mean Square Error')

plt.figure()
plt.plot(valPerformance,'b')
plt.plot(trainPerformance,'k')
plt.title('SonarNet Probability Correct Detection')
plt.xlabel('Epoch')
plt.ylabel('Pd')
plt.show()

