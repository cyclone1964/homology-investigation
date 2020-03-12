# This is my copy of the slayer code with my comments in it so that I
# can understand it.
import torch
from torch import Tensor, LongTensor
from torch.tensor import Tensor
from torch.autograd import Variable
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module

# This is the Slayer module, a module for the torch nn layers.
# pytorch modules implement the following methods by definition:
#
# __init__ the class constructor
# forward - the forward propagation
class slayerInput(Module):

    # Implementation of the proposed input layer for multisets defined in ...
    # {
    #   Hofer17c,
    #   author    = {C.~Hofer and R.~Kwitt and M.~Niethammer and A.~Uhl},
    #   title     = {Deep Learning with Topological Signatures},
    #   booktitle = {NIPS},
    #   year      = 2017,
    #   note      = {accepted}
    # }

    # This defines the init function. The inputs a I understand them are:

    # numElements - this is the number of elements in the input layer.
    # point_dimension - the dimensionality of the input points
    # centers_init - the centers of the elements
    # sharpness - the scale of the elements
    def __init__(self,
                 numElements: int,
                 maxPoints: int=256,
                 centers: Tensor=None,
                 sharpness: Tensor=None):
        """
        :param numElements: number of structure elements used
        :param centers: the initialization for the centers of the structure elements
        :param sharpness: initialization for the sharpness of the structure elements
        """
        super(slayerInput, self).__init__()

        # Store some constants
        self.maxPoints = maxPoints
        self.numElements = numElements

        # If no centers or sharpness, load with random numbers/ones
        if centers is None:
            centers = torch.rand(self.numElements, 2)

        if sharpness is None:
            sharpness = torch.ones(self.numElements, 2)*3

        # Mark them as parameters. What this does is enter them as
        # entries in an iterator over tensor parameters and also
        # forces autograd to update them with each iteration.
        self.centers = Parameter(centers)
        self.sharpness = Parameter(sharpness)

    # Forward propagate the loss. In this case, we are implementing
    # something that looks like a fully connected layer of size
    # self.numElements but with a different activation function. 
    def forward(self, input)->Variable:

        # OK ...
        #
        # Starting with the output: we will have a NumBatches x NumElements
        # matrix
        #
        # The input is NumBatches x self.MaxPoints x 3. Select the
        # first two colums please.
        batch = input[:,:,0:2]
        notDummyPoints = input[:,:,2]
        
        # The output of this is NumBatches x (numElements*maxPoints) x 2
        batch = torch.cat([batch] * self.numElements,1);
        notDummyPoints = torch.cat([notDummyPoints] * self.numElements,1)
        
        # This will be the sum of 
        # The input is numBatches x 3 x self.maxPoints. The output
        # will be numBatches x self.numElements. Each of the points
        # goes to each of the elements.
        #
        # To do this we will extract the numBatches x 2 x
        # self.maxPoints elements, subtract off the centers, square
        # the
        
        # centers are 2 x numElements. We need to subtract each of the
        # centers from each of the entry points.
        batchSize = input.shape[0]
        centers = torch.cat([self.centers] * self.maxPoints, 1)
        centers = centers.view(-1,2)
        centers = torch.stack([centers]*batchSize,0)

        sharpness = torch.cat([self.sharpness] * self.maxPoints, 1)
        sharpness = sharpness.view(-1,2)
        sharpness = torch.stack([sharpness]*batchSize,0)
        
        # This does the forward propagation.  NOTE: This does no
        # special math for the processing of the variable "nu" in the
        # paper. I can only figure that the data is pre-processed
        # somehow.
        x = centers - batch
        x = x.pow(2)
        x = torch.mul(x, sharpness)
        x = torch.sum(x, 2)
        x = torch.exp(-x)
        x = torch.mul(x, notDummyPoints)
        x = x.view(batchSize, self.numElements, -1)
        x = torch.sum(x, 2)
        x = x.squeeze()

        return x

    def __str__(self):
        return 'SLayer (... -> {} )'.format(self.numElements)
