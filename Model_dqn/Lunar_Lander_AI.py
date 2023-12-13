# Importing the required libraries and modules
import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.autograd as autograd
from torch.autograd import Variable
from collections import deque, namedtuple


# Definition of the neural network class
class Network(nn.Module):

    # Constructor to initialize the network architecture
    def __init__(self, state_size, action_size, seed=42):
        super(Network, self).__init__()

        # Setting the seed for reproducibility
        self.seed = torch.manual_seed(seed)

        # Defining the layers of the neural network
        self.fc1 = nn.Linear(state_size, 64)  # Input layer to hidden layer 1
        self.fc2 = nn.Linear(64, 64)          # Hidden layer 1 to hidden layer 2
        self.fc3 = nn.Linear(64, action_size)  # Hidden layer 2 to output layer

    # Forward method to define the forward pass of the network
    def forward(self, state):
        x = self.fc1(state)  # Applying the first linear transformation
        x = F.relu(x)        # Applying the rectified linear unit (ReLU) activation function
        x = self.fc2(x)      # Applying the second linear transformation
        x = F.relu(x)        # Applying ReLU activation again
        return self.fc3(x)   # Output layer without activation (assumed to be a Q-value output for reinforcement learning)


# The code defines a simple neural network class with two hidden layers.
# The forward method specifies the forward pass of the network, and ReLU activation functions are used between layers.

