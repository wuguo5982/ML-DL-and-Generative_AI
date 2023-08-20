import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class CNN(nn.Module):
    def __init__(self, im_size, hidden_dim, kernel_size, n_classes):
        '''
        Create components of a CNN classifier and initialize their weights.

        Arguments:
            im_size (tuple): A tuple of ints with (channels, height, width)
            hidden_dim (int): Number of hidden activations to use
            kernel_size (int): Width and height of (square) convolution filters
            n_classes (int): Number of classes to score
        '''
        super(CNN, self).__init__()
        #############################################################################
        # TODO: Initialize anything you need for the forward pass
        #############################################################################
        (channels, H, W) = im_size
        # Input channels = 3, output channels = 18
        self.conv1 = nn.Conv2d(channels, 18, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        #4608 input features, 64 output features (see sizing flow below)
        self.fc1 = nn.Linear(18 * 16 * 16, 64)
        #64 input features, 10 output features for our 10 defined classes
        self.fc2 = nn.Linear(64, n_classes)
        self.sm2 = nn.Softmax(dim = 1)
        #############################################################################
        #                             END OF YOUR CODE                              #
        #############################################################################

    def forward(self, images):
        '''
        Take a batch of images and run them through the CNN to
        produce a score for each class.

        Arguments:
            images (Variable): A tensor of size (N, C, H, W) where
                N is the batch size
                C is the number of channels
                H is the image height
                W is the image width

        Returns:
            A torch Variable of size (N, n_classes) specifying the score
            for each example and category.
        '''
        scores = None
        #############################################################################
        # TODO: Implement the forward pass. This should take few lines of code.
        #############################################################################
        #Computes the activation of the first convolution
        (N, C, H, W) = images.size()
        x = images
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = x.view(-1, 4608)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        x = self.sm2(x)
        scores = x
        #############################################################################
        #                             END OF YOUR CODE                              #
        #############################################################################
        return scores
