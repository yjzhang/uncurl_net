import torch
import torch.nn as nn
from torch.nn import functional as F

from uncurl.state_estimation import initialize_means_weights

import numpy as np
import os

# Things to try out:
# - instead of having a encoder-decoder network for M, generate
#   M heuristically - just set it as the weighted mean of the
#   data matrix given W???
# - iteration order:

def poisson_loss(outputs, correct):
    """
    Implementation of Poisson loss
    """
    batch_size = outputs.size()[0]


class UncurlNetW(nn.Module):

    def __init__(self, genes, k, M):
        super(UncurlNetW, self).__init__()
        self.genes = genes
        self.k = k
        # M is the output of UncurlNetM?
        self.M = M
        self.fc1 = nn.Linear(genes, 400)
        self.fc21 = nn.Linear(400, k)
        self.fc22 = nn.Linear(400, k)
        self.fc_dec1 = nn.Linear(k, 400)
        self.fc_dec2 = nn.Linear(400, genes)

    def encode(self, x):
        # returns two things: mu and logvar
        output = F.relu(self.fc1(x))
        return F.softmax(self.fc21(output)), self.fc22(output)

    def decode(self, x):
        output = F.relu(self.fc_dec1(x))
        output = F.relu(self.fc_dec2(output))
        return output

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)

    def forward(self, x):
        x1, logvar = self.encode(x)
        # should be a matrix-vector product
        # TODO:
        mu = torch.matmul(self.M, x1)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

    def loss(self):
        # TODO: implement Poisson loss
        pass

class UncurlNetM(nn.Module):

    def __init__(self, cells, k, W):
        super(UncurlNetM, self).__init__()
        self.cells = cells
        self.k = k
        self.W = W
        self.fc1 = nn.Linear(cells, 400)
        self.fc21 = nn.Linear(400, k)
        self.fc22 = nn.Linear(400, k)
        self.fc_dec1 = nn.Linear(k, 400)
        self.fc_dec2 = nn.Linear(400, cells)

    def encode(self, x):
        # returns two things: mu and logvar
        output = F.relu(self.fc1(x))
        return F.relu(self.fc21(output)), self.fc22(output)

    def decode(self, x):
        output = F.relu(self.fc_dec1(x))
        output = F.relu(self.fc_dec2(output))
        return output

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)

    def forward(self, x):
        x1, logvar = self.encode(x)
        # should be a matrix-vector product
        mu = torch.matmul(self.W, x1)
        # TODO: do a matrix multiplication???
        # do the reparameterization after the matrix multiplication
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

    def loss(self):
        # TODO
        pass



class UncurlNet(object):

    def __init__(self, X, k, initialization='tsvd'):
        """
        Args:
            X: data matrix (can be dense np array or sparse), of shape genes x cells
            k (int): number of clusters (latent dimensionality)
            initialization (str): see uncurl.initialize_means_weights
        """
        self.X = X
        self.k = k
        self.genes = X.shape[0]
        self.cells = X.shape[1]
        # initialize M and W using uncurl's initialization
        M, W = initialize_means_weights(X, k, initialization=initialization)
        self.M = M
        self.W = W

def train(model, device, train_loader, optimizer, **kwargs):
    """
    """
