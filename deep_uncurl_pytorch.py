import torch
import torch.nn as nn
from torch.nn import functional as F

from uncurl.state_estimation import initialize_means_weights

import numpy as np
import os

# Things to try out:
# - instead of having a encoder-decoder network for M, generate
#   M heuristically - just set it as the weighted mean of the
#   data matrix given W??? Or, use normal UNCURL for M, while keeping
#   the inference network for W???
# - don't use a decoder network - just treat M*W directly as the output.
# - don't use the reparameterization trick - this is just a deep
#   matrix factorization, removing the probabilistic aspects.
# - how do we include priors and side information??? How do we do the equivalent of QualNorm?
#   Of course we can just have an initial M, like with uncurl. But can we do anything more?
#   Can we add an additional objective that represents the constraints? For example, add an objective that indicates how well the inferred M matches with the qualNorm constraints?
# - the noise model is... sort of weird? Maybe we should do the softmax after adding in the noise???
# - do we actually need M? Or can we just have M be the weights of a dense layer? A genes x k layer in the network??? Maybe even the final layer after the reparameterization???


def poisson_loss(outputs, labels):
    """
    Implementation of Poisson loss.

    Basically, it's ||outputs - labels*log(outputs)||
    """
    batch_size = outputs.size()[0]
    log_output = torch.log(outputs)
    return torch.mean(torch.sum(outputs - labels*log_output, 1))


# https://github.com/pytorch/examples/blob/master/vae/main.py
# Reconstruction + KL divergence losses summed over all elements and batch
def loss_function(recon_x, x, mu, logvar):
    BCE = poisson_loss(recon_x, x)
    #BCE = F.binary_cross_entropy(recon_x, x.view(-1, 784), reduction='sum')
    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD


class UncurlNetW(nn.Module):

    def __init__(self, genes, k, M, use_decoder=True,
            use_reparam=True,
            use_m_layer=False):
        """
        This is an autoencoder architecture that learns W.

        Args:
            genes (int): number of genes
            k (int): latent dim (number of clusters)
            M (array): genes x k matrix
            use_decoder (bool): whether or not to use a decoder layer
            use_reparam (bool): whether or not to  use reparameterization trick
            use_m_layer (bool):
        """
        super(UncurlNetW, self).__init__()
        self.genes = genes
        self.k = k
        # M is the output of UncurlNetM?
        self.M = M
        self.use_decoder = use_decoder
        self.use_reparam = use_reparam
        self.fc1 = nn.Linear(genes, 400)
        self.fc21 = nn.Linear(400, k)
        self.fc22 = nn.Linear(400, k)
        if use_m_layer:
            self.m_layer = nn.Linear(k, genes)
        self.fc_dec1 = nn.Linear(genes, 400)
        self.fc_dec2 = nn.Linear(400, genes)

    def encode(self, x):
        # returns two things: mu and logvar
        output = F.relu(self.fc1(x))
        if self.use_reparam:
            return F.softmax(self.fc21(output)), self.fc22(output)
        else:
            return F.softmax(self.fc21(output))

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
        mu = x1
        if self.use_m_layer:
            mu = self.m_layer(x1)
        else:
            mu = torch.matmul(self.M, x1)
        if self.use_reparam:
            z = self.reparameterize(mu, logvar)
            if self.use_decoder:
                return self.decode(z), mu, logvar
            else:
                return z, mu, logvar
        else:
            if self.use_decoder:
                return self.decode(mu)
            else:
                return mu

    def loss(self, output, x, mu=None, logvar=None):
        if self.use_reparam:
            return loss_function(output, x, mu, logvar)
        else:
            return poisson_loss(output, x)


class UncurlNetM(nn.Module):

    def __init__(self, cells, k, W, use_decoder=True, use_reparam=True):
        super(UncurlNetM, self).__init__()
        self.cells = cells
        self.k = k
        self.W = W
        self.use_decoder = use_decoder
        self.use_reparam = use_reparam
        self.fc1 = nn.Linear(cells, 400)
        self.fc21 = nn.Linear(400, k)
        self.fc22 = nn.Linear(400, k)
        self.fc_dec1 = nn.Linear(cells, 400)
        self.fc_dec2 = nn.Linear(400, cells)

    def encode(self, x):
        # returns two things: mu and logvar
        output = F.relu(self.fc1(x))
        if self.use_reparam:
            return F.relu(self.fc21(output)), self.fc22(output)
        else:
            return F.relu(self.fc21(output))

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
        if self.use_reparam:
            z = self.reparameterize(mu, logvar)
            if self.use_decoder:
                return self.decode(z), mu, logvar
            else:
                return z, mu, logvar
        else:
            if self.use_decoder:
                return self.decode(mu)
            else:
                return mu

    def loss(self):
        # TODO
        pass


def train_model(model, epoch):
    model.train()
    train_loss = 0
    for batch_idx, (data, _) in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(data)
        loss = loss_function(recon_batch, data, mu, logvar)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader),
                    loss.item() / len(data)))

    print('====> Epoch: {} Average loss: {:.4f}'.format(
    epoch, train_loss / len(train_loader.dataset)))



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
        self.w_net = UncurlNetW(self.k, self.genes, M)
        self.m_net = UncurlNetM(self.k, self.cells, W)
        # TODO: set device (cpu or gpu), optimizer

    def train(self, n_outer_iters=20, n_epochs=20):
        """
        """

def train(model, device, train_loader, optimizer, **kwargs):
    """
    """
