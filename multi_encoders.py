import torch
import torch.nn as nn
from torch.nn import functional as F
import torch.utils.data

from uncurl.state_estimation import initialize_means_weights

import numpy as np
import os

# A multi-encoder architecture for batch effect correction

EPS = 1e-10

# TODO: implement a sparse Poisson loss
def train_encoder(model, X, output, n_epochs=20, lr=1e-3, weight_decay=0, disp=True,
        device='cpu', log_interval=1, batch_size=0,
        optim=torch.optim.Adam, **kwargs):
    """
    trains an autoencoder network...

    Args:
        n_epochs:
    """
    if batch_size == 0:
        batch_size = max(100, int(X.shape[1]/20))
    data_loader = torch.utils.data.DataLoader(X.T,
            batch_size=batch_size,
            shuffle=True)
    #optimizer = torch.optim.SparseAdam(lr=lr, weight_decay=weight_decay)
    optimizer = torch.optim.Adam(params=model.parameters(),
            lr=lr, weight_decay=weight_decay)
    for epoch in range(n_epochs):
        train_loss = 0.0
        for batch_idx, data in enumerate(data_loader):
            data = data.to(device)
            optimizer.zero_grad()
            if hasattr(model, 'train_batch'):
                loss = model.train_batch(data, optimizer)
            else:
                output = model(data)
                loss = F.mse_loss(output, data)
            if disp and (batch_idx % log_interval == 0):
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        epoch, batch_idx * len(data), len(data_loader.dataset),
                        100. * batch_idx / len(data_loader),
                        loss / len(data)))
            train_loss += loss
        if disp:
            print('====> Epoch: {} Average loss: {:.4f}'.format(
                epoch, train_loss / len(data_loader.dataset)))




class WEncoderMultibatch(nn.Module):

    def __init__(self, genes, k, use_reparam=True, use_batch_norm=True,
            hidden_units=400,
            hidden_layers=1):
        """
        The W Encoder generates W from the data. The MultiBatch encoder has multiple encoder
        layers for different batches.
        """
        super(WEncoderMultibatch, self).__init__()
        self.genes = genes
        self.k = k
        self.use_batch_norm = use_batch_norm
        self.use_reparam = use_reparam
        self.hidden_units = hidden_units
        self.hidden_layers = hidden_layers
        self.fc1 = nn.Linear(genes, hidden_units)
        if use_batch_norm:
            self.bn1 = nn.BatchNorm1d(hidden_units)
        self.additional_layers = nn.ModuleList()
        for i in range(hidden_layers - 1):
            layer = nn.Linear(hidden_units, hidden_units)
            self.additional_layers.append(layer)
            if use_batch_norm:
                self.additional_layers.append(nn.BatchNorm1d(hidden_units))
            self.additional_layers.append(nn.ReLU(True))
        self.fc21 = nn.Linear(hidden_units, k)
        if self.use_reparam:
            self.fc22 = nn.Linear(hidden_units, genes)

    def forward(self, x):
        output = self.fc1(x)
        if self.use_batch_norm:
            output = F.relu(self.bn1(self.fc1(x)))
        else:
            output = F.relu(self.fc1(x))
        if self.hidden_layers > 1:
            for layer in self.additional_layers:
                output = layer(output)
        if self.use_reparam:
            return F.softmax(self.fc21(output)), self.fc22(output)
        else:
            return F.softmax(self.fc21(output)), None
