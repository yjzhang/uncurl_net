import torch
import torch.nn as nn
from torch.nn import functional as F
import torch.utils.data

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
# - If we use reparameterization, should we do it on MW or just on W? On the one hand, using the reparameterization trick on MW is more like the original uncurl model. On the other hand, that's a lot more computation, and might be less interpretable or more messy. Could we do some kind of "clustering autoencoder"??? Would that even be helpful????

EPS = 1e-10

# TODO: implement a sparse Poisson loss

def poisson_loss(outputs, labels):
    """
    Implementation of Poisson loss.

    Basically, it's ||outputs - labels*log(outputs)||
    """
    log_output = torch.log(outputs)
    # TODO: should this be sum or mean?
    return torch.mean(torch.sum(outputs - labels*log_output, 1))


# https://github.com/pytorch/examples/blob/master/vae/main.py
# Reconstruction + KL divergence losses summed over all elements and batch
def loss_function(recon_x, x, mu, logvar):
    #BCE = poisson_loss(recon_x, x)
    BCE = F.poisson_nll_loss(recon_x, x, log_input=False, full=False, reduction='sum')
    #BCE = F.binary_cross_entropy(recon_x, x.view(-1, 784), reduction='sum')
    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD

#class ArrayDataSet(torch.utils.data.DataSet):

#    def __init__(self):
#        pass

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




class WEncoder(nn.Module):

    def __init__(self, genes, k, use_reparam=True, use_batch_norm=True):
        """
        The W Encoder generates W from the data.
        """
        super(WEncoder, self).__init__()
        self.genes = genes
        self.k = k
        self.use_batch_norm = use_batch_norm
        self.use_reparam = use_reparam
        self.fc1 = nn.Linear(genes, 400)
        if use_batch_norm:
            self.bn1 = nn.BatchNorm1d(400)
        self.fc21 = nn.Linear(400, k)
        #self.bn2 = nn.BatchNorm1d(k)
        self.fc22 = nn.Linear(400, genes)

    def forward(self, x):
        output = self.fc1(x)
        if self.use_batch_norm:
            output = F.relu(self.bn1(self.fc1(x)))
        else:
            output = F.relu(self.fc1(x))
        if self.use_reparam:
            return F.softmax(self.fc21(output)), self.fc22(output)
        else:
            return F.softmax(self.fc21(output)), None

    def train_batch(self, x, optim):
        """
        Trains on a data batch, with the given optimizer...
        """
        # TODO: this won't work - this isn't an autoencoder....
        optim.zero_grad()
        if self.use_reparam:
            output, mu, logvar = self(x)
            loss = loss_function(output, x, mu, logvar)
            loss.backward()
        else:
            output = self(x)
            loss = F.poisson_nll_loss(output, x, log_input=False, full=False, reduction='sum')
            loss.backward()
        optim.step()
        self.clamp_m()
        return loss.item()

class WDecoder(nn.Module):

    def __init__(self, genes, k, use_reparam=True):
        """
        The W Decoder takes M*W, and returns X.
        """
        super(WDecoder, self).__init__()
        self.fc_dec1 = nn.Linear(genes, 400)
        #self.fc_dec2 = nn.Linear(400, 400)
        self.fc_dec3 = nn.Linear(400, genes)

    def forward(self, x):
        output = F.relu(self.fc_dec1(x))
        output = F.relu(self.fc_dec3(output))
        return output



class UncurlNetW(nn.Module):

    def __init__(self, genes, k, M, use_decoder=True,
            use_reparam=True,
            use_m_layer=True,
            use_batch_norm=True,
            **kwargs):
        """
        This is an autoencoder architecture that learns a mapping from
        the data to W.

        Args:
            genes (int): number of genes
            k (int): latent dim (number of clusters)
            M (array): genes x k matrix
            use_decoder (bool): whether or not to use a decoder layer
            use_reparam (bool): whether or not to  use reparameterization trick
            use_m_layer (bool): whether or not to treat M as a differentiable linear layer
        """
        super(UncurlNetW, self).__init__()
        self.genes = genes
        self.k = k
        # M is the output of UncurlNetM?
        self.M = M
        self.use_decoder = use_decoder
        self.use_reparam = use_reparam
        self.use_batch_norm = use_batch_norm
        self.use_m_layer = use_m_layer
        # TODO: add batch norm???
        self.encoder = WEncoder(genes, k, use_reparam, use_batch_norm)
        if use_m_layer:
            self.m_layer = nn.Linear(k, genes, bias=False)
            self.m_layer.weight.data = M#.transpose(0, 1)
        self.fc_dec1 = nn.Linear(genes, 400)
        #self.fc_dec2 = nn.Linear(400, 400)
        self.fc_dec3 = nn.Linear(400, genes)

    def encode(self, x):
        # returns two things: mu and logvar
        return self.encoder(x)

    def decode(self, x):
        output = F.relu(self.fc_dec1(x))
        #output = F.relu(self.fc_dec2(output))
        output = F.relu(self.fc_dec3(output))
        return output

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)

    def forward(self, x):
        x1, logvar = self.encode(x)
        # should be a matrix-vector product
        mu = x1
        if self.use_m_layer:
            mu = F.relu(self.m_layer(x1)) + EPS
        else:
            # TODO: will this preserve the correct dimensions???
            mu = torch.matmul(self.M, x1) + EPS
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

    def clamp_m(self):
        """
        makes all the entries of self.m_layer non-negative.
        """
        w = self.m_layer.weight.data
        w[w<0] = 0
        self.m_layer.weight.data = w

    def train_batch(self, x, optim):
        """
        Trains on a data batch, with the given optimizer...
        """
        optim.zero_grad()
        if self.use_reparam:
            output, mu, logvar = self(x)
            loss = loss_function(output, x, mu, logvar)
            loss.backward()
        else:
            output = self(x)
            loss = F.poisson_nll_loss(output, x, log_input=False, full=False, reduction='sum')
            loss.backward()
        optim.step()
        self.clamp_m()
        return loss.item()

    def get_w(self, X):
        self.eval()
        X_tensor = torch.tensor(X.T, dtype=torch.float32)
        encode_results = self.encode(X_tensor)
        return encode_results[0].detach()
        #data_loader = torch.utils.data.DataLoader(X.T,
        #        batch_size=X.shape[1],
        #        shuffle=False)

    def get_m(self):
        return self.m_layer.weight.data

    def pre_train_encoder(self, W_init):
        """
        pre-trains the encoder to generate W_init
        """
        # TODO

    def pre_train_decoder(self, W_init, X):
        """
        pre-trains the decoder...
        """

class UncurlNet(object):

    def __init__(self, X=None, k=10, genes=0, cells=0, initialization='tsvd', init_m=None, **kwargs):
        """
        UncurlNet can be initialized in two ways:
            - initialize using X, a genes x cells data matrix
            - initialize using genes, cells, init_m (when X is not available)

        Args:
            X: data matrix (can be dense np array or sparse), of shape genes x cells
            k (int): number of clusters (latent dimensionality)
            initialization (str): see uncurl.initialize_means_weights
        """
        if X is not None:
            self.X = X
            self.genes = X.shape[0]
            self.cells = X.shape[1]
            M, W = initialize_means_weights(X, k, initialization=initialization)
            self.M = torch.tensor(M, dtype=torch.float32)
        else:
            self.X = None
            self.genes = genes
            self.cells = cells
            self.M = torch.tensor(init_m, dtype=torch.float32)
        self.k = k
        # initialize M and W using uncurl's initialization
        self.w_net = UncurlNetW(self.genes, self.k, self.M, **kwargs)
        # TODO: set device (cpu or gpu), optimizer

    @classmethod
    def load(path):
        """
        loads an UncurlNet object from file.
        """
        # TODO

    def save(self, path):
        # TODO
        pass

    def preprocess(self):
        """
        Preprocesses the data, converts self.X into a tensor.
        """
        from scipy import sparse
        if sparse.issparse(self.X):
            self.X = sparse.coo_matrix(self.X)
            values = self.X.data
            indices = np.vstack((self.X.row, self.X.col))
            i = torch.LongTensor(indices)
            v = torch.FloatTensor(values)
            self.X = torch.sparse.FloatTensor(i, v, torch.Size(self.X.shape))
        else:
            self.X = torch.tensor(self.X, dtype=torch.float32)

    def train(self, X=None, n_epochs=20, lr=1e-3, weight_decay=0, disp=True,
            device='cpu', log_interval=1, batch_size=0):
        """
        trains the network...

        Args:
            n_epochs:
        """
        if X is not None:
            self.X = X
        if batch_size == 0:
            batch_size = max(100, int(self.X.shape[1]/20))
        data_loader = torch.utils.data.DataLoader(self.X.T,
                batch_size=batch_size,
                shuffle=True)
        #optimizer = torch.optim.SparseAdam(lr=lr, weight_decay=weight_decay)
        optimizer = torch.optim.Adam(params=self.w_net.parameters(),
                lr=lr, weight_decay=weight_decay)
        for epoch in range(n_epochs):
            train_loss = 0.0
            for batch_idx, data in enumerate(data_loader):
                data = data.to(device)
                loss = self.w_net.train_batch(data, optimizer)
                if disp and (batch_idx % log_interval == 0):
                    print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                            epoch, batch_idx * len(data), len(data_loader.dataset),
                            100. * batch_idx / len(data_loader),
                            loss / len(data)))
                train_loss += loss
            if disp:
                print('====> Epoch: {} Average loss: {:.4f}'.format(
                    epoch, train_loss / len(data_loader.dataset)))





if __name__ == '__main__':
    import scipy.io
    mat = scipy.io.loadmat('data/10x_pooled_400.mat')

    uncurl_net = UncurlNet(mat['data'].toarray().astype(np.float32), 8,
            use_reparam=True, use_decoder=True)

    uncurl_net.train(lr=1e-3, n_epochs=250)
    X = uncurl_net.X
    w = uncurl_net.w_net.get_w(X)
    m = uncurl_net.w_net.get_m()
    print(w.argmax(1))
    labels = w.argmax(1).numpy().squeeze()
    actual_labels = mat['labels'].squeeze()
    from sklearn.metrics.cluster import normalized_mutual_info_score as nmi
    print(nmi(labels, actual_labels))

    import uncurl
    m, w = uncurl.poisson_estimate_state(mat['data'], clusters=8)
    print(nmi(actual_labels, w.argmax(1)))
