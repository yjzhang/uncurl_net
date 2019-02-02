import os

import numpy as np
from scipy import sparse
import torch
import torch.nn as nn
from torch.nn import functional as F
import torch.utils.data

from uncurl.state_estimation import initialize_means_weights

from nn_utils import BatchDataset, loss_function, ElementWiseLayer,\
        IdentityLayer


# A multi-encoder architecture for batch effect correction

EPS = 1e-10

def multibatch_loss(w_out, batches, n_batches):
    """
    Args:
        w_out (tensor): shape is (cells, k)
        batches (array or tensor): values in [0, n_batches), length=cells
        n_batches (int): number of batches
    """
    # TODO
    # sum(w_out[batches==i].mean() - w_out[batches==0].mean() for i in range(0, n_batches))
    if n_batches <= 1:
        return 0
    batch_0_mean = w_out[batches==0].mean(0)
    return sum(((w_out[batches==i].mean(0) - batch_0_mean)**2).sum() for i in range(1, n_batches))

class WEncoderMultibatch(nn.Module):

    def __init__(self, genes, k,
            num_batches=1,
            use_reparam=True,
            use_batch_norm=True,
            hidden_units=400,
            hidden_layers=1,
            use_shared_softmax=False):
        """
        The W Encoder generates W from the data. The MultiBatch encoder has multiple encoder
        layers for different batches.
        """
        super(WEncoderMultibatch, self).__init__()
        self.genes = genes
        self.k = k
        self.num_batches = num_batches
        self.use_batch_norm = use_batch_norm
        self.use_reparam = use_reparam
        self.hidden_units = hidden_units
        self.hidden_layers = hidden_layers
        self.use_shared_softmax = use_shared_softmax
        # TODO: set  multi-batches
        self.encoder_layers = nn.ModuleList()
        for batch in range(self.num_batches):
            encoder = []
            fc1 = nn.Linear(genes, hidden_units)
            encoder.append(fc1)
            if use_batch_norm:
                bn1 = nn.BatchNorm1d(hidden_units)
                encoder.append(bn1)
            for i in range(hidden_layers - 1):
                layer = nn.Linear(hidden_units, hidden_units)
                encoder.append(layer)
                if use_batch_norm:
                    encoder.append(nn.BatchNorm1d(hidden_units))
            encoder.append(nn.ReLU(True))
            if not use_shared_softmax:
                encoder.append(nn.Linear(hidden_units, k))
            seq = nn.Sequential(*encoder)
            self.encoder_layers.append(seq)
        if use_shared_softmax:
            self.fc21 = nn.Linear(hidden_units, k)
        # TODO: this won't work if use_shared_softmax is False
        if self.use_reparam:
            self.fc22 = nn.Linear(hidden_units, genes)

    def forward(self, x, batches):
        """
        x is a data batch
        batches is a vector of integers with the same length as x,
        indicating the batch from which each data point originates.
        """
        outputs = []
        inverse_indices = np.zeros(x.shape[0], dtype=int)
        num_units = 0
        for i in range(self.num_batches):
            batch_index_i = (batches == i)
            output = x[batch_index_i, :]
            if len(output) == 0:
                continue
            indices = batch_index_i.nonzero().flatten()
            inverse_indices[indices] = range(num_units, num_units + output.shape[0])
            num_units += output.shape[0]
            output = self.encoder_layers[i](output)
            outputs.append(output)
        total_output = torch.cat(outputs)
        total_output = total_output[inverse_indices]
        if self.use_shared_softmax:
            total_output = self.fc21(total_output)
        if self.use_reparam:
            return F.softmax(total_output), self.fc22(total_output)
        else:
            return F.softmax(total_output), None

class WDecoder(nn.Module):

    def __init__(self, genes, k, use_reparam=True, use_batch_norm=True):
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
            use_multibatch_encoder=True,
            use_multibatch_loss=True,
            use_shared_softmax=True,
            multibatch_loss_weight=0.5,
            hidden_units=400,
            hidden_layers=1,
            num_batches=1,
            loss='poisson',
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
            use_batch_norm (bool): whether or not to use batch norm in the encoder
            hidden_units (int): number of hidden units in encoder
            hidden_layers (int): number of hidden layers in encoder
            loss (str): 'poisson', 'l1', or 'mse' - specifies loss function.
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
        self.use_multibatch_encoder = use_multibatch_encoder
        self.use_multibatch_loss = use_multibatch_loss
        self.multibatch_loss_weight = multibatch_loss_weight
        self.loss = loss.lower()
        self.num_batches = num_batches
        if use_multibatch_encoder:
            self.encoder = WEncoderMultibatch(genes, k, num_batches,
                    use_reparam, use_batch_norm,
                    hidden_units=hidden_units, hidden_layers=hidden_layers,
                    use_shared_softmax=use_shared_softmax)
        else:
            from deep_uncurl_pytorch import WEncoder
            self.encoder = WEncoder(genes, k,
                    use_reparam, use_batch_norm,
                    hidden_units=hidden_units, hidden_layers=hidden_layers)
        if use_m_layer:
            self.m_layer = nn.Linear(k, genes, bias=False)
            self.m_layer.weight.data = M#.transpose(0, 1)
        if self.use_decoder:
            self.decoder = WDecoder(genes, k, use_reparam, use_batch_norm)
        else:
            self.decoder = None
        # batch correction layers
        # batch 0 is always the identity layer
        self.correction_layers = nn.ModuleList()
        self.correction_layers.append(IdentityLayer())
        for b in range(num_batches - 1):
            correction = ElementWiseLayer(self.genes)
            #correction = IdentityLayer()
            self.correction_layers.append(correction)

    def encode(self, x, batch):
        # returns two things: mu and logvar
        if self.use_multibatch_encoder:
            return self.encoder(x, batch)
        else:
            return self.encoder(x)

    def decode(self, x):
        return self.decoder(x)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)

    def apply_correction(self, x, batches):
        """
        Applies the batch correction layers...
        """
        # TODO: add a linear correction to w rather than do whatever this is.
        outputs = []
        inverse_indices = np.zeros(x.shape[0], dtype=int)
        num_units = 0
        for i in range(0, self.num_batches):
            batch_index_i = (batches == i)
            output = x[batch_index_i, :]
            if len(output) == 0:
                continue
            indices = batch_index_i.nonzero().flatten()
            inverse_indices[indices] = range(num_units, num_units + output.shape[0])
            num_units += output.shape[0]
            output = self.correction_layers[i](output)
            outputs.append(output)
        total_output = torch.cat(outputs)
        total_output = total_output[inverse_indices]
        return total_output

    def forward(self, x, batch=None):
        if batch is None:
            batch = torch.zeros(x.shape[0], dtype=torch.int)
        w, logvar = self.encode(x, batch)
        # should be a matrix-vector product
        mu = w
        if self.use_m_layer:
            mu = self.m_layer(w) + EPS
        else:
            mu = torch.matmul(self.M, w) + EPS
        # apply batch correction
        mu = self.apply_correction(mu, batch)
        if self.use_reparam:
            z = self.reparameterize(mu, logvar)
            if self.use_decoder:
                return self.decode(z), mu, logvar
            else:
                return z, mu, logvar
        else:
            if self.use_decoder:
                return self.decode(mu), w
            else:
                return mu, w

    def clamp_m(self):
        """
        makes all the entries of self.m_layer non-negative.
        """
        w = self.m_layer.weight.data
        w[w<0] = 0
        self.m_layer.weight.data = w

    def train_batch(self, x, optim, batches=None):
        """
        Trains on a data batch, with the given optimizer...
        """
        optim.zero_grad()
        if self.use_reparam:
            output, mu, logvar = self.forward(x, batches)
            output += EPS
            loss = loss_function(output, x, mu, logvar)
            loss.backward()
        else:
            output, w = self.forward(x, batches)
            output += EPS
            if self.loss == 'poisson':
                loss = F.poisson_nll_loss(output, x, log_input=False, full=True, reduction='sum')
            elif self.loss == 'l1':
                loss = F.l1_loss(output, x, reduction='sum')
            elif self.loss == 'mse':
                loss = F.mse_loss(output, x, reduction='sum')
            if self.use_multibatch_loss:
                loss += self.multibatch_loss_weight*multibatch_loss(w, batches, self.num_batches)
            loss.backward()
        optim.step()
        self.clamp_m()
        return loss.item()

    def get_w(self, X, batches=None):
        """
        X is a dense array or tensor of shape gene x cell.
        """
        self.eval()
        X_tensor = torch.tensor(X.T, dtype=torch.float32)
        encode_results = self.encode(X_tensor, batches)
        return encode_results[0].detach()
        #data_loader = torch.utils.data.DataLoader(X.T,
        #        batch_size=X.shape[1],
        #        shuffle=False)

    def get_m(self):
        return self.m_layer.weight.data


class UncurlNet(object):

    def __init__(self, X=None, k=10, batches=None, genes=0, cells=0, initialization='tsvd', init_m=None, **kwargs):
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
            # TODO: change default initialization??? random initialization???
            if batches is not None and len(batches) == self.cells:
                batches = np.array(batches)
                # only select batch 0?
                X = X[:, batches==0]
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
        # TODO: set device (cpu or gpu), optimizer, # of threads

    def get_w(self, data):
        return self.w_net.get_w(data)

    def get_m(self):
        return self.w_net.get_m()

    def load(self, path):
        """
        loads an UncurlNetW object from file.
        """
        # TODO
        w_net = torch.load(path)
        self.w_net = w_net


    def save(self, path):
        """
        Saves a model to a path...
        """
        # TODO: save only model parameters, or save the whole model?
        torch.save(self.w_net, path)

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

    def pre_train_encoder(self, X=None, batches=None, n_epochs=20, lr=1e-3, weight_decay=0, disp=True,
            device='cpu', log_interval=1, batch_size=0):
        """
        pre-trains the encoder for w_net - fixing M.
        """
        # sets the network to train mode
        self.w_net.train()
        for param in self.w_net.encoder.parameters():
            param.requires_grad = True
        for param in self.w_net.correction_layers.parameters():
            param.requires_grad = True
        for param in self.w_net.m_layer.parameters():
            param.requires_grad = False
        self._train(X, batches, n_epochs, lr, weight_decay, disp, device, log_interval,
                batch_size)

    def train_m(self, X=None, batches=None, n_epochs=20, lr=1e-3, weight_decay=0, disp=True,
            device='cpu', log_interval=1, batch_size=0):
        """
        trains only the m layer.
        """
        self.w_net.train()
        for param in self.w_net.encoder.parameters():
            param.requires_grad = False
        for param in self.w_net.correction_layers.parameters():
            param.requires_grad = False
        for param in self.w_net.m_layer.parameters():
            param.requires_grad = True
        self._train(X, batches, n_epochs, lr, weight_decay, disp, device, log_interval,
                batch_size)

    def train_model(self, X=None, batches=None, n_epochs=20, lr=1e-3, weight_decay=0, disp=True,
            device='cpu', log_interval=1, batch_size=0):
        """
        trains the entire model.
        """
        self.w_net.train()
        for param in self.w_net.encoder.parameters():
            param.requires_grad = True
        for param in self.w_net.correction_layers.parameters():
            param.requires_grad = True
        for param in self.w_net.m_layer.parameters():
            param.requires_grad = True
        self._train(X, batches, n_epochs, lr, weight_decay, disp, device, log_interval,
                batch_size)

    def train_1(self, X=None, batches=None, n_encoder_epochs=20, n_model_epochs=50, **params):
        """
        Trains the model, first fitting the encoder and then fitting both M and
        the encoder.
        """
        self.pre_train_encoder(X, batches, n_epochs=n_encoder_epochs, **params)
        self.train_model(X, batches, n_epochs=n_model_epochs, **params)

    def train_alternating(self, X=None, batches=None, n_outer_iters=10, n_inner_epochs=10, **params):
        """
        Trains the model using alternating minimization, first fitting the W encoder
        and then fitting M.
        """
        for i in range(n_outer_iters):
            self.pre_train_encoder(X, batches, n_epochs=n_inner_epochs, **params)
            self.train_model(X, batches, n_epochs=n_inner_epochs, **params)

    def _train(self, X=None, batches=None, n_epochs=20, lr=1e-3, weight_decay=0, disp=True,
            device='cpu', log_interval=1, batch_size=0):
        """
        trains the w_net...

        Args:
            X (array): genes x cells
            batches (array or list): list of batch indices for each cell
            n_epochs: number of epochs to train for
            lr (float): learning rate
            weight_decay (float)
            disp (bool): whether or not to display outputs
            device (str): cpu or gpu
            log_interval: how often to print log
            batch_size: default is max(100, cells/20)
        """
        if X is not None:
            self.X = X
        if batch_size == 0:
            batch_size = 100
            #batch_size = max(100, int(self.X.shape[1]/20))
        dataset = BatchDataset(X.T, batches)
        data_loader = torch.utils.data.DataLoader(dataset,
                batch_size=batch_size,
                shuffle=True)
        #optimizer = torch.optim.SparseAdam(lr=lr, weight_decay=weight_decay)
        optimizer = torch.optim.Adam(params=self.w_net.parameters(),
                lr=lr, weight_decay=weight_decay)
        for epoch in range(n_epochs):
            train_loss = 0.0
            for batch_idx, data in enumerate(data_loader):
                data, b  = data
                data = data.to(device)
                loss = self.w_net.train_batch(data, optimizer, b)
                if disp and (batch_idx % log_interval == 0):
                    print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                            epoch, batch_idx * len(data), len(data_loader.dataset),
                            100. * batch_idx / len(data_loader),
                            loss / len(data)))
                train_loss += loss
            if disp:
                print('====> Epoch: {} Average loss: {:.4f}'.format(
                    epoch, train_loss / len(data_loader.dataset)))


    def get_mw(self, data):
        """
        Returns a numpy array representing MW.
        """
        # gets MW for data denoising and imputation
        m = self.get_m()
        w = self.get_w(data).transpose(1, 0)
        mw = torch.matmul(m, w)
        return mw.numpy()

if __name__ == '__main__':
    import uncurl
    from uncurl.state_estimation import objective
    from uncurl.preprocessing import cell_normalize, log1p
    import scipy.io
    from sklearn.cluster import KMeans
    from sklearn.metrics.cluster import normalized_mutual_info_score as nmi
    import pandas as pd

    table_seqwell = pd.read_table('../uncurl_test_datasets/batch_effects_seurat/IntegratedAnalysis_ExpressionMatrices/pbmc_SeqWell.expressionMatrix.txt.gz')
    table_10x = pd.read_table('../uncurl_test_datasets/batch_effects_seurat/IntegratedAnalysis_ExpressionMatrices/pbmc_10X.expressionMatrix.txt.gz')

    genes_seqwell = table_seqwell.index
    genes_10x = table_10x.index

    genes_set = set(genes_seqwell).intersection(genes_10x)

    genes_list = list(genes_set)
    data_seqwell = table_seqwell.loc[genes_list].values
    data_10x = table_10x.loc[genes_list].values

    batch_list = [0]*data_seqwell.shape[1]
    batch_list += [1]*data_10x.shape[1]

    data_total = np.hstack([data_seqwell, data_10x])
    X_log_norm = log1p(cell_normalize(data_total)).astype(np.float32)


    net1 = UncurlNet(X_log_norm, 10,
            batches=batch_list,
            use_reparam=False, use_decoder=False,
            use_batch_norm=True,
            hidden_layers=2,
            hidden_units=400,
            num_batches=2,
            loss='mse')

    net1.train_1(X_log_norm, batch_list, log_interval=10,
            batch_size=500)
    # TODO: test clustering?
    w = net1.w_net.get_w(X_log_norm, batch_list)

    # TODO: compare to non-multibatch, run tsne, ...

