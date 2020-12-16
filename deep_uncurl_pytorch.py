import torch
from torch import Tensor
import torch.nn as nn
from torch.nn import functional as F
import torch.utils.data

from uncurl.state_estimation import initialize_means_weights

from nn_utils import loss_function

import numpy as np

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

    def __init__(self, genes, k, use_reparam=True, use_batch_norm=True,
            hidden_units=400,
            hidden_layers=1, device='cpu'):
        """
        The W Encoder generates W from the data.
        """
        super(WEncoder, self).__init__()
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
        if device == 'cuda':
            self.cuda()

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

class WDecoder(nn.Module):

    def __init__(self, genes, k, use_reparam=True, use_batch_norm=True,
            device='cpu'):
        """
        The W Decoder takes M*W, and returns X.
        """
        super(WDecoder, self).__init__()
        self.fc_dec1 = nn.Linear(genes, 400)
        #self.fc_dec2 = nn.Linear(400, 400)
        self.fc_dec3 = nn.Linear(400, genes)
        if device == 'cuda':
            self.cuda()

    def forward(self, x):
        output = F.relu(self.fc_dec1(x))
        output = F.relu(self.fc_dec3(output))
        return output



class UncurlNetW(nn.Module):

    def __init__(self, genes, k, M, use_decoder=True,
            use_reparam=True,
            use_m_layer=True,
            use_batch_norm=True,
            hidden_units=400,
            hidden_layers=1,
            loss='poisson',
            device='cpu',
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
        self.loss = loss.lower()
        # TODO: add batch norm???
        self.encoder = WEncoder(genes, k, use_reparam, use_batch_norm,
                hidden_units=hidden_units, hidden_layers=hidden_layers,
                device=device)
        if use_m_layer:
            self.m_layer = nn.Linear(k, genes, bias=False)
            self.m_layer.weight.data = M#.transpose(0, 1)
        if self.use_decoder:
            self.decoder = WDecoder(genes, k, use_reparam, use_batch_norm,
                    device=device)
        else:
            self.decoder = None
        self.device = device
        if device == 'cuda':
            self.cuda()

    def encode(self, x):
        # returns two things: mu and logvar
        x = x.to(self.device)
        return self.encoder(x)

    def decode(self, x):
        x = x.to(self.device)
        return self.decoder(x)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)

    def forward(self, x):
        x1, logvar = self.encode(x)
        # should be a matrix-vector product
        mu = x1
        if self.use_m_layer:
            mu = self.m_layer(x1) + EPS
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
            output += EPS
            loss = loss_function(output, x, mu, logvar)
            loss.backward()
        else:
            output = self(x) + EPS
            if self.loss == 'poisson':
                loss = F.poisson_nll_loss(output, x, log_input=False, full=True, reduction='sum')
            elif self.loss == 'l1':
                loss = F.l1_loss(output, x, reduction='sum')
            elif self.loss == 'mse':
                loss = F.mse_loss(output, x, reduction='sum')
            loss.backward()
        optim.step()
        self.clamp_m()
        return loss.item()

    def get_w(self, X):
        """
        X is a dense array or tensor of shape gene x cell.
        """
        self.eval()
        X_tensor = torch.tensor(X.T, dtype=torch.float32)
        encode_results = self.encode(X_tensor)
        return encode_results[0].detach()
        #data_loader = torch.utils.data.DataLoader(X.T,
        #        batch_size=X.shape[1],
        #        shuffle=False)

    def get_m(self):
        return self.m_layer.weight.data


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

    def pre_train_encoder(self, X=None, n_epochs=20, lr=1e-3, weight_decay=0, disp=True,
            device='cpu', log_interval=1, batch_size=0):
        """
        pre-trains the encoder for w_net - fixing M.
        """
        self.w_net.train()
        for param in self.w_net.encoder.parameters():
            param.requires_grad = True
        for param in self.w_net.m_layer.parameters():
            param.requires_grad = False
        self._train(X, n_epochs, lr, weight_decay, disp, device, log_interval,
                batch_size)

    def train_m(self, X=None, n_epochs=20, lr=1e-3, weight_decay=0, disp=True,
            device='cpu', log_interval=1, batch_size=0):
        """
        trains only the m layer.
        """
        self.w_net.train()
        for param in self.w_net.encoder.parameters():
            param.requires_grad = False
        for param in self.w_net.m_layer.parameters():
            param.requires_grad = True
        self._train(X, n_epochs, lr, weight_decay, disp, device, log_interval,
                batch_size)

    def train_model(self, X=None, n_epochs=20, lr=1e-3, weight_decay=0, disp=True,
            device='cpu', log_interval=1, batch_size=0):
        """
        trains the entire model.
        """
        self.w_net.train()
        for param in self.w_net.encoder.parameters():
            param.requires_grad = True
        for param in self.w_net.m_layer.parameters():
            param.requires_grad = True
        self._train(X, n_epochs, lr, weight_decay, disp, device, log_interval,
                batch_size)

    def train_1(self, X=None, n_encoder_epochs=20, n_model_epochs=50, **params):
        """
        Trains the model, first fitting the encoder and then fitting both M and
        the encoder.
        """
        self.pre_train_encoder(X, n_epochs=n_encoder_epochs, **params)
        self.train_model(X, n_epochs=n_model_epochs, **params)

    def train_alternating(self, X=None, n_outer_iters=10, n_inner_epochs=10, **params):
        """
        Trains the model using alternating minimization, first fitting the W encoder
        and then fitting M.
        """
        for i in range(n_outer_iters):
            self.pre_train_encoder(X, n_epochs=n_inner_epochs, **params)
            self.train_model(X, n_epochs=n_inner_epochs, **params)

    def _train(self, X=None, n_epochs=20, lr=1e-3, weight_decay=0, disp=True,
            device='cpu', log_interval=1, batch_size=0):
        """
        trains the w_net...

        Args:
            X (array): genes x cells
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

    mat = scipy.io.loadmat('data/10x_pooled_400.mat')
    actual_labels = mat['labels'].squeeze()
    X = mat['data'].toarray().astype(np.float32)
    genes = uncurl.max_variance_genes(X, 5, 0.2)
    X_subset = X[genes,:]

    device = 'cuda'


    X_log_norm = log1p(cell_normalize(X_subset)).astype(np.float32)
    uncurl_net = UncurlNet(X_log_norm, 8,
            use_reparam=False, use_decoder=False,
            use_batch_norm=True,
            hidden_layers=2,
            hidden_units=200,
            loss='mse',
            device=device)
    m_init = torch.tensor(uncurl_net.M)

    uncurl_net.pre_train_encoder(None, lr=1e-3, n_epochs=20,
            log_interval=10, device=device)
    uncurl_net.train_model(None, lr=1e-3, n_epochs=50,
            log_interval=10, device=device)
    w = uncurl_net.w_net.get_w(X_log_norm).transpose(1, 0)
    m = uncurl_net.w_net.get_m()
    mw = torch.matmul(m, w)
    km = KMeans(8)
    print(w.argmax(0))
    w = Tensor.cpu(w)
    m = Tensor.cpu(m)
    mw = Tensor.cpu(mw)
    labels = w.argmax(0).numpy().squeeze()
    labels_km = km.fit_predict(w.transpose(1, 0))
    labels_km_mw = km.fit_predict(mw.transpose(1, 0))
    print('nmi after alternating training:', nmi(labels, actual_labels))
    print('nmi of km(w) after alternating training:', nmi(labels_km, actual_labels))
    print('nmi of km(mw) after alternating training:', nmi(labels_km_mw, actual_labels))
    labels_km_x_subset = km.fit_predict(X_subset.T)
    print('nmi of km(x_subset):', nmi(labels_km_x_subset, actual_labels))
    print('ll of uncurlnet:', objective(X_subset, m.numpy(), w.numpy()))

    m, w, ll = uncurl.poisson_estimate_state(X_subset, clusters=8)
    print(nmi(actual_labels, w.argmax(0)))
    print('ll of uncurl:', ll)

    ############# dataset 2: Zeisel subset

    mat2 = scipy.io.loadmat('data/GSE60361_dat.mat')
    actual_labels = mat2['ActLabs'].squeeze()
    X = mat2['Dat'].astype(np.float32)
    genes = uncurl.max_variance_genes(X, 5, 0.2)
    X_subset = X[genes,:]

    X_log_norm = log1p(cell_normalize(X_subset)).astype(np.float32)
    uncurl_net = UncurlNet(X_log_norm, 7,
            use_reparam=False, use_decoder=False,
            use_batch_norm=True,
            loss='mse',
            device=device)
    m_init = torch.tensor(uncurl_net.M)
    uncurl_net.pre_train_encoder(X_log_norm, lr=1e-3, n_epochs=20,
            log_interval=10, device=device)
    uncurl_net.train_model(X_log_norm, lr=1e-3, n_epochs=50,
            log_interval=10, device=device)
    w = uncurl_net.w_net.get_w(X_log_norm)
    m = uncurl_net.w_net.get_m()
    km = KMeans(7)
    print(w.argmax(1))
    w = Tensor.cpu(w)
    m = Tensor.cpu(m)
    mw = Tensor.cpu(mw)
    labels = w.argmax(1).numpy().squeeze()
    labels_km = km.fit_predict(w)
    print('nmi after alternating training:', nmi(labels, actual_labels))
    print('nmi of km after alternating training:', nmi(labels_km, actual_labels))

    m, w, ll = uncurl.poisson_estimate_state(X_subset, clusters=7)
    print(nmi(actual_labels, w.argmax(0)))


    ############# dataset 3: Zeisel full

    zeisel_mat = scipy.io.loadmat('../uncurl_test_datasets/zeisel/Zeisel.mat')
    zeisel_data = zeisel_mat['data'].toarray().astype(np.float32)
    zeisel_labs = zeisel_mat['labels'].flatten()
    k = len(set(zeisel_labs))

    genes = uncurl.max_variance_genes(zeisel_data, 5, 0.2)
    X_subset = zeisel_data[genes, :]
    X_log_norm = log1p(cell_normalize(X_subset)).astype(np.float32)
    X_norm = cell_normalize(X_subset).astype(np.float32)

    uncurl_net = UncurlNet(X_norm, k,
            use_reparam=False, use_decoder=False,
            use_batch_norm=True,
            device=device)
    m_init = torch.tensor(uncurl_net.M)

    uncurl_net.pre_train_encoder(X_log_norm, lr=1e-3, n_epochs=20,
                log_interval=10, device=device)
    uncurl_net.train_model(X_log_norm, lr=1e-3, n_epochs=50,
                log_interval=10, device=device)


    w = uncurl_net.w_net.get_w(X_log_norm)
    m = uncurl_net.w_net.get_m()
    km = KMeans(k)
    print(w.argmax(1))
    w = Tensor.cpu(w)
    m = Tensor.cpu(m)
    mw = Tensor.cpu(mw)
    labels = w.argmax(1).numpy().squeeze()
    labels_km = km.fit_predict(w)
    print('nmi after alternating training:', nmi(labels, zeisel_labs))
    print('nmi of km after alternating training:', nmi(labels_km, zeisel_labs))

    #m, w, ll = uncurl.poisson_estimate_state(X_subset, clusters=k)
    #print(nmi(zeisel_labs, w.argmax(0)))



    ############# dataset 4: 10x_8k
    data = scipy.io.mmread('../uncurl_test_datasets/10x_pure_pooled/data_8000_cells.mtx.gz')
    data = data.toarray()
    actual_labels = np.loadtxt('../uncurl_test_datasets/10x_pure_pooled/labels_8000_cells.txt').astype(int).flatten()
    genes = uncurl.max_variance_genes(data, 5, 0.2)
    X_subset = data[genes,:]

    X_log_norm = log1p(cell_normalize(X_subset)).astype(np.float32)
    X_norm = cell_normalize(X_subset).astype(np.float32)
    uncurl_net = UncurlNet(X_log_norm, 8,
            use_reparam=False, use_decoder=False,
            use_batch_norm=True,
            hidden_layers=1,
            hidden_units=400,
            #loss='mse',
            device=device)
    m_init = torch.tensor(uncurl_net.M)

    uncurl_net.pre_train_encoder(None, lr=1e-3, n_epochs=20,
            log_interval=10, device=device)
    uncurl_net.train_model(None, lr=1e-3, n_epochs=50,
            log_interval=10, device=device)
    w = uncurl_net.w_net.get_w(X_log_norm).transpose(1, 0)
    m = uncurl_net.w_net.get_m()
    mw = torch.matmul(m, w)
    km = KMeans(8)
    print(w.argmax(0))
    w = Tensor.cpu(w)
    m = Tensor.cpu(m)
    mw = Tensor.cpu(mw)
    labels = w.argmax(0).numpy().squeeze()
    labels_km = km.fit_predict(w.transpose(1, 0))
    labels_km_mw = km.fit_predict(mw.transpose(1, 0))
    print('nmi after alternating training:', nmi(labels, actual_labels))
    print('nmi of km(w) after alternating training:', nmi(labels_km, actual_labels))
    print('nmi of km(mw) after alternating training:', nmi(labels_km_mw, actual_labels))
    labels_km_x_subset = km.fit_predict(X_subset.T)
    print('nmi of km(x_subset):', nmi(labels_km_x_subset, actual_labels))
    print('ll of uncurlnet:', objective(X_subset, m.numpy().astype(np.double), w.numpy().astype(np.double)))




    ############# dataset 5: Tasic




    ############# dataset 6: 

    # TODO: test imputation error as well...


