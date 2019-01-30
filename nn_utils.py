# classes for Datasets and various helpers for loading different data formats,
# as well as various common functions useful for nn models.

from scipy import sparse

import torch
import torch.utils.data
from torch.nn import functional as F


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


class SparseArrayDataset(torch.utils.data.Dataset):

    def __init__(self, mat):
        """
        This Dataset allows the use of sparse matrices...
        by dynamically converting them to dense matrices...

        Args:
            mat (scipy.sparse matrix) of shape cells x genes
        """
        self.mat = mat

    def __len__(self):
        return self.mat.shape[0]

    def __getitem__(self, idx):
        pass

class BatchDataset(torch.utils.data.Dataset):
    """
    This class defines a dataset that consists of
    a data matrix along with a list of ints indicating
    which batch each cell belongs to.
    """

    def __init__(self, X, batches, **params):
        """
        X should be able to be a dense or sparse array of shape
        cells x genes.

        batches should be a list of ints  in [0, num_batches).
        """
        super(BatchDataset, self).__init__()
        self.data_matrix = X
        self.batches = batches
        self.is_sparse = sparse.issparse(X)

    def __len__(self):
        return self.data_matrix.shape[0]

    def __getitem__(self, idx):
        # returns a tuple (x, batch_id)
        # where x is a tensor and batch_id is an int
        x = self.data_matrix[idx, :]
        if self.is_sparse:
            x = torch.tensor(x.toarray().flatten(), dtype=torch.float32)
        else:
            x = torch.tensor(x, dtype=torch.float32)
        return (x, self.batches[idx])
