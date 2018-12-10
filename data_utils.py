# classes for Datasets and various helpers for loading different data formats

from scipy import sparse
import torch.utils.data

class SparseArrayDataset(torch.utils.data.Dataset):

    def __init__(self, mat):
        """
        This Dataset allows the use of sparse matrices...
        by dynamically converting them to dense matrices...

        Args:
            mat (scipy.sparse matrix)
        """
        # TODO

    def __len__(self):
        pass

    def __getitem__(self):
        pass
