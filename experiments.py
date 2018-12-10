import numpy as np
from scipy import sparse
import torch
import uncurl
from uncurl import experiment_runner

from deep_uncurl_pytorch import UncurlNet

class UncurlNetRunner(experiment_runner.Preprocess):

    def __init__(self, k, use_reparam=False, use_decoder=False,
            use_batch_norm=True,
            hidden_units=400,
            hidden_layers=1,
            train_method='1',
            return_mw=False,
            **params):
        self.k = k
        self.use_reparam = use_reparam
        self.use_decoder = use_decoder
        self.use_batch_norm = use_batch_norm
        self.hidden_units = hidden_units
        self.hidden_layers = hidden_layers
        self.train_method = train_method
        self.return_mw = return_mw
        super(UncurlNetRunner, self).__init__(**params)

    def run(self, data):
        if sparse.issparse(data):
            data = data.toarray().astype(np.float32)
        else:
            data = data.astype(np.float32)
        uncurl_net = UncurlNet(data, self.k,
                use_reparam=self.use_reparam,
                use_decoder=self.use_decoder,
                use_batch_norm=self.use_batch_norm,
                hidden_units=self.hidden_units,
                hidden_layers=self.hidden_layers)
        if self.train_method == '1':
            uncurl_net.train_1(data, **self.params)
        elif self.train_method == 'alternating':
            uncurl_net.train_alternating(data, **self.params)
        w = uncurl_net.get_w(data)
        w = w.numpy().T
        if self.return_mw:
            m = uncurl_net.get_m()
            m = m.numpy()
            mw = m.dot(w)
            return [w, mw], 0
        else:
            return w, 0
