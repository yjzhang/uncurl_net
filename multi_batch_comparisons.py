# comparing uncurl_net and uncurl

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from uncurl.preprocessing import log1p, cell_normalize
from multi_encoders import UncurlNet as MultiBatchUncurlNet
from deep_uncurl_pytorch import UncurlNet

from sklearn.manifold import TSNE

if __name__ == '__main__':
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
    X_log_norm = log1p(data_total).astype(np.float32)

    """
    # TODO: create networks
    net1 = UncurlNet(X_log_norm, 10,
            use_reparam=False, use_decoder=False,
            use_batch_norm=True,
            hidden_layers=2,
            hidden_units=400,
            loss='mse')

    net1.train_1(X_log_norm, log_interval=10)
    # TODO: test clustering?
    w = net1.w_net.get_w(X_log_norm).transpose(1, 0)
    print(w.argmax(0))
    w = w.numpy().T
    tsne = TSNE(2)
    w_tsne = tsne.fit_transform(w)

    plt.cla()
    plt.scatter(w_tsne[:,0], w_tsne[:,1], c=batch_list)
    plt.savefig('multibatch_no_correction.png')
    """

    net2 = MultiBatchUncurlNet(X_log_norm, 10,
            use_reparam=False, use_decoder=False,
            use_batch_norm=True,
            hidden_layers=2,
            hidden_units=400,
            num_batches=2,
            loss='mse')

    net2.train_1(X_log_norm, batch_list, log_interval=10,
            n_encoder_epochs=0,
            n_model_epochs=70)
    # TODO: test clustering?
    import torch
    w2 = net2.w_net.get_w(X_log_norm, torch.tensor(batch_list)).transpose(1, 0)
    print(w2.argmax(0))
    w2 = w2.numpy()
    tsne = TSNE(2)
    w2_tsne = tsne.fit_transform(w2.T)
    plt.cla()
    plt.scatter(w2_tsne[:,0], w2_tsne[:,1], c=batch_list)
    plt.savefig('multibatch.png')
