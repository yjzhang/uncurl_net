import numpy as np
from scipy import sparse
import torch
import uncurl
from uncurl import experiment_runner

from deep_uncurl_pytorch import UncurlNet
from experiments import UncurlNetRunner


if __name__ == '__main__':
    ################################################ dataset 1: Tasic
    import os
    import pandas as pd
    import scipy.io
    from purity_analysis import plot_df, build_simple_table

    data_counts = pd.read_csv('../uncurl_test_datasets/tasic_allen_brain_map/genes_counts.csv')
    X1 = data_counts.iloc[:,1:].as_matrix()
    X1 = sparse.csc_matrix(X1)
    cell_classification = pd.read_csv('../uncurl_test_datasets/tasic_allen_brain_map/cell_classification.csv')
    actual_labels = cell_classification.primary

    k = 49
    genes = uncurl.max_variance_genes(X1, 5, 0.2)
    data_subset = X1[genes, :]

    log = uncurl.experiment_runner.Log()
    log_norm = uncurl.experiment_runner.LogNorm()
    uncurl_net_runner = UncurlNetRunner(k=k, loss='mse')
    uncurl_runner = experiment_runner.PoissonSE(clusters=k)
    uncurl_net_runner_2_hidden_layers = UncurlNetRunner(k=k, hidden_layers=2, loss='mse', output_names=['UncurlNetW_2_400'])
    uncurl_net_runner_2_hidden_layers_2 = UncurlNetRunner(k=k, hidden_layers=2, loss='mse', n_model_epochs=100, output_names=['UncurlNetW_2_400_100iters'])
    uncurl_net_runner_100_units = UncurlNetRunner(k=k, hidden_units=100, hidden_layers=2, loss='mse', output_names=['UncurlNetW_2_100'])

    vis_dir = 'tasic_vis'
    try:
        os.makedirs(vis_dir)
    except:
        pass

    tsne_km = experiment_runner.TsneKm(n_classes=k)
    #simlr_km = uncurl.experiment_runner.SimlrKm(n_classes=k)
    km = experiment_runner.KM(n_classes=k)
    argmax = experiment_runner.Argmax(n_classes=k)

    methods = [
            ([log, uncurl_net_runner], [argmax, km, tsne_km]),
            ([log_norm, uncurl_net_runner], [argmax, km, tsne_km]),
            ([log_norm, uncurl_net_runner_2_hidden_layers], [argmax, km, tsne_km]),
            ([log_norm, uncurl_net_runner_2_hidden_layers_2], [argmax, km, tsne_km]),
            ([log_norm, uncurl_net_runner_100_units], [argmax, km, tsne_km]),
            #(uncurl_runner, [argmax, km, tsne_km]),
    ]
    print('generating visualizations')
    uncurl.experiment_runner.generate_visualizations(methods, data_subset, actual_labels, base_dir=vis_dir, figsize=(16,9), s=5, alpha=0.5)


    print('running experiments')
    results, names, other = uncurl.experiment_runner.run_experiment(methods, data_subset, k, actual_labels, n_runs=3, use_purity=False, use_nmi=True, consensus=False)
    # save data as tsv
    df = pd.DataFrame(data=results, columns=names)

    tsv_filename = 'nmi_tasic_uncurl_net_2.tsv'.format(data_subset.shape[1], len(genes))
    df.to_csv(tsv_filename, sep='\t', index=False)
    # plot
    build_simple_table(tsv_filename, tsv_filename.split('.')[0]+'.png', metric='NMI')
    # timing
    timing = pd.DataFrame(other['timing'])
    timing_filename = 'timing_' + tsv_filename
    timing.to_csv(timing_filename, sep='\t', index=False)
    timing_outfile = timing_filename.split('.')[0]+'.png'
    plot_df(timing, timing_outfile, metric='Runtime', data_ticks=None, log=False)

    ########################## Dataset 2: Zeisel 

    zeisel_mat = scipy.io.loadmat('../uncurl_test_datasets/zeisel/Zeisel.mat')
    X1 = zeisel_mat['X'].toarray().astype(np.float32).T
    actual_labels = zeisel_mat['true_labs'].flatten()
    k = 9
    genes = uncurl.max_variance_genes(X1, 5, 0.2)
    data_subset = X1[genes, :]

    uncurl_net_runner = UncurlNetRunner(k=k, loss='mse')
    uncurl_runner = experiment_runner.PoissonSE(clusters=k)
    uncurl_net_runner_2_hidden_layers = UncurlNetRunner(k=k, hidden_layers=2, loss='mse', output_names=['UncurlNetW_2_400'])
    uncurl_net_runner_2_hidden_layers_2 = UncurlNetRunner(k=k, hidden_layers=2, loss='mse', n_model_epochs=100, output_names=['UncurlNetW_2_400_100iters'])
    uncurl_net_runner_100_units = UncurlNetRunner(k=k, hidden_units=100, hidden_layers=2, loss='mse', output_names=['UncurlNetW_2_100'])

    vis_dir = 'zeisel_vis'
    try:
        os.makedirs(vis_dir)
    except:
        pass

    tsne_km = experiment_runner.TsneKm(n_classes=k)
    #simlr_km = uncurl.experiment_runner.SimlrKm(n_classes=k)
    km = experiment_runner.KM(n_classes=k)
    argmax = experiment_runner.Argmax(n_classes=k)

    methods = [
            ([log, uncurl_net_runner], [argmax, km, tsne_km]),
            ([log_norm, uncurl_net_runner], [argmax, km, tsne_km]),
            ([log_norm, uncurl_net_runner_2_hidden_layers], [argmax, km, tsne_km]),
            ([log_norm, uncurl_net_runner_2_hidden_layers_2], [argmax, km, tsne_km]),
            ([log_norm, uncurl_net_runner_100_units], [argmax, km, tsne_km]),
            #(uncurl_runner, [argmax, km, tsne_km]),
    ]
    print('generating visualizations')
    uncurl.experiment_runner.generate_visualizations(methods, data_subset, actual_labels, base_dir=vis_dir, figsize=(16,9), s=5, alpha=0.5)


    print('running experiments')
    results, names, other = uncurl.experiment_runner.run_experiment(methods, data_subset, k, actual_labels, n_runs=3, use_purity=False, use_nmi=True, consensus=False)
    # save data as tsv
    df = pd.DataFrame(data=results, columns=names)

    tsv_filename = 'nmi_zeisel_uncurl_net_2.tsv'.format(data_subset.shape[1], len(genes))
    df.to_csv(tsv_filename, sep='\t', index=False)
    # plot
    build_simple_table(tsv_filename, tsv_filename.split('.')[0]+'.png', metric='NMI')
    # timing
    timing = pd.DataFrame(other['timing'])
    timing_filename = 'timing_' + tsv_filename
    timing.to_csv(timing_filename, sep='\t', index=False)
    timing_outfile = timing_filename.split('.')[0]+'.png'
    plot_df(timing, timing_outfile, metric='Runtime', data_ticks=None, log=False)


