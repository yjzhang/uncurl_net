from __future__ import print_function

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def plot_df(data, outfile_name, metric='NMI', data_ticks=[0, 1.1, 0.1], log=False):
    labels = data.columns
    pos = range(len(labels))
    plt.cla()
    plt.figure(figsize=(16,7))
    print(data.mean())
    if log:
        plt.yscale('log')
        # for some reason yerr messes with log scale
        plt.bar(pos, data.mean())
    else:
        plt.bar(pos, data.mean(), yerr=2*np.sqrt(data.var())/np.sqrt(10))
    plt.xticks(pos, labels, rotation=75)
    if data_ticks and not log:
        plt.yticks(np.arange(*data_ticks))
    plt.xlabel('Pre-processing method')
    plt.ylabel(metric)
    plt.grid(axis='y')
    plt.title(metric + ' for various preprocessing methods')
    plt.tight_layout()
    if outfile_name is None:
        outfile_name = 'purity.png'
    plt.savefig(outfile_name)


def build_simple_table(filename, outfile_name=None, metric='NMI'):
    data = pd.read_table(filename, index_col=False)
    labels = data.columns
    pos = range(len(labels))
    plt.cla()
    plt.figure(figsize=(16,7))
    print(data.mean())
    plt.bar(pos, data.mean(), yerr=2*np.sqrt(data.var())/np.sqrt(10))
    plt.xticks(pos, labels, rotation=75)
    plt.yticks(np.arange(0, 1.1, 0.1))
    plt.xlabel('Pre-processing method')
    plt.ylabel(metric)
    plt.grid(axis='y')
    plt.title(metric + ' for various preprocessing methods')
    plt.tight_layout()
    if outfile_name is None:
        outfile_name = 'purity_' + filename.split('.')[0] + '.png'
    plt.savefig(outfile_name)
