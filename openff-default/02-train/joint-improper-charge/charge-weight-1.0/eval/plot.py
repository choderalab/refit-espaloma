#!/usr/bin/env python
import os, sys
import glob
import shutil
import pickle
import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
#from matplotlib.ticker import MultipleLocator, FormatStrFormatter, AutoMinorLocator


# Settings
params = {'legend.fontsize': 40, 
          'font.size': 40, 
          'axes.labelsize': 48,
          'axes.titlesize': 48,
          'xtick.labelsize': 40,
          'ytick.labelsize': 40,
          'savefig.dpi': 600, 
          'figure.figsize': [64, 8],
          'xtick.major.size': 10,
          'xtick.minor.size': 7,
          'ytick.major.size': 10,
          'ytick.minor.size': 7}

plt.rcParams.update(params)

import matplotlib.colors as mcolors
mycolors = mcolors.TABLEAU_COLORS


def early_stopping(df):
    """
    Early stopping w/ joint loss
    """
    e_vl = df['e_vl']
    f_vl = df['f_vl']
    vl = e_vl + f_vl

    print(">Early stopping")
    #min_epochs = 60
    min_epochs = 80
    limit = 5
    count = 0
    index = []
    loss_min = vl[min_epochs:].max()
    for i in df.index[min_epochs:]:
        # break
        if count == limit:
            break
        loss_current = vl.iloc[i]
        # reset counter
        if loss_current < loss_min:
            count = 0
            index = [{"epoch": int(df.iloc[i]["epochs"]), "val_loss": loss_current, "e_loss": e_vl.iloc[i], "f_loss": f_vl.iloc[i]}]
            loss_min = loss_current
        # update counter
        else:
            index.append({"epoch": int(df.iloc[i]["epochs"]), "val_loss": loss_current, "e_loss": e_vl.iloc[i], "f_loss": f_vl.iloc[i]})
            count += 1
    print(index)
    epoch_es = index[0]['epoch']
    print(f">Epochs_es: joint_es={epoch_es}")

    source = f"../checkpoints/net{epoch_es}.th"
    destination = f"./net_es_epoch_{epoch_es}.th"
    shutil.copy(source, destination)


def copy_model(df):
    """
    Copy model based on lowest validation loss
    """
    e_vl = df['e_vl']
    f_vl = df['f_vl']
    vl = e_vl + f_vl
    
    # Energy
    index = np.where(e_vl == e_vl.min())
    epoch_emin = int(df.iloc[[index[0][0]]]['epochs'])
    source = f"../checkpoints/net{epoch_emin}.th"
    destination = f"./net_energy_epoch_{epoch_emin}.th"
    shutil.copy(source, destination)

    # Force
    index = np.where(f_vl == f_vl.min())
    epoch_fmin = int(df.iloc[[index[0][0]]]['epochs'])
    source = f"../checkpoints/net{epoch_fmin}.th"
    destination = f"./net_force_epoch_{epoch_fmin}.th"
    shutil.copy(source, destination)
    
    # Joint (energy + force)
    index = np.where(vl == vl.min())
    epoch_min = int(df.iloc[[index[0][0]]]['epochs'])
    source = f"../checkpoints/net{epoch_min}.th"
    destination = f"./net_joint_epoch_{epoch_min}.th"
    shutil.copy(source, destination)

    print(f">Epochs_vl: joint_vl={epoch_min} ({vl.min():.3f}), energy={epoch_emin} ({e_vl.min():.3f}), force={epoch_fmin} ({f_vl.min():.3f})")


def plot(df):
    x = df['epochs']
    e_vl = df['e_vl']
    f_vl = df['f_vl']
    vl = e_vl + f_vl

    f, ax = plt.subplots(figsize=(20, 10))
    ax.plot(x, e_vl, lw=4, label='Energy')
    ax.plot(x, f_vl, lw=4, label='Force')
    ax.plot(x, vl, lw=4, label='Energy+Force')
    
    # x-axis
    ax.set_xlim(0, x.max())
    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())
    
    # y-axis: https://matplotlib.org/3.1.1/gallery/ticks_and_spines/tick-locators.html
    ax.set_ylim(pow(10,0), pow(15,2))
    ax.set_yscale("log", base=10)
    ax.yaxis.set_major_locator(ticker.LogLocator(base=10.0, numticks=15))
    #ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
    
    # label
    ax.set_xlabel("epochs")
    ax.set_ylabel("Loss")
    #ax.set_ylabel(r"RMSE [kcal/mol${\rm \cdot}$${\rm \AA^{-1}}$]")
    #ax.set_ylabel("RMSE")
    
    # legend
    #ax.legend(loc="upper right", bbox_to_anchor=(1.00, 0.35), fontsize=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.00, 1.00), fontsize=28)
    
    plt.tight_layout()
    plt.savefig("rmse.png")
    plt.close()


def run():
    # load all pickle and save as csv
    files = glob.glob('./pkl/*.pickle')
    #files.sort()
    myresults = {}
    for file in files:
        epoch = file.split('/')[-1].split('.')[0]
        epoch = int(epoch)
        with open(file, 'rb') as handle:
            d = pickle.load(handle)
        myresults[epoch] = d
    df = pd.DataFrame.from_dict(myresults).T
    # sort rows based on index (epochs)
    df = df.sort_index(axis=0)
    # reset index (index moved to new column)
    df = df.reset_index()
    # rename column "index" to "epochs"
    df = df.rename({"index": "epochs"}, axis=1)
    df.to_csv('rmse.csv', sep='\t', float_format='%.3f')
    
    plot(df)
    early_stopping(df)
    copy_model(df)
    

if __name__ == "__main__":
    run()
