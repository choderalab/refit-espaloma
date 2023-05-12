#!/usr/bin/env python
import os, sys
import numpy as np
import click
import shutil
import glob
import torch
import espaloma as esp
import pandas as pd


BASE_PATH="openff-2.0.0_filtered"
datasets = ['gen2', 'gen2-torsion', 'pepconf-dlc', 'protein-torsion', 'spice-dipeptide', 'spice-pubchem', 'spice-des-monomers']


def _duplicate_smiles(datasets):
    """
    Get unique isomeric smiles (unique molecules) and return duplicated isomeric smiles.

    Returns
    -----
    list : duplicated isomeric smiles
    """

    df = pd.DataFrame(columns=["DATASET", "N_CONFS", "ISOMERIC_SMILES", "NONISOMERIC_SMILES"])
    for dataset in datasets:
        if dataset.startswith('rna'):
            pass
        else:
            ds = esp.data.dataset.GraphDataset.load(os.path.join(BASE_PATH, dataset))
            for g in ds:
                nonisomeric_smi = g.mol.to_smiles(isomeric=False, explicit_hydrogens=False)
                isomeric_smi = g.mol.to_smiles(isomeric=True, explicit_hydrogens=False)
                n_confs = g.nodes['n1'].data['xyz'].shape[1]

                df = df.append(
                    {"DATASET": dataset,
                     "N_CONFS": n_confs, 
                     "ISOMERIC_SMILES": isomeric_smi,
                     "NONISOMERIC_SMILES": nonisomeric_smi,
                    },
                    ignore_index=True
                )
            del ds

            ## Export pandas for each dataset
            df[df["DATASET"] == dataset].to_csv(f"report_{dataset}.csv", sep='\t', index=False)

    ## Unique isomeric smiles
    with open("report_all.log", "w") as wf:
        wf.write("{:20s}\t{:20s}\t{:20s}\t{:20s}\t{:20s}\n".format("DATASET", "TOTAL_CONFS", "TOTAL_MOLS", "UNIQUE_ISOMERIC_SMILES", "UNIQUE_NONISOMERIC_SMILES"))
        for dataset in datasets:
            _df = df[df['DATASET'] == dataset]
            n_mols = len(_df)
            n_isomeric = len(_df.ISOMERIC_SMILES.unique())
            n_nonisomeric = len(_df.NONISOMERIC_SMILES.unique())
            n_confs = _df.N_CONFS.sum()
            wf.write("{:20s}\t{:20d}\t{:20d}\t{:20d}\t{:20d}\n".format(dataset, n_confs, n_mols, n_isomeric, n_nonisomeric))
        wf.write("------------\n")
        n_mols = len(df)
        n_isomeric = len(df.ISOMERIC_SMILES.unique())
        n_nonisomeric = len(df.NONISOMERIC_SMILES.unique())
        wf.write(f"TOTAL NUMBER OF ENTRIES (MOLS): {n_mols}\n")
        wf.write(f"TOTAL NUMBER OF UNIQUE ISOMERIC SMILES: {n_isomeric}\n")
        wf.write(f"TOTAL NUMBER OF UNIQUE NONISOMERIC SMILES: {n_nonisomeric}")

    ## Duplicate isomeric smiles
    duplicate = df[df['ISOMERIC_SMILES'].duplicated(keep=False)]
    duplicate = duplicate.sort_values(by=['ISOMERIC_SMILES'])
    duplicate.to_csv("duplicated_isomeric_smiles.csv", sep='\t', index=False)

    return list(duplicate["ISOMERIC_SMILES"])


def run():
    ## Get duplicated isomeric smiles
    duplicate_smiles = _duplicate_smiles(datasets)
    #print(duplicate_smiles)

    count = 0
    mydict = {}
    for dataset in datasets:
        entry_path = os.path.join(BASE_PATH, dataset)
        paths_to_mydata = glob.glob("{}/*".format(entry_path))
        #print(paths_to_mydata)

        for p in paths_to_mydata:
            g = esp.Graph.load(p)
            s = g.mol.to_smiles(isomeric=True, explicit_hydrogens=False)

            if s in duplicate_smiles:
                ## Replace stereochemistry marker to avoid path separation and other special characters
                _s = s.replace("/", "_").replace("[", "__").replace("]", "__").replace("@", "AT")  
                os.makedirs(os.path.join(BASE_PATH, "duplicated-isomeric-smiles", _s, dataset), exist_ok=True)

                print(f"Duplicated isomeric smiles found: {p} ({s})")
                input_prefix = p
                idx = os.path.basename(p)
                output_prefix = os.path.join(BASE_PATH, "duplicated-isomeric-smiles", _s, dataset, idx)
                print(f"moving {input_prefix} to {output_prefix}")
                shutil.move(input_prefix, output_prefix)
                #shutil.copytree(input_prefix, output_prefix)


if __name__ == '__main__':
    run()