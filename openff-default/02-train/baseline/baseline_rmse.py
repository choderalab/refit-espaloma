import os, sys
import numpy as np
import glob
import random
import click
import espaloma as esp
import torch


# Parameters
HARTEE_TO_KCALPERMOL = 627.5
BOHR_TO_ANGSTROMS = 0.529177
RANDOM_SEED = 2666
TRAIN_RATIO, VAL_RATIO, TEST_RATIO = 0.8, 0.1, 0.1


def run(kwargs):
    input_prefix = kwargs['input_prefix']
    dataset = kwargs['dataset']
    _forcefields = kwargs['forcefields']
    #print("> torch default dtype is now {}".format(torch.get_default_dtype()))

    # Convert forcefields into list
    forcefields = [ str(_) for _ in _forcefields.split() ]


    #
    # Load datasets
    #
    print("# LOAD UNIQUE MOLECULES")
    path = os.path.join(input_prefix, dataset)

    if dataset == "rna-trinucleotide":
        ds = esp.data.dataset.GraphDataset.load(path).shuffle(RANDOM_SEED)
        ds_tr, ds_vl, ds_te = ds.split([0, 0, 1])
    else:
        ds = esp.data.dataset.GraphDataset.load(path).shuffle(RANDOM_SEED)
        ds_tr, ds_vl, ds_te = ds.split([TRAIN_RATIO, VAL_RATIO, TEST_RATIO])
    print("{} molecules found.".format(len(ds)))
    print("Train:Validate:Test = {}:{}:{}".format(len(ds_tr),len(ds_vl),len(ds_te)))

    #
    # Load duplicated molecules
    #
    print("# LOAD DUPLICATED MOLECULES")
    entries = glob.glob(os.path.join(input_prefix, "duplicated-isomeric-smiles-merge", "*"))
    random.seed(RANDOM_SEED)
    random.shuffle(entries)

    n_entries = len(entries)
    entries_tr = entries[:int(n_entries*TRAIN_RATIO)]
    entries_vl = entries[int(n_entries*TRAIN_RATIO):int(n_entries*TRAIN_RATIO)+int(n_entries*VAL_RATIO)]
    entries_te = entries[int(n_entries*TRAIN_RATIO)+int(n_entries*VAL_RATIO):]
    print("Found {} entries. Split data into {}:{}:{} entries.".format(n_entries, len(entries_tr), len(entries_vl), len(entries_te)))
    assert n_entries == len(entries_tr) + len(entries_vl) + len(entries_te)

    print("Load only datasets from {}.".format(dataset))
    for entry in entries_tr:
        _datasets = os.listdir(entry)
        for _dataset in _datasets:
            if _dataset == dataset:
                _ds_tr = esp.data.dataset.GraphDataset.load(os.path.join(entry, _dataset))
                ds_tr += _ds_tr
    for entry in entries_vl:
        _datasets = os.listdir(entry)
        for _dataset in _datasets:
            if _dataset == dataset:
                _ds_vl = esp.data.dataset.GraphDataset.load(os.path.join(entry, _dataset))
                ds_vl += _ds_vl
    for entry in entries_te:
        _datasets = os.listdir(entry)
        for _dataset in _datasets:
            if _dataset == dataset:
                _ds_te = esp.data.dataset.GraphDataset.load(os.path.join(entry, _dataset))
                ds_te += _ds_te
    
    print("Train:Validate:Test = {}:{}:{}".format(len(ds_tr),len(ds_vl),len(ds_te)))


    """
    Remove unnecessary data from graph
    """
    from espaloma.graphs.utils.regenerate_impropers import regenerate_impropers

    def fn(g):
        # remove
        g.nodes['n1'].data.pop('u_ref_prime')
        g.nodes['n1'].data.pop('q_ref')
        g.nodes['g'].data.pop('u_ref')
        g.nodes['g'].data.pop('u_ref_relative')

        # ensure precision match (saved with dtype fp64)
        g.nodes['g'].data['u_qm'] = g.nodes['g'].data['u_qm'].double()
        for forcefield in forcefields:
            g.nodes['g'].data['u_%s' % forcefield] = g.nodes['g'].data['u_%s' % forcefield].double()
        return g

    def add_grad(g):
        g.nodes["n1"].data["xyz"].requires_grad = True
        return g

    ds_tr.apply(fn, in_place=True)
    ds_vl.apply(fn, in_place=True)
    ds_te.apply(fn, in_place=True)
    ds_tr.apply(add_grad)
    ds_vl.apply(add_grad)
    ds_te.apply(add_grad)
    ds_tr.apply(regenerate_impropers, in_place=True)
    ds_vl.apply(regenerate_impropers, in_place=True)
    ds_te.apply(regenerate_impropers, in_place=True)



    """
    Calculate rmse metric
    """
    #
    # train
    #
    print(">train")
    print("-----------")
    import pandas as pd
    df = pd.DataFrame(columns=["SMILES"] + [forcefield + "_ENERGY_RMSE" for forcefield in forcefields] + [forcefield + "_FORCE_RMSE" for forcefield in forcefields])

    # initialize
    us = {"u_qm": [], "u_qm_prime": []}
    for forcefield in forcefields:
        us["u_%s" % forcefield] = []
        us["u_%s_prime" % forcefield] = []

    for g in ds_tr:
        # energy
        u_qm = (g.nodes['g'].data['u_qm'] - g.nodes['g'].data['u_qm'].mean(dim=-1, keepdims=True)).detach().cpu().flatten()
        us["u_qm"].append(u_qm)

        # force 
        u_qm_prime = g.nodes['n1'].data['u_qm_prime'].detach().cpu().flatten()
        us["u_qm_prime"].append(u_qm_prime)

        # base forcefields
        row = {}
        smi = g.mol.to_smiles()
        row["SMILES"] = smi
        for forcefield in forcefields:
            # energy
            u = (g.nodes['g'].data['u_%s' % forcefield] - g.nodes['g'].data['u_%s' % forcefield].mean(dim=-1, keepdims=True)).detach().cpu().flatten()
            us["u_%s" % forcefield].append(u)
            e_rmse = esp.metrics.rmse(u_qm, u) * HARTEE_TO_KCALPERMOL
            row[forcefield + "_ENERGY_RMSE"] = e_rmse.item()

            # force
            u_prime = g.nodes['n1'].data['u_%s_prime' % forcefield].detach().cpu().flatten()
            us["u_%s_prime" % forcefield].append(u_prime)
            f_rmse = esp.metrics.rmse(u_qm_prime, u_prime) * (HARTEE_TO_KCALPERMOL/BOHR_TO_ANGSTROMS)
            row[forcefield + "_FORCE_RMSE"] = f_rmse.item()
        df = df.append(row, ignore_index=True)

    # boostrap energy
    us["u_qm"] = torch.cat(us["u_qm"], dim=0) * HARTEE_TO_KCALPERMOL
    for forcefield in forcefields:
        us["u_%s" % forcefield] = torch.cat(us["u_%s" % forcefield], dim=0) * HARTEE_TO_KCALPERMOL

    print("#energy")
    for forcefield in forcefields:
        print(forcefield)
        print(
            esp.metrics.latex_format_ci(
                *esp.metrics.bootstrap(
                    esp.metrics.rmse,
                    n_samples=1000
                )(
                    us["u_qm"],
                    us["u_%s" % forcefield],
                )
            )
        )

    # boostrap force
    us["u_qm_prime"] = torch.cat(us["u_qm_prime"], dim=0) * (HARTEE_TO_KCALPERMOL/BOHR_TO_ANGSTROMS)
    for forcefield in forcefields:
        us["u_%s_prime" % forcefield] = torch.cat(us["u_%s_prime" % forcefield], dim=0) * (HARTEE_TO_KCALPERMOL/BOHR_TO_ANGSTROMS)

    print("#force")
    for forcefield in forcefields:
        print(forcefield)
        print(
            esp.metrics.latex_format_ci(
                *esp.metrics.bootstrap(
                    esp.metrics.rmse,
                    n_samples=1000
                )(
                    us["u_qm_prime"],
                    us["u_%s_prime" % forcefield],
                )
            )
        )
    print("\n")

    # save
    df = df.sort_values(by="openff-2.0.0_FORCE_RMSE", ascending=False) 
    df.to_csv("inspect_tr.csv")



    #
    # validation
    #
    print(">validation")
    print("-----------")
    import pandas as pd
    df = pd.DataFrame(columns=["SMILES"] + [forcefield + "_ENERGY_RMSE" for forcefield in forcefields] + [forcefield + "_FORCE_RMSE" for forcefield in forcefields])

    # initialize
    us = {"u_qm": [], "u_qm_prime": []}
    for forcefield in forcefields:
        us["u_%s" % forcefield] = []
        us["u_%s_prime" % forcefield] = []

    for g in ds_vl:
        # energy
        u_qm = (g.nodes['g'].data['u_qm'] - g.nodes['g'].data['u_qm'].mean(dim=-1, keepdims=True)).detach().cpu().flatten()
        us["u_qm"].append(u_qm)

        # force 
        u_qm_prime = g.nodes['n1'].data['u_qm_prime'].detach().cpu().flatten()
        us["u_qm_prime"].append(u_qm_prime)

        # base forcefields
        row = {}
        smi = g.mol.to_smiles()
        row["SMILES"] = smi
        for forcefield in forcefields:
            # energy
            u = (g.nodes['g'].data['u_%s' % forcefield] - g.nodes['g'].data['u_%s' % forcefield].mean(dim=-1, keepdims=True)).detach().cpu().flatten()
            us["u_%s" % forcefield].append(u)
            e_rmse = esp.metrics.rmse(u_qm, u) * HARTEE_TO_KCALPERMOL
            row[forcefield + "_ENERGY_RMSE"] = e_rmse.item()

            # force
            u_prime = g.nodes['n1'].data['u_%s_prime' % forcefield].detach().cpu().flatten()
            us["u_%s_prime" % forcefield].append(u_prime)
            f_rmse = esp.metrics.rmse(u_qm_prime, u_prime) * (HARTEE_TO_KCALPERMOL/BOHR_TO_ANGSTROMS)
            row[forcefield + "_FORCE_RMSE"] = f_rmse.item()
        df = df.append(row, ignore_index=True)

    # boostrap energy
    us["u_qm"] = torch.cat(us["u_qm"], dim=0) * HARTEE_TO_KCALPERMOL
    for forcefield in forcefields:
        us["u_%s" % forcefield] = torch.cat(us["u_%s" % forcefield], dim=0) * HARTEE_TO_KCALPERMOL

    print("#energy")
    for forcefield in forcefields:
        print(forcefield)
        print(
            esp.metrics.latex_format_ci(
                *esp.metrics.bootstrap(
                    esp.metrics.rmse,
                    n_samples=1000
                )(
                    us["u_qm"],
                    us["u_%s" % forcefield],
                )
            )
        )

    # boostrap force
    us["u_qm_prime"] = torch.cat(us["u_qm_prime"], dim=0) * (HARTEE_TO_KCALPERMOL/BOHR_TO_ANGSTROMS)
    for forcefield in forcefields:
        us["u_%s_prime" % forcefield] = torch.cat(us["u_%s_prime" % forcefield], dim=0) * (HARTEE_TO_KCALPERMOL/BOHR_TO_ANGSTROMS)

    print("#force")
    for forcefield in forcefields:
        print(forcefield)
        print(
            esp.metrics.latex_format_ci(
                *esp.metrics.bootstrap(
                    esp.metrics.rmse,
                    n_samples=1000
                )(
                    us["u_qm_prime"],
                    us["u_%s_prime" % forcefield],
                )
            )
        )
    print("\n")

    # save
    df = df.sort_values(by="openff-2.0.0_FORCE_RMSE", ascending=False) 
    df.to_csv("inspect_vl.csv")



    #
    # test
    #
    print(">test")
    print("-----------")
    import pandas as pd
    df = pd.DataFrame(columns=["SMILES"] + [forcefield + "_ENERGY_RMSE" for forcefield in forcefields] + [forcefield + "_FORCE_RMSE" for forcefield in forcefields])

    # initialize
    us = {"u_qm": [], "u_qm_prime": []}
    for forcefield in forcefields:
        us["u_%s" % forcefield] = []
        us["u_%s_prime" % forcefield] = []

    for g in ds_te:
        # energy
        u_qm = (g.nodes['g'].data['u_qm'] - g.nodes['g'].data['u_qm'].mean(dim=-1, keepdims=True)).detach().cpu().flatten()
        us["u_qm"].append(u_qm)

        # force 
        u_qm_prime = g.nodes['n1'].data['u_qm_prime'].detach().cpu().flatten()
        us["u_qm_prime"].append(u_qm_prime)

        # base forcefields
        row = {}
        smi = g.mol.to_smiles()
        row["SMILES"] = smi
        for forcefield in forcefields:
            # energy
            u = (g.nodes['g'].data['u_%s' % forcefield] - g.nodes['g'].data['u_%s' % forcefield].mean(dim=-1, keepdims=True)).detach().cpu().flatten()
            us["u_%s" % forcefield].append(u)
            e_rmse = esp.metrics.rmse(u_qm, u) * HARTEE_TO_KCALPERMOL
            row[forcefield + "_ENERGY_RMSE"] = e_rmse.item()

            # force
            u_prime = g.nodes['n1'].data['u_%s_prime' % forcefield].detach().cpu().flatten()
            us["u_%s_prime" % forcefield].append(u_prime)
            f_rmse = esp.metrics.rmse(u_qm_prime, u_prime) * (HARTEE_TO_KCALPERMOL/BOHR_TO_ANGSTROMS)
            row[forcefield + "_FORCE_RMSE"] = f_rmse.item()
        df = df.append(row, ignore_index=True)

    # boostrap energy
    us["u_qm"] = torch.cat(us["u_qm"], dim=0) * HARTEE_TO_KCALPERMOL
    for forcefield in forcefields:
        us["u_%s" % forcefield] = torch.cat(us["u_%s" % forcefield], dim=0) * HARTEE_TO_KCALPERMOL

    print("#energy")
    for forcefield in forcefields:
        print(forcefield)
        print(
            esp.metrics.latex_format_ci(
                *esp.metrics.bootstrap(
                    esp.metrics.rmse,
                    n_samples=1000
                )(
                    us["u_qm"],
                    us["u_%s" % forcefield],
                )
            )
        )

    # boostrap force
    us["u_qm_prime"] = torch.cat(us["u_qm_prime"], dim=0) * (HARTEE_TO_KCALPERMOL/BOHR_TO_ANGSTROMS)
    for forcefield in forcefields:
        us["u_%s_prime" % forcefield] = torch.cat(us["u_%s_prime" % forcefield], dim=0) * (HARTEE_TO_KCALPERMOL/BOHR_TO_ANGSTROMS)

    print("#force")
    for forcefield in forcefields:
        print(forcefield)
        print(
            esp.metrics.latex_format_ci(
                *esp.metrics.bootstrap(
                    esp.metrics.rmse,
                    n_samples=1000
                )(
                    us["u_qm_prime"],
                    us["u_%s_prime" % forcefield],
                )
            )
        )
    print("\n")

    # save
    df = df.sort_values(by="openff-2.0.0_FORCE_RMSE", ascending=False) 
    df.to_csv("inspect_te.csv")



@click.command()
@click.option("-i", "--input_prefix", default="data", help="input prefix to graph data", type=str)
@click.option("-d", "--dataset",      help="name of the dataset", type=str)
@click.option("-f", "--forcefields",  help="baseline forcefields in sequence [gaff-1.81, gaff-2.10, openff-1.2.0, openff-2.0.0, amber14]", type=str)
def cli(**kwargs):
    print(kwargs)
    run(kwargs)



if __name__ == '__main__':
    cli()
