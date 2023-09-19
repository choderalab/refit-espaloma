#!/usr/bin/env python
import os, sys, math
import numpy as np
import random
import click
import glob
import torch
import espaloma as esp
import dgl
import pandas as pd
from rdkit import Chem
from rdkit.Chem import PandasTools


# Parameters
HARTEE_TO_KCALPERMOL = 627.509
BOHR_TO_ANGSTROMS = 0.529177
RANDOM_SEED = 2666
TRAIN_RATIO, VAL_RATIO, TEST_RATIO = 0.8, 0.1, 0.1


#---------------
# SUBMODULE
#---------------
def load_data(input_prefix, dataset):
    """
    Load dgl graphs
    """
    # Load unique molecules
    print("# LOAD UNIQUE MOLECULES")
    path = os.path.join(input_prefix, dataset)
    
    if dataset == "rna-nucleoside":
        # All dataset as training
        ds_tr = esp.data.dataset.GraphDataset.load(path)
        ds_dummy, ds_vl, ds_te = ds_tr.split([TRAIN_RATIO, VAL_RATIO, TEST_RATIO])
    elif dataset == "rna-trinucleotide":
        # All dataset as test
        ds_te = esp.data.dataset.GraphDataset.load(path)
        ds_tr, ds_vl, ds_dummy = ds_te.split([TRAIN_RATIO, VAL_RATIO, TEST_RATIO])
    else:
        ds = esp.data.dataset.GraphDataset.load(path).shuffle(RANDOM_SEED)
        ds_tr, ds_vl, ds_te = ds.split([TRAIN_RATIO, VAL_RATIO, TEST_RATIO])
    print("Train:Validate:Test = {}:{}:{}".format(len(ds_tr),len(ds_vl),len(ds_te)))

    # Load duplicated molecules
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

    print("Only load dataset from {}.".format(dataset))
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

    return ds_tr, ds_vl, ds_te



def fn(g):
    """
    Remove unnecessarily data from graph
    """
    g.nodes['g'].data.pop('u_qm')
    g.nodes['g'].data.pop('u_gaff-1.81')
    g.nodes['g'].data.pop('u_gaff-2.11')
    g.nodes['g'].data.pop('u_openff-1.2.0')
    g.nodes['g'].data.pop('u_openff-2.0.0')
    try:
        g.nodes['g'].data.pop('u_amber14')
    except:
        pass
    g.nodes['g'].data.pop('u_ref')
    g.nodes['n1'].data.pop('q_ref')        
    g.nodes['g'].data['u_ref_relative'] = g.nodes['g'].data['u_ref_relative'].float()
    return g



def add_grad(g):
    g.nodes["n1"].data["xyz"].requires_grad = True
    return g



def bootstrap_conf(u_ref, u, u_ref_prime, u_prime):
    """
    Bootstrap over conformations
    """
    u_ref = torch.cat(u_ref, dim=0) * HARTEE_TO_KCALPERMOL
    u = torch.cat(u, dim=0) * HARTEE_TO_KCALPERMOL
    u_ref_prime = torch.cat(u_ref_prime, dim=0) * (HARTEE_TO_KCALPERMOL/BOHR_TO_ANGSTROMS)
    u_prime = torch.cat(u_prime, dim=0) * (HARTEE_TO_KCALPERMOL/BOHR_TO_ANGSTROMS)
    
    rmse_e = esp.metrics.rmse(u_ref, u).item()
    rmse_f = esp.metrics.rmse(u_ref_prime, u_prime).item()
    ci_e = esp.metrics.latex_format_ci(
            *esp.metrics.bootstrap(
                esp.metrics.rmse,
                n_samples=1000
            )(
                u_ref,
                u,
            )
        )
    ci_f = esp.metrics.latex_format_ci(
            *esp.metrics.bootstrap(
                esp.metrics.rmse,
                n_samples=1000
            )(
                u_ref_prime,
                u_prime,
            )
        )

    return ci_e, ci_f


def _bootstrap_mol(x, y, n_samples=1000, ci=0.95):
    """
    """
    z = []
    for _x, _y in zip(x, y):
        mse = torch.nn.functional.mse_loss(_x, _y).item()
        z.append(np.sqrt(mse))
    z = np.array(z)

    results = []
    for _ in range(n_samples):
        _z = np.random.choice(z, z.size, replace=True)
        results.append(_z.mean())

    results = np.array(results)
    low = np.percentile(results, 100.0 * 0.5 * (1 - ci))
    high = np.percentile(results, (1 - ((1 - ci) * 0.5)) * 100.0)
    mean = z.mean()

    return mean, low, high


def bootstrap_mol(u_ref, u, u_ref_prime, u_prime):
    """
    Bootstrap over molecules
    """
    mean, low, high = _bootstrap_mol(u_ref, u)
    ci_e = esp.metrics.latex_format_ci(
        mean * HARTEE_TO_KCALPERMOL, 
        low * HARTEE_TO_KCALPERMOL, 
        high * HARTEE_TO_KCALPERMOL
        )

    mean, low, high = _bootstrap_mol(u_ref_prime, u_prime)
    ci_f = esp.metrics.latex_format_ci(
        mean * (HARTEE_TO_KCALPERMOL/BOHR_TO_ANGSTROMS), 
        low * (HARTEE_TO_KCALPERMOL/BOHR_TO_ANGSTROMS), 
        high * (HARTEE_TO_KCALPERMOL/BOHR_TO_ANGSTROMS)
        )

    return ci_e, ci_f


def calc_metric(net, ds, output_path, dataset, suffix):
    """
    """
    u, u_ref, u_rmse = [], [], []
    u_prime, u_ref_prime, u_prime_rmse = [], [], []
    df = pd.DataFrame(columns=["SMILES", "RMSE_ENERGY", "RMSE_FORCE", "n_snapshots"])

    for g in ds:
        #g.heterograph = g.heterograph.to("cuda:0")
        loss = net(g.heterograph)

        # Energy
        _u = (g.nodes['g'].data['u'] - g.nodes['g'].data['u'].mean(dim=-1, keepdims=True)).detach().cpu().flatten()
        _u_ref = g.nodes['g'].data['u_ref_relative'].detach().cpu().flatten()
        u.append(_u)
        u_ref.append(_u_ref)

        # Force
        _u_prime = g.nodes['n1'].data['u_prime'].detach().cpu().flatten()
        _u_ref_prime = g.nodes['n1'].data['u_ref_prime'].detach().cpu().flatten()
        u_prime.append(_u_prime)
        u_ref_prime.append(_u_ref_prime)

        smi = g.mol.to_smiles()
        mol = Chem.MolFromSmiles(smi)

        df = df.append(
            {
                'SMILES': smi,
                'RMSE_ENERGY': esp.metrics.rmse(_u_ref, _u).item() * HARTEE_TO_KCALPERMOL,
                'RMSE_FORCE': esp.metrics.rmse(_u_ref_prime, _u_prime).item() * (HARTEE_TO_KCALPERMOL/BOHR_TO_ANGSTROMS),
                'n_snapshots': g.nodes['n1'].data['xyz'].shape[1]
            },
            ignore_index=True,
        )

    # Export
    df = df.sort_values(by="RMSE_FORCE", ascending=False)
    df.to_csv(f'{output_path}/rmse_{suffix}_{dataset}.csv', sep='\t')
    PandasTools.AddMoleculeColumnToFrame(df, "SMILES", "MOL")
    open(f"{output_path}/rmse_{suffix}_{dataset}.html", "w").write(df.to_html())

    # Bootstrap over conformations
    #ci_e, ci_f = bootstrap_conf(u_ref, u, u_ref_prime, u_prime)

    # Bootstrap over molecule
    ci_e, ci_f = bootstrap_mol(u_ref, u, u_ref_prime, u_prime)

    # Export
    ofile = 'report_summary.csv'
    if os.path.exists(ofile):
        wf = open(ofile, 'a')
    else:
        wf = open(ofile, 'w')

    wf.write(f">{dataset} ({suffix})\n")
    wf.write("----------\n")
    wf.write(f"energy: {ci_e}\n")
    wf.write(f"force: {ci_f}\n")
    wf.write("\n")
    wf.close()



#---------------
# MAIN
#---------------
def run(kwargs):
    """
    """
    # Options
    layer = kwargs['layer']
    units = kwargs['units']
    config = kwargs['config']
    janossy_config = kwargs['janossy_config']
    input_prefix = kwargs['input_prefix']
    dataset = kwargs['dataset']
    best_model = kwargs['best_model']

    # Convert config and janossy_config into list
    _config = []
    for _ in config.split():
        try:
            _config.append(int(_))
        except:
            _config.append(str(_))
    config = _config

    _janossy_config = []
    for _ in janossy_config.split():
        try:
            _janossy_config.append(int(_))
        except:
            _janossy_config.append(str(_))
    janossy_config = _janossy_config

    #
    # Define espaloma model
    #
    layer = esp.nn.layers.dgl_legacy.gn(layer, {"aggregator_type": "mean", "feat_drop": 0.1})
    representation = esp.nn.Sequential(layer, config=config)
    # out_features: Define modular MM parameters Espaloma will assign
    # 1: atom hardness and electronegativity
    # 2: bond linear combination, enforce positive
    # 3: angle linear combination, enforce positive
    # 4: torsion barrier heights (can be positive or negative)
    readout = esp.nn.readout.janossy.JanossyPooling(
        in_features=units, config=janossy_config,
        out_features={
                1: {'s': 1, 'e': 1},
                2: {'log_coefficients': 2},
                3: {'log_coefficients': 2},
                4: {'k': 6},
        },
    )
    readout_improper = esp.nn.readout.janossy.JanossyPoolingWithSmirnoffImproper(in_features=units, config=janossy_config, out_features={"k": 2})

    class ExpCoeff(torch.nn.Module):
        def forward(self, g):
            g.nodes['n2'].data['coefficients'] = g.nodes['n2'].data['log_coefficients'].exp()
            g.nodes['n3'].data['coefficients'] = g.nodes['n3'].data['log_coefficients'].exp()
            return g

    class GetLoss(torch.nn.Module):
        def forward(self, g):

            g.nodes['n1'].data['u_prime'] = torch.autograd.grad(
                g.nodes['g'].data['u'].sum(),
                g.nodes['n1'].data['xyz'],
                create_graph=True,
                retain_graph=True,
                allow_unused=True,
            )[0]

            # Evaluate energy rmse using precalculated refenerence enegy
            return torch.nn.MSELoss()(
                g.nodes['g'].data['u'] - g.nodes['g'].data['u'].mean(dim=-1, keepdims=True),
                g.nodes['g'].data['u_ref_relative'],
            )        

    net = torch.nn.Sequential(
            representation,
            readout,
            readout_improper,
            ExpCoeff(),
            esp.nn.readout.charge_equilibrium.ChargeEquilibrium(),
            esp.mm.geometry.GeometryInGraph(),
            esp.mm.energy.EnergyInGraph(terms=["n2", "n3", "n4", "n4_improper"]),
            GetLoss(),
    )

    state_dict = torch.load(best_model, map_location=torch.device('cpu'))
    net.load_state_dict(state_dict)
    net.eval()
    

    #
    # Load data
    # 

    # Convert datasets into list
    ds_tr, ds_vl, ds_te = load_data(input_prefix, dataset)

    # Remove unnecessary data from graph
    from espaloma.graphs.utils.regenerate_impropers import regenerate_impropers
    ds_tr.apply(fn, in_place=True)
    ds_vl.apply(fn, in_place=True)
    ds_te.apply(fn, in_place=True)
    ds_tr.apply(add_grad)
    ds_vl.apply(add_grad)
    ds_te.apply(add_grad)
    ds_tr.apply(regenerate_impropers, in_place=True)
    ds_vl.apply(regenerate_impropers, in_place=True)
    ds_te.apply(regenerate_impropers, in_place=True)

    #
    # Evaluate energy and force rmse
    #
    output_path = best_model.split('/')[-1].split('.')[0]
    if not os.path.exists(output_path):
        os.mkdir(output_path)

    if dataset == "rna-trinucleotide":
        pass
    else:
        suffix = "tr"
        calc_metric(net, ds_tr, output_path, dataset, suffix)
        del ds_tr

    if dataset in ["rna-nucleoside", "rna-trinucleotide"]:
        pass
    else:
        suffix = "vl"
        calc_metric(net, ds_vl, output_path, dataset, suffix)
        del ds_vl

    if dataset == "rna-nucleoside":
        pass
    else:
        suffix = "te"
        calc_metric(net, ds_te, output_path, dataset, suffix)
        del ds_te



@click.command()
@click.option("-l",   "--layer",           default="SAGEConv", type=click.Choice(["SAGEConv", "GATConv", "TAGConv", "GINConv", "GraphConv"]), help="GNN architecture")
@click.option("-u",   "--units",           default=128, help="GNN layer", type=int)
@click.option("-act", "--activation",      default="relu", type=click.Choice(["relu", "leaky_relu"]), help="activation method")
@click.option("-c",   "--config",          default="128 relu 128 relu 128 relu", help="sequence of numbers (for units) and strings (for activation functions)", type=str)
@click.option("-jc",  "--janossy_config",  default="128 relu 128 relu 128 relu 128 relu", help="sequence of numbers (for units) and strings (for activation functions)", type=str)
@click.option("-i",   "--input_prefix",    default="data", help="input prefix to graph data", type=str)
@click.option("-d",   "--dataset",         help="name of the datasets", type=str)
@click.option("-m",   "--best_model",      required=True, help="best model (e.g. net.th)", type=str)
def cli(**kwargs):
    print(kwargs)
    run(kwargs)



if __name__ == "__main__":
    cli()
