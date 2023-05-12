#!/usr/bin/env python
import os, sys
import glob
import copy
import warnings
import torch
import espaloma as esp


# ----
# SUBROUTINE
# ----
def assert_graph(ds):
    """
    Check if graphs are equivalent (Too much inspection but better than less).

    Parameter
    -----
    ds : multiple dgl graphs

    Return
    -----
    """
    for i in range(1, len(ds)):
        ## openff molecule
        assert ds[0].mol == ds[i].mol
        ## mapped isomeric smiles
        assert ds[0].mol.to_smiles(isomeric=True, explicit_hydrogens=True, mapped=True) == ds[i].mol.to_smiles(isomeric=True, explicit_hydrogens=True, mapped=True)
        ## other node data
        for key in ["sum_q"]:
            np.testing.assert_array_equal(ds[0].nodes['g'].data[key].flatten().numpy(), ds[i].nodes['g'].data[key].flatten().numpy())
        for key in ["q_ref", "idxs", "h0"]:
            np.testing.assert_array_equal(ds[0].nodes['n1'].data[key].flatten().numpy(), ds[i].nodes['n1'].data[key].flatten().numpy())


def filter_graph(g):
    """
    Filter high energy conformer and recalculate relative u_ref_energy.

    Parameter
    ----
    g : single dgl graph

    Return
    -----
    g : filtered dgl graph
    """

    ## Relative qm energy
    index1 = g.nodes['g'].data['u_qm'] <= g.nodes['g'].data['u_qm'].min() + MAX_ENERGY
    index1 = index1.flatten()
    ## Relative qm energy after nonbonded interactions are subtracted
    index2 = g.nodes['g'].data['u_ref'] <= g.nodes['g'].data['u_ref'].min() + MAX_ENERGY
    index2 = index2.flatten()        
    
    index = torch.logical_and(index1, index2)
    
    ## Check 
    if len(index) != torch.count_nonzero(index):
        n_false = (len(index) - torch.count_nonzero(index)).item()
        warnings.warn(f"found {n_false} false index")

    ## Filter
    for key in g.nodes['g'].data.keys():
        if key.startswith('u_'):    
            g.nodes['g'].data[key] = g.nodes['g'].data[key][:, index]            
    g.nodes['n1'].data['u_ref_prime'] = g.nodes['n1'].data['u_ref_prime'][:, index, :]
    g.nodes['n1'].data['xyz'] = g.nodes['n1'].data['xyz'][:, index, :]

    ## Recalculate relative u_ref
    g.nodes['g'].data['u_ref_relative'] = g.nodes['g'].data['u_ref_relative'] - g.nodes['g'].data['u_ref_relative'].mean(dim=-1, keepdims=True)

    return g


def update_graph(ds):
    """
    Merge multiple dgl graph in place.

    Parameter
    ----
    ds : multiple dgl graph

    Return
    ----
    g : single dgl graph
    """
    g = copy.deepcopy(ds[0])
    n_confs = g.nodes['n1'].data['xyz'].shape[1]
    print(f"Number of conformation: {n_confs}")

    for key in g.nodes['g'].data.keys():
        if key not in ["sum_q"]:
            #print(key, g.nodes['g'].data[key].shape)
            for i in range(1, len(ds)):
                g.nodes['g'].data[key] = torch.cat((g.nodes['g'].data[key], ds[i].nodes['g'].data[key]), dim=-1)
    for key in g.nodes['n1'].data.keys():
        if key not in ["q_ref", "idxs", "h0"]:
            #print(key, g.nodes['n1'].data[key].shape)
            for i in range(1, len(ds)):
                if key == "xyz":
                    n_confs = ds[i].nodes['n1'].data['xyz'].shape[1]
                    print(f"Number of conformation: {n_confs}")
                g.nodes['n1'].data[key] = torch.cat((g.nodes['n1'].data[key], ds[i].nodes['n1'].data[key]), dim=1)
    
    return g


# ----
# RUN
# ----
MAX_ENERGY = 0.1
entries = glob.glob("./openff-2.0.0_filtered/duplicated-isomeric-smiles/*/*")
n_entries = len(entries)
print(f">{n_entries} entries found.\n")

mydict = {}
for i, entry in enumerate(entries):
    print(f"{i}: {entry}")
    dataset = os.path.basename(entry)
    smile_name = entry.split('/')[-2]
    if smile_name not in mydict.keys():
        mydict[smile_name] = str(i)
    if len(os.listdir(entry)) != 1:
        name = "-".join(os.listdir(entry))
        ## Work around to enable espaloma to save graphs. g.save will not save if file exists.
        _output_prefix = os.path.join("./openff-2.0.0_filtered/duplicated-isomeric-smiles-merge", mydict[smile_name], dataset)
        output_prefix = os.path.join(_output_prefix, name)
        print("Found {} entries. Merge and save as {}.".format(len(os.listdir(entry)), output_prefix))

        ds = esp.data.dataset.GraphDataset.load(entry)
        g = update_graph(ds)
        g = filter_graph(g)
        n_confs = g.nodes['n1'].data['xyz'].shape[1]
        print(f"Total number of conformation: {n_confs}\n")
    else:
        name = os.listdir(entry)[0]
        _output_prefix = os.path.join("./openff-2.0.0_filtered/duplicated-isomeric-smiles-merge", mydict[smile_name], dataset)
        output_prefix = os.path.join(_output_prefix, name)
        print("Found {} entry. Resave as {}.\n".format(len(os.listdir(entry)), output_prefix))
    
    ## Save
    os.makedirs(_output_prefix, exist_ok=True)
    g.save(output_prefix)