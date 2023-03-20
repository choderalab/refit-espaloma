#!/usr/bin/env python
import os, sys
import numpy as np
import click
import copy
import glob
import torch
import espaloma as esp


# ---
# Basic settings
# ---
BASE_FORCEFIELD = "openff-2.0.0"
MAX_ENERGY = 0.1   # hartee (62.75 kcal/mol)
MAX_ENERGY_DIFFERENCE = 0.065   # hartee (40.7875 kcal/mol)
HARTEE_TO_KCALPERMOL = 627.5
MIN_CONF = 5



def run(kwargs):
    dataset = kwargs['dataset']
    entry_path = os.path.join(BASE_FORCEFIELD, dataset)
    paths_to_mydata = glob.glob("{}/*".format(entry_path))

    n_valid_confs = 0
    n_valid_mols = 0
    n_total_confs = 0
    n_total_mols = len(paths_to_mydata)

    with open("calc_ff_{}_filtered.log".format(dataset), "w") as wf:
        wf.write(">{}: {} molecules found\n".format(dataset, n_total_mols))
        for p in paths_to_mydata:
            _g = esp.Graph.load(p)
            g = copy.deepcopy(_g)
            n_confs = g.nodes['n1'].data['xyz'].shape[1]
            
            #
            # Filter high energy conformers and qm/mm inconsistant molecules
            #

            # Relative qm energy
            index1 = g.nodes['g'].data['u_qm'] <= g.nodes['g'].data['u_qm'].min() + MAX_ENERGY
            index1 = index1.flatten()
            #print(index1)

            # Relative qm energy after nonbonded interactions are subtracted
            index2 = g.nodes['g'].data['u_ref'] <= g.nodes['g'].data['u_ref'].min() + MAX_ENERGY
            index2 = index2.flatten()
            #print(index2)
            
            # Relative baseline ff energy
            index3 = g.nodes['g'].data['u_%s' % BASE_FORCEFIELD] <= g.nodes['g'].data['u_%s' % BASE_FORCEFIELD].min() + MAX_ENERGY
            index3 = index3.flatten()
            #print(index3)

            # Energy difference between qm and baseline ff
            _u = g.nodes['g'].data['u_%s' % BASE_FORCEFIELD] - g.nodes['g'].data['u_%s' % BASE_FORCEFIELD].min()
            _u_qm = g.nodes['g'].data['u_qm'] - g.nodes['g'].data['u_qm'].min()
            index4 = abs(_u - _u_qm) <= MAX_ENERGY_DIFFERENCE
            index4 = index4.flatten()
            #print(index4)


            #
            # Get valid conformations that passed the filtering
            #
            index12 = torch.logical_and(index1, index2)
            index34 = torch.logical_and(index3, index4)
            index = torch.logical_and(index12, index34)
            #print(index)

            for key in g.nodes['g'].data.keys():
                if key.startswith('u_'):    
                    g.nodes['g'].data[key] = g.nodes['g'].data[key][:, index]            
            g.nodes['n1'].data['u_ref_prime'] = g.nodes['n1'].data['u_ref_prime'][:, index, :]
            g.nodes['n1'].data['xyz'] = g.nodes['n1'].data['xyz'][:, index, :]

            # Calculate relative u_ref energy by subtracting mean
            g.nodes['g'].data['u_ref_relative'] = g.nodes['g'].data['u_ref'].detach().clone()
            g.nodes['g'].data['u_ref_relative'] = g.nodes['g'].data['u_ref_relative'] - g.nodes['g'].data['u_ref_relative'].mean(dim=-1, keepdims=True)


            #
            # Export
            #
            n_valid_confs += np.array(index, dtype=int).sum()
            n_total_confs += n_confs
            entry_id = int(p.split('/')[-1])
            n_passed_confs = np.array(index, dtype=int).sum()

            if n_passed_confs == n_confs:
                wf.write("{:8d}: {:4d} / {:4d} conformations passed the filter\n".format(entry_id, n_passed_confs, n_confs))
            elif n_passed_confs < MIN_CONF:
                wf.write("{:8d}: {:4d} / {:4d} conformations passed the filter. Number of valid conformations did not pass the threshold. (excluded all)\n".format(entry_id, n_passed_confs, n_confs))
            else:
                wf.write("{:8d}: {:4d} / {:4d} conformations passed the filter ({} excluded)\n".format(entry_id, n_passed_confs, n_confs, n_confs - n_passed_confs))

            if n_passed_confs >= MIN_CONF:
                g.save('{}/{}/{}'.format(BASE_FORCEFIELD + "_filtered", dataset, entry_id))
                n_valid_mols += 1

        wf.write("------------------\n")
        wf.write(">total molecules: {}\n".format(n_total_mols))
        wf.write(">total conformations: {}\n".format(n_total_confs))
        wf.write(">total valid molecules: {}\n".format(n_valid_mols))
        wf.write(">total valid conformations: {}".format(n_valid_confs))
        


@click.command()
@click.option("--dataset",  required=True, type=click.Choice(['gen2', 'pepconf', 'pepconf-dlc', 'rna-diverse-trinucleotide', 'rna-trinucleotide', 'rna-nucleoside', 'spice-dipeptide', 'spice-pubchem', 'spice-des-monomers']), help="name of the dataset")
def cli(**kwargs):
    print(kwargs)
    print(esp.__version__)
    run(kwargs)



if __name__ == '__main__':
    cli()