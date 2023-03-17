#!/usr/bin/env python
import os, sys
import numpy as np
import h5py
import torch
import espaloma as esp
from espaloma.units import *
import click
from openff.toolkit.topology import Molecule
from simtk import unit
from simtk.unit import Quantity



def get_graph(record, key, idx):
    """
    Convert HDF5 entries into dgl graphs.

    NOTES
    Dispersions terms needs to be added to energies and forces for Basic dataset as they are calculated seperately for openff-default QC specification.
    This is not required for Optimization datasets as the dispersions terms are added to energies and forces when qcfractal returns the results.
    """

    # Convert mapped smiles to openff molecules
    try:
        smi = record["smiles"][0].decode('UTF-8')
    except:
        # GEN2 Optimization dataset nests smiles one layer deeper
        smi = record["smiles"][0][0].decode('UTF-8')
    offmol = Molecule.from_mapped_smiles(smi, allow_undefined_stereo=True)

    # Compute AM1-BCC ELF10 using openeye-toolkit
    try:
        offmol.assign_partial_charges(partial_charge_method="am1bccelf10")
    except:
        failures = open('../../partial_charge_failures.txt', 'a')
        failures.write("{}\t{}\t{}\n".format(key, smi, idx))
        failures.close()
        msg = 'could not assign partial charge'
        raise ValueError(msg)
    charges = offmol.partial_charges.value_in_unit(esp.units.CHARGE_UNIT)
    g = esp.Graph(offmol)

    try:
        energy = record["total_energy"]
        grad = record["total_gradient"]
    except:
        energy = []
        for e, e_corr in zip(record["dft_total_energy"], record["dispersion_correction_energy"]):
            energy.append(e + e_corr)
        grad = []
        for gr, gr_corr in zip(record["dft_total_gradient"], record["dispersion_correction_gradient"]):
            grad.append(gr + gr_corr)
    conformations = record["conformations"]

    g.nodes["g"].data["u_ref"] = torch.tensor(
        [
            Quantity(
                _energy,
                esp.units.HARTREE_PER_PARTICLE,
            ).value_in_unit(esp.units.ENERGY_UNIT)
            for _energy in energy
        ],
        #dtype=torch.get_default_dtype(),
        dtype=torch.float64,
    )[None, :]

    g.nodes["n1"].data["xyz"] = torch.tensor(
        np.stack(
            [
                Quantity(
                    xyz,
                    unit.bohr,
                ).value_in_unit(esp.units.DISTANCE_UNIT)
                for xyz in conformations
            ],
            axis=1,
        ),
        requires_grad=True,
        #dtype=torch.get_default_dtype(),
        dtype=torch.float32,
    )

    g.nodes["n1"].data["u_ref_prime"] = torch.stack(
        [
            torch.tensor(
                Quantity(
                    _grad,
                    esp.units.HARTREE_PER_PARTICLE / unit.bohr,
                ).value_in_unit(esp.units.FORCE_UNIT),
                #dtype=torch.get_default_dtype(),
                dtype=torch.float32,
            )
            for _grad in grad
        ],
        dim=1,
    )

    g.nodes['n1'].data['q_ref'] = c = torch.tensor(charges, dtype=torch.float32,).unsqueeze(-1)
    
    return g



def load_from_hdf5(kwargs):
    filename = kwargs["hdf5"]
    key = kwargs["keyname"]
    output_prefix = kwargs["output_prefix"]
    idx = kwargs["index"]

    hdf = h5py.File(filename)
    try:
        record = hdf[key]
    except:
        _key = list(hdf.keys())[int(idx)]
        print("Invalid key ({}). Get key from entry index ({}).".format(key, _key))
        record = hdf[_key]
    
    g = get_graph(record, key, idx)
    g.save(output_prefix)



@click.command()
@click.option("--hdf5", required=True, help='hdf5 filename')
@click.option("--keyname", required=True, help='keyname of the hdf5 group')
@click.option("--output_prefix", required=True, help='output directory to save graph data')
@click.option("--index", required=True, help="key entry id used to load hdf5 if it fails to load from keyname.")
def cli(**kwargs):
    print(kwargs)
    load_from_hdf5(kwargs)



if __name__ == "__main__":
    cli()
