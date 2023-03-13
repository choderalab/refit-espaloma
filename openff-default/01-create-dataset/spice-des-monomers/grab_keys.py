#!/usr/bin/env python

import os, sys
import numpy as np
import h5py
import click
import shutil



def load_from_hdf5(kwargs):
    filename = kwargs["hdf5"]
    hdf = h5py.File(filename)

    for i, key in enumerate(list(hdf.keys())):
        print(i, key)   # index number is two less than the dl/XXXX.info because of the header and zero-indexing
                


@click.command()
@click.option("--hdf5", default="/home/takabak/data/qca-dataset/openff-default/spice-des-monomers/SPICE-DES-MONOMERS-OPENFF-DEFAULT.hdf5", help='hdf5 filename')
def cli(**kwargs):
    #print(kwargs)
    load_from_hdf5(kwargs)



if __name__ == "__main__":
    cli()