#!/usr/bin/env python
import os, sys
import numpy as np
import h5py
import click
import shutil


def check_files(f, i, key, keylist):
        molfile = os.path.join(f, "mol.json")
        heterograph = os.path.join(f, "heterograph.bin")
        homograph = os.path.join(f, "homograph.bin")
        if os.path.exists(f) and os.path.exists(molfile) and os.path.exists(heterograph) and os.path.exists(homograph) and key not in keylist:
            pass
        elif key in keylist:
            f = os.path.join("data", str(i))
            if os.path.exists(f):
                src = os.path.join("data", str(i))
                dst = os.path.join("data.failure", str(i))
                shutil.move(src, dst)
        else:
            print(i, key)


def load_from_hdf5(kwargs):
    filename = kwargs["hdf5"]
    hdf = h5py.File(filename)

    try:
        failures = open('partial_charge_failures.txt', 'r')
        keylist = [ l.split()[0] for l in failures.readlines() ]
    except:
        keylist = []

    for i, key in enumerate(list(hdf.keys())):
        record = hdf[key]
        try:
            if record["rna_type"][0].decode('UTF-8') == "trinucleotide":
                f = os.path.join("data", str(i), "mydata")
                check_files(f, i, key, keylist)
        except:
            f = os.path.join("data", str(i), "mydata")
            check_files(f, i, key, keylist)


@click.command()
@click.option("--hdf5", required=True, help='hdf5 filename')
def cli(**kwargs):
    #print(kwargs)
    load_from_hdf5(kwargs)


if __name__ == "__main__":
    cli()