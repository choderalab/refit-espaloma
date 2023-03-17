#!/usr/bin/env python
# coding: utf-8
import os, sys, math
import numpy as np
import random
import glob
import shutil
import espaloma as esp


BASE_FORCEFIELD = "openff-2.0.0"
DATASETS = [ "rna-diverse-trinucleotide", "rna-trinucleotide" ]
print(esp.__version__)

for DATASET in DATASETS:
    print('# {}'.format(DATASET))
    print('-------------------------')
    path = os.path.join(BASE_FORCEFIELD + "_filtered", DATASET)
    files = glob.glob(path + "/*")

    # create directories
    for seq in ["aaa", "aba", "abc"]:
        os.makedirs(os.path.join(path, seq), exist_ok=True)


    # get hill formula
    mydict = {}
    for f in files:
        idx = f.split('/')[-1]
        g = esp.Graph.load(f)
        l = g.mol.to_hill_formula()
        if l in mydict:
            mydict[l].append(idx)
        else:
            mydict[l] = [idx]
    #print(mydict)


    # move each molecule to corresponding group
    for hill_formula, idxs in mydict.items():    
        print(hill_formula, idxs)

        if len(idxs) == 1:
            dst = "aaa"
        elif len(idxs) == 3:
            dst = "aba"
        elif len(idxs) == 6:
            dst = "abc"
        else:
            "unexpected number of entries found for {}".format(hill_formula)
            raise ValueError(msg)

        os.makedirs(os.path.join(path, dst, hill_formula), exist_ok=True)    
        for idx in idxs:
            input_prefix = os.path.join(path, idx)
            output_prefix = os.path.join(path, dst, hill_formula, idx)
            shutil.move(input_prefix, output_prefix)
            