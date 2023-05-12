#!/usr/bin/env python
import os, sys, math
import numpy as np
import random
import glob
import shutil
import espaloma as esp


BASE_FORCEFIELD = "openff-2.0.0"
DATASETS = [ "rna-diverse", "rna-trinucleotide" ]
print(esp.__version__)

for DATASET in DATASETS:
    print('# {}'.format(DATASET))
    print('-------------------------')
    input_path = os.path.join(BASE_FORCEFIELD + "_filtered", DATASET)
    files = glob.glob(input_path + "/*")

    ## Get hill formula
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

    output_path = os.path.join(BASE_FORCEFIELD + "_filtered", DATASET + "-group")
    for seq in ["aaa", "aba", "abc"]:
        os.makedirs(os.path.join(output_path, seq), exist_ok=True)

    ## Move each molecule to corresponding group
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

        os.makedirs(os.path.join(output_path, dst, hill_formula), exist_ok=True)    
        for idx in idxs:
            input_prefix = os.path.join(input_path, idx)
            output_prefix = os.path.join(output_path, dst, hill_formula, idx)
            shutil.copytree(input_prefix, output_prefix)
            