# Refit espaloma with energies and forces
This repository includes scripts to retrain and generate `espaloma-0.3.0` forcefield.
`espaloma-0.3.0` force field is a Class I force field where the valence parameters are assigned and optimized via machine learning framework.


## Description
We first convert the downloaded HDF5 obtained from [download-qca-dataset](https://github.com/choderalab/download-qca-datasets) to DGL graphs.
Here, we compute the [AM1BCC-ELF10](https://docs.eyesopen.com/toolkits/python/quacpactk/molchargetheory.html) partial charges using openeye toolkit as a reference.

Molecules with a gap between minimum and maximum energy larger than 0.1 Hartree (62.5 kcal/mol) were excluded from the dataset prior to the refitting experiment, 
similar to the [original paper of Wang et al.](https://pubs.rsc.org/en/content/articlehtml/2022/sc/d2sc02739a). 
Since the van der Waals parameters affect the physical property prediction, which is computationally challenging to optimize, 
we focus on optimizing the valence parameters and use `openff-2.0.0` force field 
(details can be found [here](https://github.com/openforcefield/openff-forcefields)) for the van der Waals paremeters.

Espaloma was trained to minimize the quantum mechanics energies and forces, and also applied L2 regularization to improper and proper torsion force contants. 
Electronegativity and hardness of atoms were predicted to predict the atomic partial charges following the same protocol from the 
[original paper of Wang et al.](https://pubs.rsc.org/en/content/articlehtml/2022/sc/d2sc02739a). 


## Manifest
- `openff-default/`
    - `01-create-dataset/` - Convert HDF5 data to dgl graphs
        - `Dataset/` - Collection of Dataset group in QCArchive
            - `spice-des-monomers/`
            - `spice-dipeptide/`
            - `spice-pubchem/`
            - `rna-diverse/`
            - `rna-trincleotide/`
            - `rna-nucleoside/`
        - `OptimizationDataset/` - Collection of BasicDataset group in QCArchive
            - `gen2/`
            - `pepconf-dlc/`
        - `TorsionDriveDataset/` - Collection of TorsionDriveDataset group in QCArchive
            - `gen2-torsion/`
            - `protein-torsion/`
    - `02-train/` - Refit and evaluate espaloma
        - `baseline/` - Scripts used to calculate baseline energies and forces using other forcefields
        - `joint-improper-charge/charge-weight-1.0/` - Scripts used to refit and evaluate espaloma
        - `merge-data/` - Scripts used to preprocess dgl graphs prior to training
- `envs/` - Stores conda environment files
    - `environment-create-dataset.yaml` - Conda environment used to convert HDF5 data to DGL graphs in `01-create-dataset/`
    - `environment-refit.yaml` - Conda environment to train and evaluate espaloma in `02-train/`

### Dependencies
[Espaloma ver. 0.3.0](https://github.com/choderalab/espaloma/tree/0.3.0) was used to create the DGL graphs in `01-create-dataset/`.
Note that version 0.3.0 is no longer compatible with the 0.2.x models, and vice versa.
A fixed version of 0.3.0 (commit hash:[4c6155b72d00ce0190b3cb551e7e59f0adc33a56](https://github.com/choderalab/espaloma/tree/4c6155b72d00ce0190b3cb551e7e59f0adc33a56)) 
was used for the refitting experinment and model evaluation which allows improper torsions to be fit to n=1,2 phase multiplicity.