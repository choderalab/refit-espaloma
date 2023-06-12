# Refit espaloma with energies and forces
This repository includes scripts to retrain espaloma forcefield.


### Description


### Usage


### Manifest
- `openff-default/`
    - `01-create-dataset/` - Convert HDF5 data to dgl graphs
        - `Dataset/` - BasicDataset group in QCArchive
            - `spice-des-monomers/`
            - `spice-dipeptide/`
            - `spice-pubchem/`
            - `rna-diverse/`
            - `rna-trincleotide/`
            - `rna-nucleoside/`
        - `OptimizationDataset/` - BasicDataset group in QCArchive
            - `gen2/`
            - `pepconf-dlc/`
        - `TorsionDriveDataset/` - TorsionDriveDataset group in QCArchive
            - `gen2-torsion/`
            - `protein-torsion/`
    - `02-train/` - Refit and evaluate espaloma
        - `baseline/` - Scripts used to calculate baseline energies and forces using other forcefields
        - `joint-improper-charge/charge-weight-1.0/` - Scripts used to refit and evaluate espaloma
        - `merge-data/` - Scripts used to preprocess dgl graphs prior to training

### Notes

