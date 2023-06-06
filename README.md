# Refit espaloma with energies and forces
This repository includes scripts to retrain espaloma forcefield 


### Description


### Usage



### Notes



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
    - `02-train/` - Train espaloma
        - `merge-data/` - Scripts used to preprocess dgl graphs prior to training
        - `baseline/` - Scripts used to calculate baseline energies and forces using other forcefields

### CHANGE LOG
- 2023.03.11 `$WORKDIR/rna-espaloma/espaloma-openff-default.3` moved to `$WORKDIR/refit-espaloma/openff-default`
    - Clean-up diretory
- 2023.03.29 `$WORKDIR/rna-espaloma/openff-default` moved to `$WORKDIR/exploring-rna/refit-espaloma/openff-default.arhived`
    - Archived to rerun experiment with improved input features with `espaloma-0.2.4+11`

