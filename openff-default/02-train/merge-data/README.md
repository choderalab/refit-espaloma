## Description
This repository contains scripts to prepare the input DGL graphs used in training espaloma. These include:

- Calculating baseline force field energies and forces
- Filtering high energy conformers
- Seperating duplicated isomeric smiles within different datasets into a new dataset
- Merge duplicated isomeric smiles from different datasets into a single DGL graph

## Manifest
- `script`: Stores main scripts to filter and prepare input DGL graphs in training espaloma
    - `calc_ff.py`: Compute baseline energies and forces using `gaff-1.81`, `gaff-2.11`, `openff-1.2.1`, `openff-2.0.0`, and `Amber ff14SB (RNA.OL3)`
        - `openff-2.0.0` is used to compute the van del Waals parameters
    - `filter.py`: Exclude molecules with a gap between minimum and maximum energy larger than 0.1 Hartree (62.5 kcal/mol)
    - `seperate_duplicated_isomeric_smiles.py`: Seperate DGL graphs of duplicated isomeric smiles into a new directory
    - `merge_graphs.py`: 

## Basic Usage
1. Calculate baseline force field energies and forces.  
    Ex)
    >bsub < calc_ff_gen2.sh
    
    - DGL graphs are saved into a new directory `openff-2.0.0/[dataset]/`
    - Generates log files that summarize the number of conformations found in the datasets

2. Filter high energy conformers. 
    >bsub < filter.sh

    - DGL graphs are saved into a new directory `openff-2.0.0_filtered/[dataset]/`
    - Generates log files that summarize the number of conformations excluded from the datasets

3. Seperate DGL graphs of duplicated isomeric smiles within different datasets into a different directory.
    >bsub < seperate_duplicated_isomeric_smiles.sh

    - Duplicated isomeric smiles within `openff-2.0.0_filtered/[dataset]/` is moved to `openff-2.0.0_filtered/duplicated-isomeric-smiles/`
    - Generates `duplicated_isomeric_smiles.csv` that exports the duplicated isomeric smiles found within the entire dataset
    - Generates `report_all.log` that summarizes the total number of conforamtions, molecules, unique isomeric smiles, and unique non-isomeric smiles within the entire dataset (RNA datasets are excluded since they contain isomerically unique molecules)

4. Merge DGL graphs of duplicated isomeric smiles into a new DGL graph.
    >bsub < merge_graphs.sh

    - Merged DGL graphs are saved into `openff-2.0.0_filtered/duplicated-isomeric-smiles-merge/`
