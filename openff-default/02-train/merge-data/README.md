## Description
This repository contains scripts used to prepare and filter the input data for training espaloma. This includes:
- calculating baseline force field energies and forces
- filtering high energy conformers
- seperate non-unique smiles (isomeric=False) from other unique entries

## Usage
1. Calculate baseline energy and force using baseline force fields.
    >bsub < calc_ff_gen2.sh
2. Filter high energy conformers.
    >bsub < filter.sh
3. Seperate nonunique isomeric smiles within different datasets into a different directory.
    >bsub < seperate_duplicated_isomeric_smiles.sh
4. Merge the nonunique isomeric smiles from different data source into a single graph object.
    >bsub < merge_graphs.sh
5. (Optional) Split RNA dataset into sequence categories.
    >bsub < split_rna_into_group.sh
    
## Manifest

