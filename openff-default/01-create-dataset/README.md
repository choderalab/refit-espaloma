## Description

- `Dataset/`: Stores LSF job scripts to convert HDF5 files downloaded from QCArchive `Dataset` into DGL graphs
    - `spice-des-monomers/`
    - `spice-dipeptide/`
    - `spice-pubchem/`
    - `rna-diverse/`
    - `rna-trincleotide/`
    - `rna-nucleoside/`
- `OptimizationDataset/`: Stores LSF job scripts to convert HDF5 files downloaded from QCArchive `OptimizationDataset` into DGL graphs
    - `gen2/`
    - `pepconf-dlc/`
- `TorsionDriveDataset/`: Stores LSF job scripts to convert HDF5 files downloaded from QCArchive `TorsionDriveDataset` into DGL graphs
    - `gen2-torsion/`
    - `protein-torsion/`
- `script/`: Stores main scripts to convert HDF5 file into DGL graphs


## Basic Usage
1. Move to one of the directories (e.g. `Dataset/spice-pubchem/`).
2. Run script to create an LSF submission script that converts record data from an HDF5 file into DGL graphs.
    >./run.sh  

    This will submit LSF jobs and DGL graphs will be stored into `Dataset/spice-pubchem/data/`. Note this process can be simplified using job arrays.
