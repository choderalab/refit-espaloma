## Description

- `Dataset/`: Directory for data downloaded from QCArchive `Dataset` which is a collection of single point quantum mechanical energy calculations performed on a set of molecules.
- `OptimizationDataset/`: Directory for the data downloaded from QCArchive `OptimizationDataset` which is a collection of geometry optimizations performed on a set of molecules.
- `TorsionDriveDataset/`: Directory for the data downloaded from QCArchive `TorsionDriveDataset` which is a collection of torsion scans performed on a set of rotatable torsions for a set of molecules.
- `script/`: Stores main scripts to convert HDF5 data to DGL graphs.


## Basic Usage
1. Move to one of the directories (e.g. `Dataset/spice-pubchem`).
2. Run script to create an LSF submission script to convert each HDF5 record data into DGL graphs.
    >./run.sh  

    This will submit lsf jobs automatically. DGL graphs will be stored in `Dataset/spice-pubchem/data`. Note this process can be simplified using job arrays.
