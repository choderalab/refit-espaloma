## Description

- `merge-data/` - Filter and merge DGL graphs
    - `script/` - Stores scripts to filter and merge DGL graphs from different data sources
- `joint-improper-charge/charge-weight-1.0/` - Refit and evaluate espaloma
    - `eval/` - Inspect validation loss and choose espaloma model
    - `metric/` - Compute energy and force RMSE metrics
    - 
- `baseline/` - Compute baseline energies and forces against various data sources
    - `gen2-torsion/` - OpenFF Gen2 Torsion datasets from QCArchive TorsionDriveDataset
    - `gen2/` - OpenFF Gen2 Optimization datasets from QCArchive OptimizationDataset
    - `pepconf-dlc/` - Pepconf Optimization dataset from QCArchive OptimizationDataset
    - `protein-torsion/` - OpenFF Protein backbone and sidechain torsions from QCArchive TorsionDriveDataset
    - `rna-diverse/` - RNA-Diverse dataset from QCArchive Dataset
    - `rna-nucloeside/` - RNA-Nucleoside dataset from QCArchive Dataset
    - `rna-trinucleotide/` - RNA-Trinucleotide dataset from QCArchive Dataset
    - `spice-des-monomers/` - SPICE-DES-Monomers dataset from QCArchive Dataset
    - `spice-dipeptide/` - SPICE-Dipeptide dataset from QCArchive Dataset
    - `spice-pubchem/` - SPICE-Pubchem dataset from QCArchive Dataset
