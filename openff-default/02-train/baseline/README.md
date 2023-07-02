## Description
This repository contains scripts for computing the root mean square error (RMSE) of energies and forces obtained from baseline force fields (`gaff-1.81`, `gaff-2.11`, `openff-1.2.1`, `openff-2.0.0`, and `Amber ff14SB (RNA.OL3)`) with respect to the reference QM energies and forces stored in the DGL graphs. All statistics are computed by centering the predicted and reference energies to have zero mean for each molecule, following a similar apporach to [previous work](https://pubs.rsc.org/en/content/articlelanding/2022/sc/d2sc02739a). The 95% confidence intervals are computed by bootstrapping molecular conformers with replacement using 1000 replicates.


## Basic Usage
1. Move to one of the directories (e.g. `gen2-torsion/`)
2. Run LSF script
    >bsub < baseline_rmse.sh

    - This will generate a summary file `summary.csv`
