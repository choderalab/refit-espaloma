## Description
This repository contains scripts to train and evaluate espaloma.

## Manifest
- `eval/` - Stores scripts to inspect validation loss
    - `eval.py` - Inspect loss trajectory
    - `plot.py` - Plot loss vs epochs
- `metric/` - Stores scripts to compute energy and force RMSE metrics *w.r.t* reference QM enegies and forces
    - `metric.py` - Compute RMSE metric for a given espaloma checkpoint file
- `train.py` - Train espaloma
- `save_model.py` - Save espaloma model

## Usage
1. Train espaloma
    >bsub < lsf-submit.sh
    - Checkpoint files will be stored in `checkpoints/`

2. Inspect validation loss trajectory and choose espaloma model candidates
    - Move to `eval/`
    - Submit LSF job
        > bsub < lsf-submit.sh
        - Pickle files containing RMSE metrics information for every checkpoint file in `checkpoint/` will be saved in `pkl/`
    - Plot loss trajectory
        > python plot.py > plot.log
        - Plot loss trajectories based on `energy`, `force`, and `energy+force` against the validation dataset. Loss trajectory is also saved as `rmse.csv`.
        - Espaloma model candidates will be exported based on several criteria from the validation loss trajectory:
            - Lowest `energy`
            - Lowest `force`
            - Lowest `energy+force`
            - Early-stopping with `energy+force`

3. Save serialized espaloma model  
    Example
    >python save_model.py --model `eval/net_es_epoch_860.th`
    - Exports `net_es_epoch_860.pt`

4. Compute RMSE metric
    - Move to `metric/`
    - Run LSF job
        > bsub < lsf-submit.sh
        - RMSE metrics for all datasets will be stored in `rmse_summary.csv`
    