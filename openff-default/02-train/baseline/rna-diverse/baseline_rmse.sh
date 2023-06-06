#!/bin/bash
#BSUB -P "esp"
#BSUB -J "baseline"
#BSUB -n 1
#BSUB -R rusage[mem=8]
#BSUB -R span[hosts=1]
#BSUB -q cpuqueue
#BSUB -sp 1 # low priority. default is 12, max is 25
###BSUB -W 3:00
#BSUB -W 6:00
###BSUB -o out_%J_%I.stdout
###BSUB -eo out_%J_%I.stderr
#BSUB -L /bin/bash

source ~/.bashrc
OPENMM_CPU_THREADS=1
#export OE_LICENSE=~/.openeye/oe_license.txt   # Open eye license activation/env


# change dir
echo "changing directory to ${LS_SUBCWD}"
cd $LS_SUBCWD


# Report node in use
echo "======================"
hostname
env | sort | grep 'CUDA'
nvidia-smi
echo "======================"


# run
input_prefix="../../merge-data/openff-2.0.0_filtered"
dataset="rna-diverse"
forcefields="gaff-1.81 gaff-2.11 openff-1.2.0 openff-2.0.0 openff-2.1.0 amber14"

conda activate espaloma
python ../baseline_rmse.py --input_prefix $input_prefix --dataset $dataset --forcefields="$forcefields"
