#!/bin/bash
#BSUB -P "duplicate"
#BSUB -J "duplicate"
#BSUB -n 1
#BSUB -R rusage[mem=256]
#BSUB -R span[hosts=1]
#BSUB -q cpuqueue
#BSUB -sp 1 # low priority. default is 12, max is 25
#BSUB -W 24:00
#BSUB -o stdout/duplicated_%J_%I.stdout
#BSUB -eo stderr/duplicated_%J_%I.stderr
#BSUB -L /bin/bash


source ~/.bashrc
OPENMM_CPU_THREADS=1
export OE_LICENSE=~/.openeye/oe_license.txt   # Open eye license activation/env


# change dir
echo "changing directory to ${LS_SUBCWD}"
cd $LS_SUBCWD


# Report node in use
echo "======================"
hostname
#env | sort | grep 'CUDA'
#nvidia-smi
echo "======================"


# run job
conda activate espaloma
python ./script/seperate_duplicated_isomeric_smiles.py
