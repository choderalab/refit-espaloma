#!/bin/bash
#BSUB -P "esp"
#BSUB -J "@@@JOBNAME@@@"
#BSUB -n 1
#BSUB -R rusage[mem=8]
#BSUB -R span[hosts=1]
#BSUB -q cpuqueue
#BSUB -sp 1 # low priority. default is 12, max is 25
#BSUB -W 2:00

## Asking for A100s
#BSUB -o out_%J_%I.stdout
#BSUB -eo out_%J_%I.stderr
#BSUB -L /bin/bash

source ~/.bashrc
OPENMM_CPU_THREADS=1
export OE_LICENSE=~/.openeye/oe_license.txt   # Open eye license activation/env


# chnage dir
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
BASE_PATH=/home/takabak/data/exploring-rna/rna-espaloma/espaloma-openff-default.3/01-create-dataset/rna-diverse
DATASET_PATH=/home/takabak/data/qca-dataset/openff-default/rna-diverse
python ${BASE_PATH}/getgraph_hdf5.py --hdf5 ${DATASET_PATH}/RNA-DIVERSE-OPENFF-DEFAULT.hdf5 --output_prefix "mydata" --keyname "@@@KEYNAME@@@"
