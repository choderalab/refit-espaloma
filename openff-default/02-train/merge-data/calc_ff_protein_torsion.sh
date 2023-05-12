#!/bin/bash
#BSUB -P "merge"
#BSUB -J "protein-torsion"
#BSUB -n 1
#BSUB -R rusage[mem=8]
#BSUB -R span[hosts=1]
#BSUB -q cpuqueue
#BSUB -sp 1 # low priority. default is 12, max is 25
#BSUB -W 23:00
#BSUB -o stdout/calc_%J_%I.stdout
#BSUB -eo stderr/calc_%J_%I.stderr
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
dataset="protein-torsion"
mkdir -p openff-2.0.0/${dataset}
forcefields="gaff-1.81 gaff-2.11 openff-1.2.0 openff-2.0.0 amber14-all.xml"
path_to_dataset="../../01-create-dataset/TorsionDriveDataset"

conda activate espaloma
python ./script/calc_ff.py --path_to_dataset $path_to_dataset --dataset $dataset --forcefields "$forcefields"
