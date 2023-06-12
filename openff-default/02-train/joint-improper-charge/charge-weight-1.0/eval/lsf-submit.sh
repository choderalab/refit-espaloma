#!/bin/bash
#BSUB -P "esp"
#BSUB -J "eval-[1-150]"
#BSUB -n 1
#BSUB -R rusage[mem=384]
#BSUB -R span[hosts=1]
#BSUB -q cpuqueue
#BSUB -sp 1 # low priority. default is 12, max is 25
#BSUB -W 2:00
#BSUB -o out/out_%J_%I.stdout
#BSUB -eo out/out_%J_%I.stderr
#BSUB -L /bin/bash

source ~/.bashrc
OPENMM_CPU_THREADS=1
#export OE_LICENSE=~/.openeye/oe_license.txt   # Open eye license activation/env

# chnage dir
echo "changing directory to ${LS_SUBCWD}"
cd $LS_SUBCWD


# Report node in use
echo "======================"
hostname
env | sort | grep 'CUDA'
nvidia-smi
echo "======================"


# parameters
layer="SAGEConv"
units=512
activation="relu"
config="${units} relu 0.1 ${units} relu 0.1 ${units} relu 0.1"
janossy_config="${units} relu 0.1 ${units} relu 0.1 ${units} relu 0.1 ${units} relu 0.1"
input_prefix="../../../merge-data/openff-2.0.0_filtered"
checkpoint_path="../checkpoints"

# temporal directory
mkdir -p pkl out

# conda
conda activate espaloma-fix

# run
datasets="gen2 gen2-torsion pepconf-dlc protein-torsion spice-pubchem spice-dipeptide spice-des-monomers rna-diverse"
python eval.py --layer $layer --units $units --activation $activation --config "$config" --janossy_config "$janossy_config" \
--input_prefix $input_prefix --datasets="$datasets" --checkpoint_path $checkpoint_path --epoch $(( $LSB_JOBINDEX * 10 ))

