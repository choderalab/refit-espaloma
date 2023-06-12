#!/bin/bash
#BSUB -P "esp"
#BSUB -J "metric"
#BSUB -n 1
#BSUB -R rusage[mem=512]
#BSUB -R span[hosts=1]
#BSUB -q cpuqueue
#BSUB -sp 1 # low priority. default is 12, max is 25
#BSUB -W 12:00
#BSUB -o out_%J_%I.stdout
#BSUB -eo out_%J_%I.stderr
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


# paramters
layer="SAGEConv"
units=512
activation="relu"
config="${units} relu 0.1 ${units} relu 0.1 ${units} relu 0.1"
janossy_config="${units} relu 0.1 ${units} relu 0.1 ${units} relu 0.1 ${units} relu 0.1"
input_prefix="../../../merge-data/openff-2.0.0_filtered"
best_model="../eval/net_es_epoch_860.th"


# activate
conda activate espaloma-fix


# run jobs
dataset="gen2"
python metric.py --layer $layer --units $units --activation $activation --config "$config" --janossy_config "$janossy_config" --input_prefix $input_prefix --dataset "$dataset" --best_model $best_model

dataset="pepconf-dlc"
python metric.py --layer $layer --units $units --activation $activation --config "$config" --janossy_config "$janossy_config" --input_prefix $input_prefix --dataset "$dataset" --best_model $best_model

dataset="gen2-torsion"
python metric.py --layer $layer --units $units --activation $activation --config "$config" --janossy_config "$janossy_config" --input_prefix $input_prefix --dataset "$dataset" --best_model $best_model

dataset="protein-torsion"
python metric.py --layer $layer --units $units --activation $activation --config "$config" --janossy_config "$janossy_config" --input_prefix $input_prefix --dataset "$dataset" --best_model $best_model

dataset="spice-pubchem"
python metric.py --layer $layer --units $units --activation $activation --config "$config" --janossy_config "$janossy_config" --input_prefix $input_prefix --dataset "$dataset" --best_model $best_model

dataset="spice-dipeptide"
python metric.py --layer $layer --units $units --activation $activation --config "$config" --janossy_config "$janossy_config" --input_prefix $input_prefix --dataset "$dataset" --best_model $best_model

dataset="spice-des-monomers"
python metric.py --layer $layer --units $units --activation $activation --config "$config" --janossy_config "$janossy_config" --input_prefix $input_prefix --dataset "$dataset" --best_model $best_model

dataset="rna-diverse"
python metric.py --layer $layer --units $units --activation $activation --config "$config" --janossy_config "$janossy_config" --input_prefix $input_prefix --dataset "$dataset" --best_model $best_model

dataset="rna-trinucleotide"
python metric.py --layer $layer --units $units --activation $activation --config "$config" --janossy_config "$janossy_config" --input_prefix $input_prefix --dataset "$dataset" --best_model $best_model

dataset="rna-nucleoside"
python metric.py --layer $layer --units $units --activation $activation --config "$config" --janossy_config "$janossy_config" --input_prefix $input_prefix --dataset "$dataset" --best_model $best_model