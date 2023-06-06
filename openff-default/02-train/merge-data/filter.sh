#!/bin/bash
#BSUB -P "filter"
#BSUB -J "filter"
#BSUB -n 1
#BSUB -R rusage[mem=8]
#BSUB -R span[hosts=1]
#BSUB -q cpuqueue
#BSUB -sp 1 # low priority. default is 12, max is 25
#BSUB -W 24:00
#BSUB -o stdout/filter_%J_%I.stdout
#BSUB -eo stderr/filter_%J_%I.stderr
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

# rna
dataset="rna-diverse"
mkdir -p openff-2.0.0_filtered/${dataset}
python ./script/filter.py --dataset ${dataset}

# rna-trinucleotide
dataset="rna-trinucleotide"
mkdir -p openff-2.0.0_filtered/${dataset}
python ./script/filter.py --dataset ${dataset}

# rna-nucleoside
dataset="rna-nucleoside"
mkdir -p openff-2.0.0_filtered/${dataset}
python ./script/filter.py --dataset ${dataset}

# spice-pubchem
dataset="spice-pubchem"
mkdir -p openff-2.0.0_filtered/${dataset}
python ./script/filter.py --dataset ${dataset}

# spice-dipeptide
dataset="spice-dipeptide"
mkdir -p openff-2.0.0_filtered/${dataset}
python ./script/filter.py --dataset ${dataset}

# spice-des-monomers
dataset="spice-des-monomers"
mkdir -p openff-2.0.0_filtered/${dataset}
python ./script/filter.py --dataset ${dataset}

# gen2
dataset="gen2"
mkdir -p openff-2.0.0_filtered/${dataset}
python ./script/filter.py --dataset ${dataset}

# pepconf-dlc
dataset="pepconf-dlc"
mkdir -p openff-2.0.0_filtered/${dataset}
python ./script/filter.py --dataset ${dataset}

# gen2-torsion
dataset="gen2-torsion"
mkdir -p openff-2.0.0_filtered/${dataset}
python ./script/filter.py --dataset ${dataset}

# protein-torsion
dataset="protein-torsion"
mkdir -p openff-2.0.0_filtered/${dataset}
python ./script/filter.py --dataset ${dataset}
