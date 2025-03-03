#!/usr/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=30
#SBATCH --mem=250GB
#SBATCH --time=2:00:00

source activate pda
python convert.py
