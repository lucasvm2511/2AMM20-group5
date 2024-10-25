#!/bin/bash
#Set job requirements
#SBATCH -n 16
#SBATCH -t 16:00:00
#SBATCH -p gpu_a100
#SBATCH --gpus=1
 
#Loading modules
module load 2022
module load Python/3.10.4-GCCcore-11.3.0
 
#Copy input file to scratch

#Execute a Python program located in $HOME, that takes an input file and output directory as arguments.
pip3 install -r $HOME/nif-master/requirements.txt
./nif-master/run_all_experiments_3d.sh $HOME/nif-master/experiments/configs/ $HOME/nif-master/experiments/images/ $HOME/nif-master/experimental-results-3d
