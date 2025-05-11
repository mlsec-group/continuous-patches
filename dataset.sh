#!/bin/sh
#SBATCH --partition=gpu-2h
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=2

# source /home/hanfeld/.yolopatches/bin/activate
# python src/attacks.py --file $1 --model $2
apptainer run --nv container.sif python attacks.py --file $1 --model $2