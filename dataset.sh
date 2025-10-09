#!/bin/sh
#SBATCH --partition=gpu-2h
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=2

# source /home/hanfeld/.yolopatches/bin/activate
# python src/attacks.py --file $1 --model $2
#apptainer run --nv /home/piha/container_new.sif python attacks.py --file $1 --model $2

apptainer run --nv /home/piha/container_new.sif python attack_refactor.py --results_path $1 --target $2 $3 $4 --seed $5 --lr $6 --epochs $7