#!/bin/bash 
#SBATCH --time=12:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
python Train.py -p ../config/train_res.yaml
