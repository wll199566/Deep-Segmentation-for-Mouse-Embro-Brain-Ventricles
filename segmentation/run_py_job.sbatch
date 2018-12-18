#!/bin/bash

#SBATCH --job-name=jupyterTest2
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=120GB
#SBATCH --time=40:00:00
#SBATCH --gres=gpu:p40:2

module purge

#module load jupyter-kernels/py2.7
#module load jupyter-kernels/py3.5
#module load keras/2.0.2 
#module load tensorflow/python3.5/1.4.0
#module load opencv/intel/3.2
module load python3/intel/3.6.3
module load cuda/9.0.176
module load cudnn/9.0v7.0.5
source /scratch/zq415/grammar_cor/bin/activate

#python train_seg_net_parse.py --batch_size 4 --epochs 5 --lr 0.01 --SGD --save_version 1
python train_seg_net.py

