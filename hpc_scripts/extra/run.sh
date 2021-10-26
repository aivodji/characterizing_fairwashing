#!/bin/bash
#SBATCH --time=00:30:00
#SBATCH --ntasks=301
#SBATCH --mem-per-cpu=4G


export TMPDIR=/tmp
cd ../../core



## Rule List
srun python LaundryML.py --dataset=4 --rseed=8 --metric=1  --model_class=XgBoost --transfer



          

