#!/bin/bash
#SBATCH --time=03:00:00
#SBATCH --array=1-10
#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=5G
#SBATCH --mail-user=a.u.matchi@gmail.com
#SBATCH --mail-type=ALL


export TMPDIR=/tmp
cd ..

Rscript compute_unfairness_range.R --dataset=compas --rseed=0 --bbox=DNN --pos=$SLURM_ARRAY_TASK_ID