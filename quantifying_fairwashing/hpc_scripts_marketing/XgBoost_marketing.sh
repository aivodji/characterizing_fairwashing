#!/bin/bash
#SBATCH --time=20:00:00
#SBATCH --array=1-10
#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=10G
#SBATCH --mail-user=a.u.matchi@gmail.com
#SBATCH --mail-type=ALL


export TMPDIR=/tmp
cd ..

Rscript compute_unfairness_range.R --dataset=marketing --rseed=0 --bbox=XgBoost --pos=$SLURM_ARRAY_TASK_ID