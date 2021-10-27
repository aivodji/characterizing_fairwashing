#!/bin/bash
#SBATCH --time=02:00:00
#SBATCH --ntasks=160
#SBATCH --mem-per-cpu=8G

export TMPDIR=/tmp
xargs -P 160 -n 1 -d "\n" -a sweeps.sh bash -c