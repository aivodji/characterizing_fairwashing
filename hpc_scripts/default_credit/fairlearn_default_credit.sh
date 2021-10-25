#!/bin/bash
#SBATCH --time=02:00:00
#SBATCH --array=1,3,4,5
#SBATCH --ntasks=301
#SBATCH --mem-per-cpu=4G


export TMPDIR=/tmp
cd ../../core

models=(AdaBoost DNN RF XgBoost)
rseed=(0 1 2 3 4 5 6 7 8 9)


for r in ${rseed[@]}
do
        for model in "${models[@]}" 
        do	
               srun python LaundryML_Fairlearn.py --dataset=3 --rseed=$r --metric=$SLURM_ARRAY_TASK_ID --explainer=lm  --model_class=$model --transfer
               srun python LaundryML_Fairlearn.py --dataset=3 --rseed=$r --metric=$SLURM_ARRAY_TASK_ID --explainer=dt  --model_class=$model --transfer
        done
done
               

