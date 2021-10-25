#!/bin/bash
#SBATCH --time=02:00:00
#SBATCH --ntasks=301
#SBATCH --mem-per-cpu=4G


export TMPDIR=/tmp
cd ../../core



## Logistic regression
srun python LaundryML_Fairlearn.py --dataset=1 --rseed=6 --metric=1 --explainer=lm  --model_class=XgBoost --transfer
srun python LaundryML_Fairlearn.py --dataset=4 --rseed=2 --metric=3 --explainer=lm  --model_class=XgBoost --transfer

# Decision tree
srun python LaundryML_Fairlearn.py --dataset=1 --rseed=5 --metric=5 --explainer=dt  --model_class=RF --transfer
srun python LaundryML_Fairlearn.py --dataset=1 --rseed=7 --metric=4 --explainer=dt  --model_class=AdaBoost --transfer
srun python LaundryML_Fairlearn.py --dataset=3 --rseed=9 --metric=1 --explainer=dt  --model_class=DNN --transfer
srun python LaundryML_Fairlearn.py --dataset=4 --rseed=2 --metric=5 --explainer=dt  --model_class=AdaBoost --transfer
srun python LaundryML_Fairlearn.py --dataset=4 --rseed=4 --metric=1 --explainer=dt  --model_class=RF --transfer





          

