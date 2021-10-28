#!/bin/bash
#SBATCH --time=02:00:00
#SBATCH --ntasks=301
#SBATCH --mem-per-cpu=4G
export TMPDIR=/tmp
cd ../../core
srun python LaundryML_Fairlearn.py --dataset=1 --rseed=7 --metric=1 --explainer=dt  --model_class=AdaBoost --transfer
srun python LaundryML_Fairlearn.py --dataset=1 --rseed=7 --metric=1 --explainer=lm  --model_class=AdaBoost --transfer
srun python LaundryML_Fairlearn.py --dataset=1 --rseed=7 --metric=1 --explainer=dt  --model_class=DNN --transfer
srun python LaundryML_Fairlearn.py --dataset=1 --rseed=7 --metric=1 --explainer=lm  --model_class=DNN --transfer
srun python LaundryML_Fairlearn.py --dataset=1 --rseed=7 --metric=1 --explainer=dt  --model_class=RF --transfer
srun python LaundryML_Fairlearn.py --dataset=1 --rseed=7 --metric=1 --explainer=lm  --model_class=RF --transfer
srun python LaundryML_Fairlearn.py --dataset=1 --rseed=7 --metric=1 --explainer=dt  --model_class=XgBoost --transfer
srun python LaundryML_Fairlearn.py --dataset=1 --rseed=7 --metric=1 --explainer=lm  --model_class=XgBoost --transfer
srun python LaundryML_Fairlearn.py --dataset=1 --rseed=8 --metric=1 --explainer=dt  --model_class=AdaBoost --transfer
srun python LaundryML_Fairlearn.py --dataset=1 --rseed=8 --metric=1 --explainer=lm  --model_class=AdaBoost --transfer
srun python LaundryML_Fairlearn.py --dataset=1 --rseed=8 --metric=1 --explainer=dt  --model_class=DNN --transfer
srun python LaundryML_Fairlearn.py --dataset=1 --rseed=8 --metric=1 --explainer=lm  --model_class=DNN --transfer
srun python LaundryML_Fairlearn.py --dataset=1 --rseed=8 --metric=1 --explainer=dt  --model_class=RF --transfer
srun python LaundryML_Fairlearn.py --dataset=1 --rseed=8 --metric=1 --explainer=lm  --model_class=RF --transfer
srun python LaundryML_Fairlearn.py --dataset=1 --rseed=8 --metric=1 --explainer=dt  --model_class=XgBoost --transfer
srun python LaundryML_Fairlearn.py --dataset=1 --rseed=8 --metric=1 --explainer=lm  --model_class=XgBoost --transfer
srun python LaundryML_Fairlearn.py --dataset=1 --rseed=9 --metric=1 --explainer=dt  --model_class=AdaBoost --transfer
srun python LaundryML_Fairlearn.py --dataset=1 --rseed=9 --metric=1 --explainer=lm  --model_class=AdaBoost --transfer
srun python LaundryML_Fairlearn.py --dataset=1 --rseed=9 --metric=1 --explainer=dt  --model_class=DNN --transfer
srun python LaundryML_Fairlearn.py --dataset=1 --rseed=9 --metric=1 --explainer=lm  --model_class=DNN --transfer
srun python LaundryML_Fairlearn.py --dataset=1 --rseed=9 --metric=1 --explainer=dt  --model_class=RF --transfer
srun python LaundryML_Fairlearn.py --dataset=1 --rseed=9 --metric=1 --explainer=lm  --model_class=RF --transfer
srun python LaundryML_Fairlearn.py --dataset=1 --rseed=9 --metric=1 --explainer=dt  --model_class=XgBoost --transfer
srun python LaundryML_Fairlearn.py --dataset=1 --rseed=9 --metric=1 --explainer=lm  --model_class=XgBoost --transfer