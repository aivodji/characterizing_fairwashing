#!/bin/bash

datasets=(adult_income compas default_credit marketing)
models=(AdaBoost DNN RF XgBoost)
rseed=(0 1 2 3 4 5 6 7 8 9)


for dataset in "${datasets[@]}" 
    do
        for r in ${rseed[@]}
            do
                for model in "${models[@]}" 
                    do	
                        # pretraining the black-box model
                        echo "python train_models.py --dataset=${dataset} --model_class=${model} --nbr_evals=50 --rseed=${r}"
                        
                    done
            done
    done