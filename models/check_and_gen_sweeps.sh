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
                        FILE="pretrained/${dataset}/${model}_${r}.txt"
                        CMD="python train_models.py --dataset=${dataset} --model_class=${model} --nbr_evals=25 --rseed=${r}"

                        if [ ! -f $FILE ]; then
                            echo "${CMD}"
                        fi

                        
                    done
            done
    done
