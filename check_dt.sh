#!/bin/bash
datasets=(adult_income compas default_credit marketing)
models=(AdaBoost DNN RF XgBoost)
metrics=(equal_opportunity equalized_odds predictive_equality statistical_parity)
rseed=(0 1 2 3 4 5 6 7 8 9)

echo "checking results for decision trees"
for dataset in "${datasets[@]}" 
    do
        for r in ${rseed[@]}
            do
                for model in "${models[@]}" 
                    do	
                        for metric in "${metrics[@]}"
                            do 
                                FILE="./results_dt/${dataset}/${model}/${metric}_${r}.csv"

                                if [ ! -f $FILE ]; then
                                    echo "$FILE not exists."
                                fi
                            done

                        
                    done
            done
    done


echo "checking results for decision trees transfer"
for dataset in "${datasets[@]}" 
    do
        for r in ${rseed[@]}
            do
                for model in "${models[@]}" 
                    do	
                        for metric in "${metrics[@]}"
                            do 
                                FILE="./results_dt/${dataset}/${model}/${metric}_transfer_${r}.csv"

                                if [ ! -f $FILE ]; then
                                    echo "$FILE not exists."
                                fi
                            done

                        
                    done
            done
    done