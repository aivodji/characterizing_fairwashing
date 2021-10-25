#!/bin/bash

# inputs
datasets=(adult_income compas default_credit marketing)
models=(AdaBoost DNN RF XgBoost)
metrics=(1 3 4 5)



# compute the unfairness of the black-box

python unfairness_bbox.py 

# compute average trade-offs over the seed
for dataset in "${datasets[@]}" 
    do
        for metric in ${metrics[@]}
            do
                for model in "${models[@]}" 
                    do	
                        #python average_trade_off.py --dataset=$dataset --metric=$metric --model_class=$model --explainer=rl
                        python average_trade_off.py --dataset=$dataset --metric=$metric --model_class=$model --explainer=lm
                        #python average_trade_off.py --dataset=$dataset --metric=$metric --model_class=$model --explainer=dt
                    done
            done
    done

# compute pareto front
for dataset in "${datasets[@]}" 
    do
        for metric in ${metrics[@]}
            do
                for model in "${models[@]}" 
                    do	
                        #python compute_pareto.py --dataset=$dataset --metric=$metric --model_class=$model --explainer=rl
                        python compute_pareto.py --dataset=$dataset --metric=$metric --model_class=$model --explainer=lm
                        #python compute_pareto.py --dataset=$dataset --metric=$metric --model_class=$model --explainer=dt
                    done
            done
    done


# merge results 
for dataset in "${datasets[@]}" 
    do
        #python merge_pareto.py --dataset=$dataset --explainer=rl
        python merge_pareto.py --dataset=$dataset --explainer=lm
        #python merge_pareto.py --dataset=$dataset --explainer=dt     
    done

: <<'END'

## compute label agreement between teacher and student models
python label_agreements_bbox.py


# compute transferabilty average
for dataset in "${datasets[@]}" 
    do
        for metric in ${metrics[@]}
            do
                for model in "${models[@]}" 
                    do	
                        #python average_transferabily.py --dataset=$dataset --metric=$metric --model_class=$model --explainer=rl
                        python average_transferabily.py --dataset=$dataset --metric=$metric --model_class=$model --explainer=dt
                        python average_transferabily.py --dataset=$dataset --metric=$metric --model_class=$model --explainer=lm
                        
                    done
            done
    done



## compute transferability heatmap
for dataset in "${datasets[@]}" 
do
    for epsilon in 0.03 0.05 0.1
    do
        python compute_transferability.py --dataset=$dataset --epsilon=$epsilon --explainer=rl
        python compute_transferability.py --dataset=$dataset --epsilon=$epsilon --explainer=dt
        python compute_transferability.py --dataset=$dataset --epsilon=$epsilon --explainer=lm
    done
done


# compute average trade-off
for dataset in "${datasets[@]}" 
    do
        for metric in ${metrics[@]}
            do
                for model in "${models[@]}" 
                    do	
                        python average_trade_off.py --dataset=$dataset --metric=$metric --model_class=$model --explainer=rl
                        python average_trade_off.py --dataset=$dataset --metric=$metric --model_class=$model --explainer=dt
                        python average_trade_off.py --dataset=$dataset --metric=$metric --model_class=$model --explainer=lm
                        
                    done
            done
    done


## fidelity of unconstrained explainers
for explainer in "dt" "lm" "rl" 
    do
        for group in "sg" "test"
            do
                python fidelity_unconstrained.py --explainer=$explainer --group=$group
            done
    done


## explainers with half unfairness
for explainer in "dt" "lm" "rl" 
    do
        for group in "sg" "test"
            do
                python half_unfairness.py --explainer=$explainer --group=$group
                python half_unfairness.py --explainer=$explainer --group=$group --all
            done
    done



# Other analysis

## summary of black-box models, results save in latex tables: in results/latex/perfs.tex
python summary.py
python latex_summary.py

END