#!/bin/bash

datasets=(adult_income compas default_credit marketing)
models=(AdaBoost DNN RF XgBoost)



for dataset in "${datasets[@]}" 
do
        for model in "${models[@]}" 
        do	
               python merge_results.py --dataset=$dataset --rseed=0 --model_class=$model
        done
done