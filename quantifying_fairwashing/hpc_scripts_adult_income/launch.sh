#!/bin/bash


sbatch AdaBoost_adult_income.sh
sleep 2

sbatch DNN_adult_income.sh
sleep 2

sbatch RF_adult_income.sh
sleep 2

sbatch XgBoost_adult_income.sh

