#!/bin/bash


sbatch AdaBoost_default_credit.sh
sleep 2

sbatch DNN_default_credit.sh
sleep 2

sbatch RF_default_credit.sh
sleep 2

sbatch XgBoost_default_credit.sh

