#!/bin/bash


sbatch AdaBoost_marketing.sh
sleep 2

sbatch DNN_marketing.sh
sleep 2

sbatch RF_marketing.sh
sleep 2

sbatch XgBoost_marketing.sh

