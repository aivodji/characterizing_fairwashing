#!/bin/bash


sbatch AdaBoost_compas.sh
sleep 2

sbatch DNN_compas.sh
sleep 2

sbatch RF_compas.sh
sleep 2

sbatch XgBoost_compas.sh

