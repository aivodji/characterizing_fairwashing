#!/bin/bash

for ((i=1;i<=10;i++));
do 
   Rscript compute_unfairness_range.R --dataset=compas --rseed=0 --bbox=DNN --pos=$i
done 

