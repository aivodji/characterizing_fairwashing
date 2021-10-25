
# Characterizing the risk of fairwashing

This repository contains the code to reproduce the experiments in our paper [Characterizing the risk of fairwashing](https://arxiv.org/abs/2106.07504).


## Requirements
The project requires the following python packages:

* numpy
* pandas
* h5py
* mpi4py
* scikit-learn
* xgboost
* tensorflow 
* faircorels
* fairlearn

The R scripts requires the following R packages:

* tidyverse
* here
* ranger

## Experiments on characterizing the risk of fairwashing (Sectioon 3 of the paper)

### Data preprocessing
```
cd preprocessing
source main.sh
```

### Black-box models pretraining
```
cd models
source main.sh
```

### Fairwashing attacks, generalization evaluations, and tranferability attacks
Use LaundryML.py for rule list explainers and LaundryML_Fairlearn.py for decision tree and logistic regression explainers.

#### On your local machine. 
Assuming nbr_core is the number of core you want to use:
```
cd core
mpiexec -n nbr_core python LaundryML.py
```

#### On a HPC Cluster
You will have to provide the number of cores in your submission file and srun will use all the core available:
```
cd core
srun python LaundryML.py
```

#### Parameters
* `dataset` (int): Dataset Id. 1: Adult Income, 2: COMPAS,  3: Default Credit, or 4: Marketing
* `rseed` (int): Id of the sample. Choose between 0 and 9
* `metric` (in): Fairness metric Id: 1: SP, 2: PP, 3: PE, 4: EOpp, 5: EOdds, or 6: CUAE'
* `model_class` (string): Black-box model. AdaBoost: AdaBoost classifier, DNN: Deep neural network, RF: Random Forest, or XgBoost: XgBoost classifier 
* `transfer` (int): Boolean indicator to perform the transferability attack. 1: Yes, 0: No



## Reproducing the analysis in the paper
We assume that you have done all the experiments fairwashing attacks, generalization evaluations, and tranferability attacks for all the four datasets, fairness metrics, and black-box models. 

#### Computing the analysis. 
Results are saved in analysis/results. 
The summary of the performances of the black-box models is saved in analysis/results/latex/perfs.tex.
```
cd analysis
source get_results.sh
```

#### Creating the graphs. 
Results are saved in analysis/plotting_scripts/results.
```
cd plotting_scripts
Rscript pareto_fronts.R
Rscript half_unfairness_all.R
Rscript transferability.R
```

## Experiments on quantifying the risk of fairwashing (Sectioon 4 of the paper)
For this part of the experiments, we rely on the code provided by the autors of the paper "Characterizing Fairness Over the Set of Good Models Under Selective Labels" (https://arxiv.org/abs/2101.00352). The original code is available at https://github.com/asheshrambachan/Fairness_In_The_Rashomon_Set.

```
cd quantifying_fairwashing
source compute_unfairness_range.sh
source merge_results.sh
cd quantifying_fairwashing/plotting_scripts
Rscript analysis.R
```