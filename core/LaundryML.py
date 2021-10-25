import pandas as pd
import numpy as np
from joblib import Parallel, delayed, parallel_backend
from faircorels import load_from_csv, CorelsClassifier
from metrics import ConfusionMatrix, Metric
import csv
import argparse
import os
from config import get_data, get_metric, get_strategy

# parser initialization
parser = argparse.ArgumentParser(description='Evaluation of FairCORELS')
parser.add_argument('--dataset', type=int, default=1, help='Dataset. 1 Adult, 2: COMPAS, 3: German Credit, 4: Default Credit, 5: Adult_marital, 6: Adult_no_relationship')
parser.add_argument('--rseed', type=int, default=0, help='random seed: choose between 0 - 9')
parser.add_argument('--metric', type=int, default=1, help='Fairness metric. 1: SP, 2:  PP, 3: PE, 4: EOpp, 5: EOdds, 6: CUAE')
parser.add_argument('--model_class', type=str, default='DNN', help='Model class: DNN, RF, SVM, XgBoost, AdaBoost')
parser.add_argument('--transfer', action='store_true', help='Perform transferability attacks.')
parser.add_argument('--debug', action='store_true', help='debug mode')

args = parser.parse_args()


#get dataset and relative infos
dataset, decision, prediction_name, min_feature, min_pos, maj_feature, maj_pos = get_data(args.dataset)
rseed = args.rseed
transfer = args.transfer

# suing group files
filename_X_train = '../preprocessing/preprocessed/{}/{}_attackRules_{}.csv'.format(dataset, dataset, rseed)
filename_y_train = '../models/labels/{}/{}_sg_{}.csv'.format(dataset, args.model_class, rseed)
filename_y_train_true = '../models/true_labels/{}/label_sg_{}.csv'.format(dataset, rseed)

# test file
filename_X_test = '../preprocessing/preprocessed/{}/{}_testRules_{}.csv'.format(dataset, dataset, rseed)
filename_y_test = '../models/labels/{}/{}_test_{}.csv'.format(dataset, args.model_class, rseed)
filename_y_test_true = '../models/true_labels/{}/label_test_{}.csv'.format(dataset, rseed)

# load dataset
X_train = pd.read_csv(filename_X_train)
y_train = pd.read_csv(filename_y_train)['prediction']
y_train_true = pd.read_csv(filename_y_train_true)['prediction']
X_test = pd.read_csv(filename_X_test)
y_test = pd.read_csv(filename_y_test)['prediction']
y_test_true = pd.read_csv(filename_y_test_true)['prediction']
features = list(X_train)

#------------------------setup config

#iterations
N_ITER = 300000


#fairness constraint
fairness_metric_name = get_metric(args.metric)
fairness_metric = args.metric


#epsilons
epsilon_range = np.arange(0.701, 0.991, 0.001)
base = [0.0, 0.05, 0.1, 0.3, 0.4, 0.5, 0.6, 0.7]
epsilon_range = base + list(epsilon_range) + [0.9999]
epsilons = [round(x,3) for x in epsilon_range] #300 values

if args.debug:
    epsilons = [0.942]
    N_ITER = 7000000
    print(">>>>>>>>>>>>>>> Debug Mode <<<<<<<<<<<<<<<<<<<<<<<<")


# not use ulb (unfairness lower bound of FairCORELS). No need, for this particular experiment. 
ulb = False


# get search strategy. We use the default search strategy (BFS) of FairCORELS 
strategy, bfsMode, strategy_name = get_strategy(1)


#save direcory
save_dir = "../results_rl/{}/{}".format(dataset, args.model_class)
os.makedirs(save_dir, exist_ok=True)

def compute_tranferability(clf, epsilon):
    all_models = ['DNN', 'RF', 'AdaBoost', 'XgBoost']

    # sg
    df_sg = pd.DataFrame(X_train, columns=features)

    res = {}
    res['epsilon'] = epsilon

    for mdl in all_models:
        filename    = '../models/labels/{}/{}_sg_{}.csv'.format(dataset, mdl, rseed)
        y           = pd.read_csv(filename)['prediction']
        df_sg[decision] = y
        df_sg["predictions"] = clf.predict(X_train)
        cm_sg = ConfusionMatrix(df_sg[min_feature], df_sg[maj_feature], df_sg["predictions"], df_sg[decision])
        cm_minority_sg, cm_majority_sg = cm_sg.get_matrix()
        fm_sg = Metric(cm_minority_sg, cm_majority_sg)
        fidelity = clf.score(X_train, y)
        unfairness = fm_sg.fairness_metric(fairness_metric)

        res['fidelity_{}'.format(mdl)] = fidelity
        res['unfairness_{}'.format(mdl)] = unfairness
    
    return res


def fit(epsilon, fairness, transfer=True):

    print("===="*20 + ">>>>"*10 + str(epsilon))

    clf = CorelsClassifier(n_iter=N_ITER, 
                            min_support=0.01,
                            c=1e-3, 
                            max_card=1, 
                            policy=strategy,
                            bfs_mode=bfsMode,
                            mode=3,
                            useUnfairnessLB=ulb,
                            forbidSensAttr=False,
                            fairness=fairness, 
                            epsilon=epsilon,
                            maj_pos=maj_pos, 
                            min_pos=min_pos,
                            verbosity=[]
                            )

    clf.fit(X_train, y_train, features=features, prediction_name=prediction_name)

    #train 
    df_train = pd.DataFrame(X_train, columns=features)
    df_train[decision] = y_train
    df_train["predictions"] = clf.predict(X_train)
    cm_train = ConfusionMatrix(df_train[min_feature], df_train[maj_feature], df_train["predictions"], df_train[decision])
    cm_minority_train, cm_majority_train = cm_train.get_matrix()
    fm_train = Metric(cm_minority_train, cm_majority_train)

    acc_train = clf.score(X_train, y_train)
    unf_train = fm_train.fairness_metric(fairness_metric)
    mdl = clf.rl().__str__()

    

    print("===="*25 + ">>>>"*10 + " unfairness: {}, fidelity {}".format(unf_train, acc_train))


    #test
    df_test = pd.DataFrame(X_test, columns=features)
    df_test[decision] = y_test
    df_test["predictions"] = clf.predict(X_test)
    cm_test = ConfusionMatrix(df_test[min_feature], df_test[maj_feature], df_test["predictions"], df_test[decision])
    cm_minority_test, cm_majority_test = cm_test.get_matrix()
    fm_test = Metric(cm_minority_test, cm_majority_test)

    acc_test = clf.score(X_test, y_test)
    unf_test = fm_test.fairness_metric(fairness_metric)

    print("===="*25 + ">>>>"*10 + " unfairness test: {}, acc test {}".format(unf_test, acc_test))

    # accuracy on the true labels
    acc_train_true = clf.score(X_train, y_train_true)
    acc_test_true = clf.score(X_test, y_test_true)

    base_output = [epsilon, acc_train, unf_train, acc_test, unf_test, mdl, acc_train_true, acc_test_true]

    print("===="*10 + " training done ......")

    return (base_output, compute_tranferability(clf, epsilon)) if transfer else base_output

def split(container, count):
    return [container[_i::count] for _i in range(count)]

def process_results(epsilons, results, save_path, metric_name, transfer):

    base_results, transfer_results = None, None
    
    if transfer:
        base_results = [res[0] for res in results]
        transfer_results = [res[1] for res in results]
    else:
        print(results)
        base_results = results
    
    row_list = []
    row_list_model = []
    row_list_transfer = []

    # process base results
    for res in base_results:
        row = {}
        row_model = {}
        row['epsilon']              = res[0]
        row['fidelity_sg']          = res[1]
        row['unfairness_sg']        = res[2]
        row['fidelity_test']        = res[3]
        row['unfairness_test']      = res[4]

        # perfs on true labels
        row['accuracy_sg']          = res[6]
        row['accuracy_test']        = res[7]

        # explanation models description
        row_model['epsilon']        = res[0]
        row_model['model']          = res[5]
        row_list.append(row)
        row_list_model.append(row_model)

    filename = '{}/{}_{}.csv'.format(save_path, metric_name, rseed)
    filename_model = '{}/{}_model_{}.csv'.format(save_path, metric_name, rseed)

    df = pd.DataFrame(row_list)
    df_model = pd.DataFrame(row_list_model)

    df.to_csv(filename, encoding='utf-8', index=False)
    df_model.to_csv(filename_model, encoding='utf-8', index=False)

    # process for transfer
    if transfer:
        for transfer_res in transfer_results:
            row_list_transfer.append(transfer_res)
        df_transfer = pd.DataFrame(row_list_transfer)
        filename_transfer = '{}/{}_transfer_{}.csv'.format(save_path, metric_name, rseed)
        df_transfer.to_csv(filename_transfer, encoding='utf-8', index=False)


import mpi4py.rc
mpi4py.rc.threads = False

from mpi4py import MPI

print(">>>>>>>>>"*20)

COMM = MPI.COMM_WORLD

if COMM.rank == 0:
    jobs = split(epsilons, COMM.size)
else:
    jobs = None

jobs = COMM.scatter(jobs, root=0)



results = []
for epsilon in jobs:
    print("---------------------------->>> epsilon = {}".format(epsilon))
    results.append(fit(epsilon, fairness_metric, transfer))


# Gather results on rank 0.
results = MPI.COMM_WORLD.gather(results, root=0)

if COMM.rank == 0:
    results = [_i for temp in results for _i in temp]
    process_results(epsilons, results, save_dir, fairness_metric_name, transfer)

