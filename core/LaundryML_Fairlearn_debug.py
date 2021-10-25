import pandas as pd
import numpy as np
from joblib import Parallel, delayed, parallel_backend
import csv
import argparse
import os
from config_fairlearn import get_data, get_metric
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from fairlearn.metrics import MetricFrame
from fairlearn.reductions import ExponentiatedGradient, DemographicParity, FalsePositiveRateParity, TruePositiveRateParity, EqualizedOdds
from metrics import ConfusionMatrix, Metric



metrics_map = {
    1 : DemographicParity,
    3 : FalsePositiveRateParity,
    4 : TruePositiveRateParity,
    5 : EqualizedOdds
}


explainer_map = {
    'lm' : LogisticRegression(solver='liblinear', fit_intercept=True),
    'dt' : DecisionTreeClassifier(min_samples_leaf=10, max_depth=4)
}

# parser initialization
parser = argparse.ArgumentParser(description='Evaluation of FairCORELS')
parser.add_argument('--dataset', type=int, default=1, help='Dataset. 1 Adult, 2: COMPAS, 3: German Credit, 4: Default Credit, 5: Adult_marital, 6: Adult_no_relationship')
parser.add_argument('--rseed', type=int, default=0, help='random seed: choose between 0 - 9')
parser.add_argument('--metric', type=int, default=1, help='Fairness metric. 1: SP, 2:  PP, 3: PE, 4: EOpp, 5: EOdds, 6: CUAE')
parser.add_argument('--model_class', type=str, default='DNN', help='Model class: DNN, RF, SVM, XgBoost, AdaBoost')
parser.add_argument('--transfer', action='store_true', help='Perform transferability attacks.')
parser.add_argument('--debug', action='store_true', help='debug mode')

parser.add_argument('--explainer', type=str, default='lm', help='Model class for the explainer. lm for logistic regression, dt for decision tree')


args = parser.parse_args()


#get dataset and relative infos
dataset, decision, maj_grp, min_grp = get_data(args.dataset)
rseed = args.rseed
transfer = args.transfer
explainer = args.explainer

#fairness constraint
fairness_metric_name = get_metric(args.metric)
fairness_metric = args.metric


# suing group files
filename_X_train = '../preprocessing/preprocessed/{}/{}_attackOneHot_{}.csv'.format(dataset, dataset, rseed)
filename_y_train = '../models/labels/{}/{}_sg_{}.csv'.format(dataset, args.model_class, rseed)
filename_y_train_true = '../models/true_labels/{}/label_sg_{}.csv'.format(dataset, rseed)

# test file
filename_X_test = '../preprocessing/preprocessed/{}/{}_testOneHot_{}.csv'.format(dataset, dataset, rseed)
filename_y_test = '../models/labels/{}/{}_test_{}.csv'.format(dataset, args.model_class, rseed)
filename_y_test_true = '../models/true_labels/{}/label_test_{}.csv'.format(dataset, rseed)

# load dataset
X_train = pd.read_csv(filename_X_train)
maj_features_train = X_train[maj_grp]
min_features_train = X_train[min_grp]

#X_train.drop([decision], axis = 1, inplace = True)
X_train.drop([decision, maj_grp, min_grp], axis = 1, inplace = True)
y_train = pd.read_csv(filename_y_train)['prediction']
y_train_true = pd.read_csv(filename_y_train_true)['prediction']

X_test = pd.read_csv(filename_X_test)
maj_features_test = X_test[maj_grp]
min_features_test = X_test[min_grp]

X_test.drop([decision, maj_grp, min_grp], axis = 1, inplace = True)
y_test = pd.read_csv(filename_y_test)['prediction']
y_test_true = pd.read_csv(filename_y_test_true)['prediction']


#exit(0)

#epsilons
epsilon_range = np.arange(0.701, 0.991, 0.001)
base = [0.0, 0.05, 0.1, 0.3, 0.4, 0.5, 0.6, 0.7]
epsilon_range = base + list(epsilon_range) + [0.9999]
epsilons = [round(x,3) for x in epsilon_range] #300 values


if args.debug:
    #epsilons = [0.90, 0.91, 0.92, 0.93, 0.94, 0.95, 0.96, 0.97, 0.98, 0.99]
    epsilons = [0.0]
    print(">>>>>>>>>>>>>>> Debug Mode <<<<<<<<<<<<<<<<<<<<<<<<")


#save direcory
save_dir = "../results_{}/{}/{}".format(explainer, dataset, args.model_class)
os.makedirs(save_dir, exist_ok=True)

def compute_tranferability(clf, epsilon):
    all_models = ['DNN', 'RF', 'AdaBoost', 'XgBoost']

    res = {}
    res['epsilon'] = epsilon

    for mdl in all_models:
        filename    = '../models/labels/{}/{}_sg_{}.csv'.format(dataset, mdl, rseed)
        y_train           = pd.read_csv(filename)['prediction']
        y_pred_train = clf.predict(X_train)

        cm_sg = ConfusionMatrix(min_features_train, maj_features_train, y_pred_train, y_train)
        cm_minority_sg, cm_majority_sg = cm_sg.get_matrix()
        fm_sg = Metric(cm_minority_sg, cm_majority_sg)
        fidelity = accuracy_score(y_train, y_pred_train)
        unfairness = fm_sg.fairness_metric(fairness_metric)

        res['fidelity_{}'.format(mdl)] = fidelity
        res['unfairness_{}'.format(mdl)] = unfairness
    
    return res


def fit(epsilon, fairness, transfer=True):

    print("===="*20 + ">>>>"*10 + str(epsilon))

    clf = explainer_map[explainer]
    constraint = metrics_map[fairness](difference_bound=1.0 - epsilon)
    mitigator = ExponentiatedGradient(clf, constraint)
    mitigator.fit(X_train, y_train, sensitive_features=min_features_train)


    # train perf
    y_pred_train = mitigator.predict(X_train)
    cm_train = ConfusionMatrix(min_features_train, maj_features_train, y_pred_train, y_train)
    cm_minority_train, cm_majority_train = cm_train.get_matrix()
    fm_train = Metric(cm_minority_train, cm_majority_train)
    acc_train = accuracy_score(y_train, y_pred_train)
    unf_train = fm_train.fairness_metric(fairness)
    mdl = None
    print("===="*25 + ">>>>"*10 + " unfairness: {}, fidelity {}".format(unf_train, acc_train))

    filename_explainer_pred     = "../quantifying_fairwashing/data/explainers_labels/{}/{}_eps_{}_{}.csv".format(dataset, args.model_class, epsilon, rseed)
    df_explainer_pred           = pd.DataFrame()
    df_explainer_pred["prediction"] = y_pred_train
    df_explainer_pred.to_csv(filename_explainer_pred, encoding='utf-8', index=False)


    # test perf
    y_pred_test = mitigator.predict(X_test)
    cm_test = ConfusionMatrix(min_features_test, maj_features_test, y_pred_test, y_test)
    cm_minority_test, cm_majority_test = cm_test.get_matrix()
    fm_test = Metric(cm_minority_test, cm_majority_test)
    acc_test = accuracy_score(y_test, y_pred_test)
    unf_test = fm_test.fairness_metric(fairness)

    print("===="*25 + ">>>>"*10 + " unfairness test: {}, acc test {}".format(unf_test, acc_test))

    # accuracy on the true labels
    acc_train_true = accuracy_score(y_train_true, y_pred_train)
    acc_test_true = accuracy_score(y_test_true, y_pred_test)

    base_output = [epsilon, acc_train, unf_train, acc_test, unf_test, mdl, acc_train_true, acc_test_true]

    print("===="*10 + " training done ......")

    return (base_output, compute_tranferability(mitigator, epsilon)) if transfer else base_output

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



