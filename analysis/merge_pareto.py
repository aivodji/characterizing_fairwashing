import csv
import numpy as np
import pandas as pd 
from six.moves import xrange
import os

import argparse
import csv

from ndf import is_pareto_efficient


# parser initialization
parser = argparse.ArgumentParser(description='Analysis of the results')
parser.add_argument('--dataset', type=str, default='adult_income', help='adult_income, compas, default_credit, marketing')
parser.add_argument('--explainer', type=str, default='lm', help='Model class for the explainer. lm for logistic regression, dt for decision tree')

args = parser.parse_args()



#save direcory
save_dir = "./results/pareto_merged/{}".format(args.dataset)
os.makedirs(save_dir, exist_ok=True)

metrics = ['statistical_parity', 'predictive_equality', 'equal_opportunity', 'equalized_odds']
model_classes = ['AdaBoost', 'DNN', 'RF', 'XgBoost']


metrics_map = {
    'statistical_parity'                : 'SP',
    'predictive_equality'               : 'PE',
    'equal_opportunity'                 : 'EOpp',
    'equalized_odds'                    : 'EOdds',
}

df = pd.DataFrame()
filename = '{}/{}_{}.csv'.format(save_dir, args.dataset, args.explainer)



for metric in metrics:
    for model_class in model_classes:
        # Members
        filename_sg = './results/pareto/{}/{}_{}_{}_sg.csv'.format(args.dataset, metric, model_class, args.explainer)
        df_sg = pd.read_csv(filename_sg)
        # Non-Members
        filename_test = './results/pareto/{}/{}_{}_{}_test.csv'.format(args.dataset, metric, model_class, args.explainer)
        df_test = pd.read_csv(filename_test)

        # Merge both and add model class and metric
        df_current = pd.concat([df_sg, df_test], axis=0)
        df_current['model_class']   = [model_class for x in df_current['fidelity']]
        df_current['metric']        = [metrics_map[metric] for x in df_current['fidelity']]
        df = pd.concat([df, df_current], axis=0)

df.to_csv(filename, encoding='utf-8', index=False)
    