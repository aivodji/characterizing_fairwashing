import csv
import numpy as np
import pandas as pd 
from six.moves import xrange
import os
import sys

import argparse
import csv

from ndf import is_pareto_efficient


# parser initialization
parser = argparse.ArgumentParser(description='Analysis of the results')
parser.add_argument('--dataset', type=str, default='adult_income', help='adult_income, compas, default_credit, marketing')
parser.add_argument('--metric', type=int, default=1, help='Fairness metric. 1: SP, 3: PE, 4: EOpp, 5: EOdds')
parser.add_argument('--model_class', type=str, default='DNN', help='DNN, RF, AdaBoost, XgBoost')
parser.add_argument('--explainer', type=str, default='lm', help='Model class for the explainer. lm for logistic regression, dt for decision tree')



explainers_map = {
    'rl' : '_rl',
    'lm' : '_lm',
    'dt' : '_dt'
}

metrics_map = {
    1 : 'statistical_parity',
    3 : 'predictive_equality',
    4 : 'equal_opportunity',
    5 : 'equalized_odds',
}

eval_group_map = {
    'sg' : "Members",
    'test'  : "Non-Members"
}


def compute_front(df_average, output_file, eval_group):

    errors     = 1.0 - df_average['fidelity_{}_mean'.format(eval_group)]
    unfairness = df_average['unfairness_{}_mean'.format(eval_group)]
    fidelity_std   = df_average['fidelity_{}_std'.format(eval_group)]
    #fidelity_gap_sg_mean = df_average['fidelity_gap_sg_mean']

    pareto_input = [[error, unf] for (error, unf) in zip(errors, unfairness)]
    pareto_input = np.array(pareto_input)
    msk = is_pareto_efficient(pareto_input)


    df = pd.DataFrame()          
    df['fidelity']         =  [1.0 - errors[i] for i in xrange(len(errors)) if msk[i]]
    df['unfairness']       =  [unfairness[i] for i in xrange(len(errors)) if msk[i]]
    df['fidelity_std']     =  [fidelity_std[i] for i in xrange(len(errors)) if msk[i]]
    #df['fidelity_gap_sg_mean']     =  [fidelity_gap_sg_mean[i] for i in xrange(len(errors)) if msk[i]]

    df['group']            = [eval_group_map[eval_group] for i in xrange(len(errors)) if msk[i]]

    df.to_csv(output_file, encoding='utf-8', index=False)



def main():
    args = parser.parse_args()
    dataset = args.dataset
    metric = args.metric
    model_class = args.model_class
    explainer = args.explainer

    #save direcory
    save_dir = "./results/pareto/{}".format(dataset)
    
    os.makedirs(save_dir, exist_ok=True)

    input_file = "./results/average_trade_off/{}/{}/{}/{}.csv".format(dataset, explainer, model_class, metrics_map[metric])
    df_average = pd.read_csv(input_file)

    output_file_train = '{}/{}_{}_{}_sg.csv'.format(save_dir, metrics_map[metric], model_class, explainer)
    compute_front(df_average, output_file_train, 'sg')

    output_file_test = '{}/{}_{}_{}_test.csv'.format(save_dir, metrics_map[metric], model_class, explainer)
    compute_front(df_average, output_file_test, 'test')
    



if __name__ == '__main__':
    sys.exit(main())


