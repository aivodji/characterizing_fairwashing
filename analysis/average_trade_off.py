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




def compute_average(dataset, model_class, metric, explainer):
    df_list = []
    row_list = []

    for rseed in range(10):
        input_file = "../results{}/{}/{}/{}_{}.csv".format(explainers_map[explainer], dataset, model_class, metrics_map[metric], rseed)
        df_list.append(pd.read_csv(input_file))

    
    for index in range(len(df_list[0])):
        row = {
            'epsilon'                   : df_list[0].iloc[index]['epsilon'],
            'fidelity_sg_mean'          : np.mean([df_list[j].iloc[index]['fidelity_sg'] for j in range(10)]),
            'fidelity_sg_std'           : np.std([df_list[j].iloc[index]['fidelity_sg'] for j in range(10)]),
            'unfairness_sg_mean'        : np.mean([df_list[j].iloc[index]['unfairness_sg'] for j in range(10)]),
            'unfairness_sg_std'         : np.std([df_list[j].iloc[index]['unfairness_sg'] for j in range(10)]),
            'fidelity_test_mean'        : np.mean([df_list[j].iloc[index]['fidelity_test'] for j in range(10)]),
            'fidelity_test_std'         : np.std([df_list[j].iloc[index]['fidelity_test'] for j in range(10)]),
            'unfairness_test_mean'      : np.mean([df_list[j].iloc[index]['unfairness_test'] for j in range(10)]),
            'unfairness_test_std'       : np.std([df_list[j].iloc[index]['unfairness_test'] for j in range(10)]),
            'accuracy_sg_mean'          : np.mean([df_list[j].iloc[index]['accuracy_sg'] for j in range(10)]),
            'accuracy_sg_std'           : np.std([df_list[j].iloc[index]['accuracy_sg'] for j in range(10)]),
            'accuracy_test_mean'        : np.mean([df_list[j].iloc[index]['accuracy_test'] for j in range(10)]),
            'accuracy_test_std'         : np.std([df_list[j].iloc[index]['accuracy_test'] for j in range(10)])
            #'fidelity_gap_sg_mean'      : np.mean([df_list[j].iloc[index]['fidelity_gap_sg'] for j in range(10)]),
            #'fidelity_gap_sg_std'       : np.std([df_list[j].iloc[index]['fidelity_gap_sg'] for j in range(10)])
        }
        
        row_list.append(row)
        
    return pd.DataFrame(row_list)





def main():
    args = parser.parse_args()
    dataset = args.dataset
    metric = args.metric
    model_class = args.model_class
    explainer = args.explainer

    #save direcory
    save_dir = "./results/average_trade_off/{}/{}/{}".format(dataset, explainer, model_class)
    
    os.makedirs(save_dir, exist_ok=True)

    df_average = compute_average(dataset, model_class, metric, explainer)

    output_file = '{}/{}.csv'.format(save_dir, metrics_map[metric], model_class, explainer)
    df_average.to_csv(output_file, encoding='utf-8', index=False)

    
    



if __name__ == '__main__':
    sys.exit(main())


