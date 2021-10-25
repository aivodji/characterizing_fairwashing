import csv
import numpy as np
import pandas as pd 
from six.moves import xrange
import os
import sys
import argparse
import csv



# parser initialization
parser = argparse.ArgumentParser(description='Analysis of the results')
parser.add_argument('--dataset', type=str, default='adult_income', help='adult_income, compas, default_credit, marketing')
parser.add_argument('--rseed', type=int, default=0, help='random seed: choose between 0 - 9')
parser.add_argument('--epsilon', type=float, default=0.05, help='Value of the fairness constraint')
parser.add_argument('--explainer', type=str, default='lm', help='Model class for the explainer. lm for logistic regression, dt for decision tree')

args = parser.parse_args()


metrics_map = {
    1 : 'statistical_parity',
    3 : 'predictive_equality',
    4 : 'equal_opportunity',
    5 : 'equalized_odds'
}

metrics_abrv = {
    1 : 'SP',
    3 : 'PE',
    4 : 'EOpp',
    5 : 'EOdds'
}

dataset_map = {
    'adult_income'      : 'Adult Income',
    'compas'            : 'COMPAS',
    'default_credit'    : 'Default Credit',
    'marketing'         :  'Marketing'
}


def find_position(df, model_class, epsilon):
    df_filtered = df.loc[df['unfairness_{}'.format(model_class)] <= epsilon].sort_values(by=['fidelity_{}'.format(model_class)], ascending=False)
    pos = df_filtered['epsilon'].iloc[0]
    return pos


def compute_table(dataset, rseed, explainer, metric, epsilon):
    filename_label_agreements = './results/bbox_label_agreement/label_agreements_{}.csv'.format(rseed)
    df_label_agreements = pd.read_csv(filename_label_agreements)
    model_classes = ['DNN', 'RF', 'AdaBoost', 'XgBoost']
    row_list = []
    for mdl_master in model_classes:
        input_master = "../results_{}/{}/{}/{}_transfer_{}.csv".format(explainer, dataset, mdl_master, metrics_map[metric], rseed)
        df_master = pd.read_csv(input_master)
        pos = find_position(df_master, mdl_master, epsilon)
        for mdl_student in model_classes:
            row = {}
            row['master']     = mdl_master
            row['student']    = mdl_student
            row['fidelity']   = df_master[df_master['epsilon']==pos]['fidelity_{}'.format(mdl_student)].iloc[0]
            row['unfairness'] = df_master[df_master['epsilon']==pos]['unfairness_{}'.format(mdl_student)].iloc[0]
            unf = np.round(row['unfairness'], 2)
            if (unf <= epsilon):
                row['check'] = "yes"
            else :
                row['check'] = "no"
            
            row['label_agreement'] = df_label_agreements[(df_label_agreements['dataset']==dataset_map[args.dataset]) & (df_label_agreements['model_x']==mdl_master) & (df_label_agreements['model_y']==mdl_student)]['fidelity'].iloc[0]

            row_list.append(row)

    return pd.DataFrame(row_list)


def main():
    args = parser.parse_args()
    dataset = args.dataset
    explainer = args.explainer
    epsilon = args.epsilon
    rseed = args.rseed

    #save direcory
    save_dir = "./results/analysis_transferability_per_seed/{}".format(dataset)
    os.makedirs(save_dir, exist_ok=True)

    output_file='{}/transfer_{}_{}_eps={}.csv'.format(save_dir, explainer, rseed, epsilon)
    df_all = pd.DataFrame()

    for metric in [1, 3, 4, 5]:
        df = compute_table(dataset, rseed, explainer, metric, epsilon)
        df['metric'] = [metrics_abrv[metric] for x in df['fidelity']]
        df_all = pd.concat([df_all, df], axis=0)

    df_all.to_csv(output_file, encoding='utf-8', index=False)
    



if __name__ == '__main__':
    sys.exit(main())




