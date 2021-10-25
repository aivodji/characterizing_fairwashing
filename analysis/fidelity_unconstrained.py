import numpy as np
import pandas as pd 
import os
import argparse


# parser initialization
parser = argparse.ArgumentParser(description='Analysis of the results')
parser.add_argument('--group', type=str, default='sg', help='suing group: sg, otherwise: test')
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

group_map = {
    'sg' : 'Members',
    'test'  : 'Non-Members'
}

dataset_map = {
    'adult_income'      : 'Adult Income',
    'COMPAS'            : 'COMPAS',
    'default_credit'    : 'Default Credit',
    'marketing'         : 'Marketing'
}



def find_position(df, group):
    df_filtered = df.loc[df['epsilon'] == 0.0]
    fid = df_filtered['fidelity_{}_mean'.format(group)].iloc[0]
    return fid


explainer = args.explainer
group = args.group
datasets = ['adult_income', 'COMPAS', 'default_credit', 'marketing']
model_classes = ['DNN', 'RF', 'AdaBoost', 'XgBoost']
metrics = [1, 3, 4, 5]

row_list = []

for  dataset in datasets:
    for model_class in model_classes:
        for metric in metrics:
            input_file = "./results/average_trade_off/{}/{}/{}/{}.csv".format(dataset, explainer, model_class, metrics_map[metric])
            df = pd.read_csv(input_file)
            fid = find_position(df, group)
            row = {
                'dataset'       : dataset_map[dataset],
                'metric'        : metrics_abrv[metric],
                'fidelity'      : np.round(100*fid, 2),
                'model_class'   : model_class
            }
            row_list.append(row)

save_dir = "./results/fidelity_unconstrained"
os.makedirs(save_dir, exist_ok=True)

df = pd.DataFrame(row_list)
filename = '{}/{}_{}.csv'.format(save_dir, group, explainer)
df.to_csv(filename, encoding='utf-8', index=False)