import numpy as np
import pandas as pd 
import os
import argparse


# parser initialization
parser = argparse.ArgumentParser(description='Analysis of the results')
parser.add_argument('--group', type=str, default='train', help='suing group: train, otherwise: test')
parser.add_argument('--explainer', type=str, default='lm', help='Model class for the explainer. lm for logistic regression, dt for decision tree')
parser.add_argument('--all', action='store_true', help='Use all the 4 black_box: otherwise use 3, for simplicity.')
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



def find_position(df, df_ref_val, model_class, metric, group, prop):
    df_ref_val_filtered = df_ref_val.loc[(df_ref_val['model_class']==model_class) & (df_ref_val['metric']==metrics_abrv[metric]) & (df_ref_val['group']==group_map[group])]
    ref_val = df_ref_val_filtered['unfairness'].iloc[0]

    df_filtered = df.loc[df['unfairness_{}_mean'.format(group)] <= prop*ref_val].sort_values(by=['fidelity_{}_mean'.format(group)], ascending=False)
    fid = df_filtered['fidelity_{}_mean'.format(group)].iloc[0] if len(df_filtered) > 0 else 0.0
    return fid

save_dir = "./results/half_unfairness"
os.makedirs(save_dir, exist_ok=True)

all_bbox = args.all
explainer = args.explainer
group = args.group
datasets = ['adult_income', 'COMPAS', 'default_credit', 'marketing']
model_classes = ['DNN', 'RF', 'AdaBoost', 'XgBoost'] if all_bbox else ['DNN', 'RF', 'AdaBoost']
suffix = "_all_bbox" if all_bbox else ""
metrics = [1, 3, 4, 5]

row_list = []

filename_uncons = './results/fidelity_unconstrained/{}_{}.csv'.format(group, explainer)
df_uncons = pd.read_csv(filename_uncons)

for  dataset in datasets:
    filename_ref = './results/unfairness_bbox/{}.csv'.format(dataset)
    df_ref_val = pd.read_csv(filename_ref)
    for model_class in model_classes:
        for metric in metrics:
            input_file = "./results/average_trade_off/{}/{}/{}/{}.csv".format(dataset, explainer, model_class, metrics_map[metric])
            df = pd.read_csv(input_file)
            fid = find_position(df, df_ref_val, model_class, metric, args.group, 0.5)
            row = {
                'dataset'       : dataset_map[dataset],
                'metric'        : metrics_abrv[metric],
                'fidelity'      : np.round(100*fid, 2),
                'model_class'   : model_class,
                'fidelity_uncons' : df_uncons[(df_uncons['dataset']==dataset_map[dataset]) & (df_uncons['metric']==metrics_abrv[metric]) & (df_uncons['model_class']==model_class)]['fidelity'].iloc[0]
            }
            row_list.append(row)




df = pd.DataFrame(row_list)
filename = '{}/{}{}_{}.csv'.format(save_dir, group, suffix, explainer)
df.to_csv(filename, encoding='utf-8', index=False)