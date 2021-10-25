import numpy as np
import pandas as pd
from itertools import product
import os

# inputs
datasets = ['adult_income', 'compas', 'default_credit', 'marketing']
model_classes = ['DNN', 'RF', 'XgBoost', 'AdaBoost']

dataset_map = {
    'adult_income'      : 'Adult Income',
    'compas'            : 'COMPAS',
    'default_credit'    : 'Default Credit',
    'marketing'         :  'Marketing'
}
save_dir = "./results/bbox_label_agreement"
os.makedirs(save_dir, exist_ok=True)

df_list = []
for rseed in range(10):
    row_list = []
    for dataset in datasets:
        for elmt in product(model_classes, model_classes):
            filename_x = '../models/labels/{}/{}_sg_{}.csv'.format(dataset, elmt[0], rseed)
            filename_y = '../models/labels/{}/{}_sg_{}.csv'.format(dataset, elmt[1], rseed)
            df_x = pd.read_csv(filename_x)
            df_y = pd.read_csv(filename_y)
            row = {
                'model_x' : elmt[0],
                'model_y' : elmt[1],
                'dataset' : dataset_map[dataset],
                'fidelity': np.round(100*np.average(df_x.prediction == df_y.prediction), 2)
            }

            row_list.append(row)
    df = pd.DataFrame(row_list)
    df_list.append(df)
    filename_current = '{}/label_agreements_{}.csv'.format(save_dir, rseed)
    df.to_csv(filename_current, encoding='utf-8', index=False)



average_row_list = []
for index in range(len(df_list[0])):
        average_row = {
            'model_x'      : df_list[0].iloc[index]['model_x'],
            'model_y'      : df_list[0].iloc[index]['model_y'],
            'dataset'      : df_list[0].iloc[index]['dataset'],
            'fidelity'     : np.round(np.mean([df_list[j].iloc[index]['fidelity'] for j in range(10)]), 2)
        }
        average_row_list.append(average_row)
df_average = pd.DataFrame(average_row_list)


filename = '{}/label_agreements.csv'.format(save_dir)
df_average.to_csv(filename, encoding='utf-8', index=False)





