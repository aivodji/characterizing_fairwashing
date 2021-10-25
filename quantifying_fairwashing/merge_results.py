import argparse
import os
import pandas as pd 
import numpy as np


def get_explainer_perfs(df, id):
    df_filtered = df.loc[df['epsilon'] == id]
    fidelity = df_filtered['fidelity_sg'].iloc[0]
    unfairness = df_filtered['unfairness_sg'].iloc[0]
    return fidelity, unfairness

# parser initialization
parser = argparse.ArgumentParser(description='Analysis of the results')
parser.add_argument('--dataset', type=str, default='adult_income', help='adult_income, compas, default_credit, marketing')
parser.add_argument('--rseed', type=int, default=0, help='random seed: choose between 0 - 9')
parser.add_argument('--model_class', type=str, default='DNN', help='Model class: DNN, RF, SVM, XgBoost, AdaBoost')


args = parser.parse_args()

dataset = args.dataset
rseed = args.rseed
model_class = args.model_class



#save direcory
save_dir = "./results"
os.makedirs(save_dir, exist_ok=True)

vals = [0.90, 0.91, 0.92, 0.93, 0.94, 0.95, 0.96, 0.97, 0.98, 0.99]

df = pd.DataFrame()
filename = '{}/{}_{}_{}.csv'.format(save_dir, dataset, model_class, rseed)

filename_explainer_perfs = '../results_lm/{}/{}/statistical_parity_{}.csv'.format(dataset, model_class, rseed)
df_explainer_perfs = pd.read_csv(filename_explainer_perfs)

fid_list = []
for pos in range(10):
    filename_current = '{}/{}_{}_{}_{}.csv'.format(save_dir, dataset, rseed, model_class,  pos + 1)
    df_current = pd.read_csv(filename_current)
    fidelity, _ = get_explainer_perfs(df_explainer_perfs, vals[pos])
    fidelity = np.round(fidelity, 3)
    if (fidelity not in fid_list):
        df_current["fidelity_explainer"] = [fidelity]
        df = pd.concat([df, df_current], axis=0)
        fid_list.append(fidelity)

df.to_csv(filename, encoding='utf-8', index=False)
    