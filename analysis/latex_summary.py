import pandas as pd 
import numpy as np 
import os
from tabulate import tabulate

df = pd.read_csv('results/summary/summary.csv')


df.index = pd.MultiIndex.from_frame(df[["Dataset", "Model", "Partition"]])
df = df.drop(["Dataset", "Model", "Partition"], axis=1)


save_dir = "./results/latex"

os.makedirs(save_dir, exist_ok=True)

filename = '{}/perfs.tex'.format(save_dir)

df.to_latex(filename, multirow=True, index=True)