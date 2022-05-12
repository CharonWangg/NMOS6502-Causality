# calculate the Mutual Information Score between two variables
import os

from numpy.random import randn
from numpy.random import seed
from sklearn.metrics import mutual_info_score

import pandas as pd
import numpy as np

import warnings
warnings.filterwarnings("ignore")

def mutual_info_test(df, seqs):
    """
    Calculate the Mutual Information Score between two variables
    """
    # Create a new dataframe
    df_corr = df.copy()
    # Create a new column with the correlation between the two variables
    df_corr['mutual_info_score'] = df_corr.apply(lambda row: mutual_info_score(seqs[int(row['transistor_1'])], seqs[int(row['transistor_2'])]), axis=1)
    # Return the correlation
    correct = sum((df_corr['mutual_info_score'].abs() > 0.5).astype(int)==df_corr["label"])
    total = len(df_corr)
    # return df_corr
    return correct/total



#
# if __name__ == "__main__":
#     games = ["DonkeyKong", "Pitfall", "SpaceInvaders"]
#
#     for game in games:
#         df = pd.read_csv(f'../envs/{game}/valid_balanced_42.csv')
#         # original
#         # seqs = np.load("./data_v1/original_3510_1000.npy", mmap_mode='r')
#         # stablized
#         seqs = np.load(f"../envs/{game}/original_3510_512.npy", mmap_mode='r')
#         # seqs = np.diff(seqs, n=2, axis=-1)
#
#         # granger test
#         nums = len(df)
#         # "supervised" way to tune the time_lag parameter
#         print(f"{game} mutual information score: ", mutual_info_test(df))
