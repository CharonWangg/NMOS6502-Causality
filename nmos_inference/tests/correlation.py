# calculate the Pearson's correlation between two variables
import os

from numpy.random import randn
from numpy.random import seed
from scipy.stats import pearsonr

import pandas as pd
import numpy as np

import warnings
warnings.filterwarnings("ignore")


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def correlation_test(df, seqs):
    """
    Calculate the Pearson's correlation between two variables
    """
    # Create a new dataframe
    df_corr = df.copy()
    # Create a new column with the correlation between the two variables
    df_corr['correlation'] = df_corr.apply(lambda row: pearsonr(seqs[int(row['transistor_1'])], seqs[int(row['transistor_2'])])[0], axis=1)
    # Return the correlation
    correct = sum((sigmoid(df_corr['correlation']).abs() > 0.5).astype(int)==df_corr["label"])
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
#         print(f"{game} Correlation: ", correlation_test(df))
