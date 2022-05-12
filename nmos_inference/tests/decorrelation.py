import numpy as np
import pandas as pd


def get_decorr_mat(x, y):
    """
    Returns the decorrelation matrix of x and y.
    """
    cov = np.cov(x, y)
    U, S, V = np.linalg.svd(cov)
    # assert np.allclose(np.dot(U, V), np.eye(U.shape[0]))
    #decorr_mat = np.linalg.inv(eig_vecs)
    return U[1, 0]


def decorr_test(df, seqs):
    """
    Calculate the Mutual Information Score between two variables
    """
    # Create a new dataframe
    df_corr = df.copy()
    # Create a new column with the correlation between the two variables
    df_corr['decorr'] = df_corr.apply(lambda row: get_decorr_mat(seqs[int(row['transistor_1'])], seqs[int(row['transistor_2'])]), axis=1)
    # Return the correlation
    correct = sum((df_corr['decorr'].abs() > 0.5).astype(int)==df_corr["label"])
    total = len(df_corr)
    # return df_corr
    return correct/total


def plot_test():
    """
    Plot the correlation between two variables
    """
    # Create a new dataframe
    df_corr = df.copy()
    # Create a new column with the correlation between the two variables
    df_corr['decorr'] = df_corr.apply(lambda row: get_decorr_mat(seqs[int(row['transistor_1'])], seqs[int(row['transistor_2'])]), axis=1)
    # Plot the correlation
    df_corr.plot.scatter(x='transistor_1', y='transistor_2', c='decorr')
    plt.show()


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
#         print(f"{game} Decorrelation: ", decorr_test(df))