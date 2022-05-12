import pandas as pd
import numpy as np
from tqdm import tqdm
from joblib import Parallel, delayed
from multiprocessing import cpu_count
from causallearn.search.Granger.Granger import Granger
import warnings
warnings.filterwarnings("ignore")


def numpy_split_df(df, split_num):
    lst_index = list(map(lambda a: a.tolist(), np.array_split([i for i in range(len(df))], split_num)))
    chunk_list = []
    for idx in lst_index:
        df_split = df.iloc[idx[0]: idx[-1] + 1]
        chunk_list.append(df_split)
    return chunk_list


def apply_parallel(df, func, num_cpu, seqs, G):
    # divide events into chunks
    chunk_list = numpy_split_df(df, num_cpu*10)
    examples_list = Parallel(n_jobs=num_cpu, backend='multiprocessing')(delayed(func)(split_df, seqs, G)
                                                                        for split_df in tqdm(chunk_list, desc="Parallel", total=len(chunk_list)))
    return examples_list

def granger_predict(ts, G):
    try:
        adj_mat = G.granger_test_2d(ts)[1]
    except ValueError:
        adj_mat = np.zeros((2, 2*5))
    adj_by_time_lag = [adj_mat[:, i:i + 2] for i in range(0, len(adj_mat[0]), 2)]
    # for p in adj_by_time_lag:
    #     if np.sum(p) > 2:
    adj_by_time_pred = []
    for p in adj_by_time_lag:
        if p[1, 0] == 1:
            # t1 causing t2 --> t1 cause t2 == label(1)
            adj_by_time_pred.append(1)
        else:
            adj_by_time_pred.append(0)
    return adj_by_time_pred


def granger_test(df, seqs, G):
    total = len(df)
    correct = []
    preds = []
    # for idx, row in tqdm(df.iterrows(), total=total):
    for idx, row in df.iterrows():
        t1 = seqs[int(row["transistor_1"])].reshape(-1, 1)
        t2 = seqs[int(row["transistor_2"])].reshape(-1, 1)
        ts = np.concatenate((t1, t2), axis=1)
        pred = granger_predict(ts, G)
        preds.append(pred)
    # acc = np.sum(np.stack(correct), axis=0) / total
    return preds


def granger_test_parallel(df, seqs, G, num_workers=8):
    total = len(df)
    preds = apply_parallel(df, granger_test, num_workers, seqs, G)
    preds = [pp for p in preds for pp in p]
    correct = [p==df["label"].iloc[idx] for idx, p in enumerate(preds)]
    acc = np.sum(np.stack(correct), axis=0) / total

    return acc

#
# if __name__ == "__main__":
#     for game in ["DonkeyKong", "Pitfall", "SpaceInvaders"]:
#         print("*****************************")
#         print("Game: {}".format(game))
#         df = pd.read_csv(f'../envs/{game}/valid_balanced_42.csv')
#         # original
#         # seqs = np.load("./data_v1/original_3510_1000.npy", mmap_mode='r')
#         # stablized
#         seqs = np.load(f"../envs/{game}/original_3510_512.npy", mmap_mode='r')
#
#         seqs = np.diff(seqs, n=2, axis=-1)
#
#         # granger test
#         nums = len(df)
#         maxlag = 10
#         # "supervised" way to tune the time_lag parameter
#         G = Granger(maxlag=maxlag, significance_level=0.05 / nums)  # Bonferroni correction
#         acc = granger_test_parallel(df, seqs, G)
#         for i in range(maxlag):
#             print(f"When time lag is {i}: ", acc[i])
