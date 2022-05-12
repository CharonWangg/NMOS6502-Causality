import numpy as np
import pandas as pd
from collections import defaultdict
from tests.correlation import correlation_test
from tests.decorrelation import decorr_test
from tests.granger_gc_test import granger_test, granger_test_parallel
from tests.mutual_info import mutual_info_test
from causallearn.search.Granger.Granger import Granger
from pipeline.src.utils.data_util import white_noisen, down_sample, gaussian_noisen
from tqdm import tqdm
import pickle

def run_tests(noise=None, downsample=None):
    res = defaultdict(dict)
    games = ["DonkeyKong"]
    for game in games:
        df = pd.read_csv(f'../envs/{game}/valid_balanced_42.csv')
        nums = len(df)
        # original
        # seqs = np.load("./data_v1/original_3510_1000.npy", mmap_mode='r')
        # stablized
        seqs = np.load(f"../envs/{game}/original_3510_512.npy", mmap_mode='r').astype(np.float32)

        if noise is not None:
            seqs = seqs + gaussian_noisen(seqs, noise)

        if downsample is not None:
            seqs = seqs[:, ::int(1/downsample)]

        # correlation
        res[game]["corr_res"] = correlation_test(df, seqs)
        print(f"{game} correlation test done")
        # decorrelation
        res[game]["decorr_res"] = decorr_test(df, seqs)
        print(f"{game} decorrelation test done")
        # mutual information
        res[game]["mutual_res"] = mutual_info_test(df,seqs)
        print(f"{game} mutual information test done")
        # granger
        # diff to stabilize time series
        # seqs = np.diff(seqs, n=2, axis=-1)
        maxlag = 5
        G = Granger(maxlag=maxlag, significance_level=0.05 / nums)  # Bonferroni correction
        acc = granger_test_parallel(df, seqs, G, num_workers=12)
        res[game]["granger_res"] = acc
        print(f"{game} granger test done")
    return res

if __name__ == "__main__":


    res = {}
    downsample = None
    res["Original"] = run_tests(noise=None, downsample=None)
    for noise in tqdm([0.1, 0.3, 0.5]):
        res[f"Noise={noise}"] = run_tests(noise=noise, downsample=None)

    noise = None
    for downsample in tqdm([0.5, 0.25, 0.125]):
        res[f"Downsample={downsample}"] = run_tests(noise=None, downsample=downsample)

    for (noise, downsample) in tqdm([(0.1, 0.5), (0.3, 0.25), (0.5, 0.125)]):
        res[f"Noise={noise}|Downsample={downsample}"] = run_tests(noise=noise, downsample=downsample)

    with open(f"./test_results_v2.pkl", 'wb') as f:
        pickle.dump(res, f)
