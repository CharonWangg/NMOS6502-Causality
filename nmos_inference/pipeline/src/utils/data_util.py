import numpy as np
import pandas as pd
from tqdm import tqdm
from joblib import Parallel, delayed
from multiprocessing import cpu_count
from sklearn.model_selection import train_test_split
import yaml
from scipy import signal

Config = yaml.safe_load(open("/home/charon/project/nmos_inference/pipeline/configs/config.yaml"))


# downsample the signal by given factor
def down_sample(seqs, fraction=1.0):
    num = int(len(seqs[0]) * fraction)
    seqs = np.stack([signal.resample(seqs[idx], num) for idx in range(len(seqs))])
    return seqs


def gaussian_noisen(x, mu=0.0, sigma=0.1):
    x = x + np.random.normal(mu, sigma, x.shape)
    return x


# add white noise to the signal by given factor
def white_noisen(x, fraction=0.1):
    snr = 1.0 / fraction
    new_sig = []
    for i in range(x.shape[0]):
        P_signal = np.sum(abs(x)**2)/len(x[i])
        P_noise = P_signal/(10**((snr/10)))
        white_noise = np.random.randn(len(x[i])) * np.sqrt(P_noise)
        new_sig.append(x[i] + white_noise)
    return np.stack(new_sig)


def df_balance(df, decrease=True):
    positive_sample = df[df["label"]==1]
    negative_sample = df[df["label"]==0]
    if decrease:
        if len(positive_sample) > len(negative_sample):
            positive_sample = positive_sample.sample(len(negative_sample))
        else:
            negative_sample = negative_sample.sample(len(positive_sample))
    else:
        # TODO upsample
        pass
    return pd.concat((positive_sample, negative_sample))


def apply_parallel(df, func, num_cpu, tokenizer, valid=False):
    # divide events into chunks
    chunk_list = numpy_split_df(df, num_cpu*10)
    examples_list = Parallel(n_jobs=num_cpu, backend='multiprocessing')(delayed(func)(split_df, df, tokenizer, valid) for split_df in
                                             tqdm(chunk_list, desc="Parallel", total=len(chunk_list)))
    return examples_list


def df_split(df, train_size):
    train, valid = train_test_split(df, train_size=train_size, shuffle=True, random_state=Config["SEED"])
    return train, valid

