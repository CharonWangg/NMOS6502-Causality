import pandas as pd
import numpy as np
from utils.data_util import df_split, df_balance
from pipeline.src.utils.util import fix_all_seeds
from scipy.signal import find_peaks


def remove_constant_seqs(data, label):
    data_dict = {i: data[i] for i in range(data.shape[0])}
    # screen the constant seq and approximately constant seq(fluctuations < 10)
    ## find peaks >= 5
    peaks = []
    num_spike = 5
    for idx, seq in data_dict.items():
        # peaks.append(len(find_peaks(seq)[0])) # spike count
        if len(find_peaks(seq)[0]) >= num_spike:
            peaks.append(idx)

    # screened seqs
    data_dict = {i: data_dict[i] for i in peaks}

    # synchronously change label
    label = {(i, j): label[i, j] for j in data_dict for i in data_dict}

    # return a df
    df = pd.DataFrame()
    df["id"] = range(len(label))
    df["transistor_1"] = [key[0] for key, value in label.items()]
    df["transistor_2"] = [key[1] for key, value in label.items()]
    df["label"] = [value for key, value in label.items()]
    return df


if __name__ == "__main__":
    games = ["DonkeyKong", "Pitfall", "SpaceInvaders"]

    for game in games:
        data = np.load(f"../envs/{game}/original_3510_512.npy", mmap_mode='r')
        label = np.load(f"../envs/{game}/causal_1c_label.npy", mmap_mode='r')
        df = remove_constant_seqs(data, label)  # exclude the constant and approximately constant seq
        df = df_balance(df, decrease=True)  # balance the causal and non-causal data

        fix_all_seeds(42)  # fix all seeds for reproducibility of results
        train, valid = df_split(df, train_size=0.8) # use 0.8 as split ratio since balanced data is much less
        train.to_csv(f"../envs/{game}/train_balanced_42.csv")
        valid.to_csv(f"../envs/{game}/valid_balanced_42.csv")
        print(f"{game} data is ready")

    # stable the data by doing second oder Differencing
    # seqs = np.load("../data_v1/original_3510_400.npy", mmap_mode='r')
    # seqs = np.diff(seqs, n=2, axis=-1)
    # np.save("../data_v1/2nd_diff_3510_400.npy", seqs)


