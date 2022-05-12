import os
import pandas as pd
import numpy as np
import torch
from ..utils.data_util import white_noisen, down_sample, gaussian_noisen


class NmosData(torch.utils.data.Dataset):
    def __init__(self, ds_type="train", aug=False, aug_prob=0.0, data_dir=None,
                 train_path=None, valid_path=None, test_path=None):
        # Set all input args as attributes
        self.__dict__.update(locals())
        self.aug = self.ds_type=="train" and self.aug
        self.check_files()

    def check_files(self):
        if self.ds_type == "train":
            assert os.path.exists(self.train_path), "Train file not found"
            path = self.train_path
        elif self.ds_type == "valid":
            assert os.path.exists(self.valid_path), "Valid file not found"
            path = self.valid_path
        elif self.ds_type == "test":
            assert os.path.exists(self.test_path), "Test file not found"
            path = self.test_path
        else:
            raise Exception("Invalid dataset type")

        # Load data
        self.time_series_df = pd.read_csv(path)
        self.seqs = np.load(self.data_dir, mmap_mode='r')
        ## Augment data
        if isinstance(self.aug_prob, list):
            # downsample the seqs
            if self.aug_prob[0] > 0:
                self.seqs = self.seqs[:, ::int(1/self.aug_prob[0])]  # down_sample(self.seqs, self.aug_prob[0])
            # # add gaussian noise: 0.1 0.3 0.5
            if self.aug_prob[1] > 0:
                self.seqs = gaussian_noisen(self.seqs, mu=0.0, sigma=self.aug_prob[1])

        print(self.aug_prob, self.seqs.shape)


    def __getitem__(self, idx):
        ## 1st transistor
        seq_t1 = torch.tensor(self.seqs[int(self.time_series_df.iloc[idx]["transistor_1"])], dtype=torch.float32)
        ## 2nd transistor
        seq_t2 = torch.tensor(self.seqs[int(self.time_series_df.iloc[idx]["transistor_2"])], dtype=torch.float32)
        label = torch.tensor(self.time_series_df.iloc[idx]["label"], dtype=torch.int64).repeat(seq_t1.shape[0])

        data = {"t1": seq_t1, "t2": seq_t2}
        return data, label

    def __len__(self):
        return len(self.time_series_df)

