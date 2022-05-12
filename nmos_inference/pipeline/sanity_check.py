import pickle
from collections import defaultdict

import pytorch_lightning as pl
from pytorch_lightning import Trainer
from src.model import ModelInterface
from src.data import DataInterface
from src.utils.config_util import args_setup, load_model_path_by_hparams, configure_args

def main(args, load_path=None):
    pl.seed_everything(args.seed)
    print(f'load_path: {load_path}')
    data_module = DataInterface(**vars(args))

    if load_path is None:
        model = ModelInterface(**vars(args))
    else:
        model = ModelInterface(**vars(args))
        print('Found checkpoint, stop training.')

    trainer = Trainer.from_argparse_args(args)
    # only need to load the ckpt once in a process,
    # and it will be forever loaded in the trainer and model
    pred = trainer.test(model, data_module, ckpt_path=load_path)
    return pred


if __name__ == '__main__':
    cfg = "./configs/config.yaml"
    # args.data_path = "../data_v1/original_3510_400.npy"
    games = ["DonkeyKong", "Pitfall", "SpaceInvaders"]
    res = defaultdict(dict)
    # For LSTM model
    for game in games:
        hparams = {"model_name": "nmos_lstm", "encoder_hidden_size": 32,
                   "arg_comp_hidden_size": 32,
                   "dropout": 0.3,
                   "save_dir": "/home/charon/project/nmos_inference/models/LSTM/",
                   "log_dir": "/home/charon/project/nmos_inference/log/",
                   "gpus": [0],
                    "test_path": f"../envs/{game}/valid_balanced_42.csv"}
        args = configure_args(cfg, hparams)
        load_path = "/home/charon/project/nmos_inference/models/LSTM/model_name=nmos_lstm-dataset=nmos-lr=0.001-wd=0.05-batch=64-l1=0.0-l2=0.0-dropout=0.3-opt=Adam-loss=diy-seed=42-epoch=43-val_epoch_acc=0.786-val_epoch_auc=0.851.ckpt"
        pred = main(args, load_path)[0]
        res[game]["lstm"] = pred


    # For TCN model
    for game in games:
        hparams = {"model_name": "nmos_tcn", "encoder_hidden_size": [32, 32],
                   "arg_comp_hidden_size": 32,
                   "dropout": 0.3,
                   "save_dir": "/home/charon/project/nmos_inference/models/LSTM/",
                   "log_dir": "/home/charon/project/nmos_inference/log/",
                   "gpus": [0],
                    "test_path": f"../envs/{game}/valid_balanced_42.csv"}
        args = configure_args(cfg, hparams)
        load_path = "/home/charon/project/nmos_inference/models/TCN/uid=5-model_name=nmos_tcn-dataset=nmos-val_epoch_acc=0.668-val_epoch_auc=0.750.ckpt"
        pred = main(args, load_path)[0]
        res[game]["tcn"] = pred

    # For TRM model
    # for game in games:
    #     hparams = {"model_name": "nmos_trm", "encoder_hidden_size": 32,
    #                "arg_comp_hidden_size": 32,
    #                "dropout": 0.3,
    #                "save_dir": "/home/charon/project/nmos_inference/models/LSTM/",
    #                "log_dir": "/home/charon/project/nmos_inference/log/",
    #                "gpus": [0],
    #                 "test_path": f"../envs/{game}/valid_balanced_42.csv"}
    #     args = configure_args(cfg, hparams)
    #     load_path = "/home/charon/project/nmos_inference/models/TRM/model_name=nmos_trm-dataset=nmos-lr=0.0005-wd=0.05-batch=256-l1=0.0-l2=0.0-dropout=0.0-opt=Adam-loss=diy-seed=42-epoch=8-val_epoch_acc=0.616-val_epoch_auc=0.660.ckpt"
    #     pred = main(args, load_path)[0]
    #     res[game]["trm"] = pred

    with open("../results/transfer_learning_result.pkl", "wb") as f:
        pickle.dump(res, f)

