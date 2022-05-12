import yaml
import pandas as pd
from argparse import ArgumentParser
from pytorch_lightning import Trainer
from .log_util import match_hparams, static_hparams
import argparse
import os
import ast
from pathlib2 import Path


# load model in uid format by searching its hparams in a csv file
def load_model_path_by_csv(root, hparams=None, mode='train'):
    if hparams is None:
        return None
    elif isinstance(hparams, dict):
        pass
    elif isinstance(hparams, object):
        hparams = vars(hparams)

    if mode == "train":
        # concat the root and .csv
        csv_path = root + "models_log.csv"
        hparams = {k: v for k, v in hparams.items() if k in static_hparams}
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            df, cur_uid = match_hparams(hparams, df)
            if (pd.isna(df.loc[df["uid"]==cur_uid, "model_path"])).any():
                return None
            else:
                # return the best val_acc model path
                load_path = df.groupby(by="uid").get_group(cur_uid)
                load_path = load_path.iloc[0]["model_path"] if load_path.shape[0] == 1 else load_path.sorted_values(by="val_acc", ascending=False)["model_path"].values[0]
                print('Loading the best model of uid {} from {}'.format(cur_uid, load_path))
                return load_path if not pd.isna(load_path) else None
        else:
            print("The csv file does not exist, will create one during training.")
            return None
    elif mode == "inference":
        # concat the root and .csv
        csv_path = root + "models_log.csv"
        df = pd.read_csv(csv_path)
        # exclude the inference hparams that could be different from training hparams
        hparams.pop("inference_seed")
        hparams.pop("gpus")
        try:
            for k, v in hparams.items():
                df = df.groupby(by=k).get_group(v)
            # return the best val_acc model path
            load_path = df.iloc[0]["model_path"] if df.shape[0] == 1 else df.sorted_values(by="val_acc", ascending=False)["model_path"].values[0]

            return load_path
        except KeyError:
            print("The hparams are not in the csv file.")
            return None


def load_model_path_by_hparams(root, hparams=None):
    """ When best = True, return the best model's path in a directory
        by selecting the best model with largest epoch. If not, return
        the last model saved. You must provide at least one of the
        first three args.
    Args:
        root: The root directory of checkpoints. It can also be a
            model ckpt file. Then the function will return it.
        version: The name of the version you are going to load.
        v_num: The version's number that you are going to load.
        best: Whether return the best model.
    """
    if hparams is None:
        return None
    elif isinstance(hparams, object):
        hparams = hparams.__dict__

    if Path(root).is_file():
        return root

    # in case model save dir is exist
    if not Path(root).exists():
        Path(root).mkdir(parents=True)
    # in case epoch is not set
    if "epoch" not in hparams:
        hparams["epoch"] = "*"
    # match the hparams to the file name
    files = [str(i) for i in list(Path(root).iterdir())
             if f'model_name={hparams["model_name"]}'.lower() in str(i) and
             f'dataset={hparams["dataset"]}'.lower() in str(i) and
             f'epoch={hparams["epoch"]}'.lower() in str(i) and
             f'lr={hparams["lr"]}'.lower() in str(i) and
             f'weight_decay={hparams["weight_decay"]}'.lower() in str(i) and
             f'train_batch_size={hparams["train_batch_size"]}'.lower() in str(i) and
             f'l1={hparams["l1"]}'.lower() in str(i) and
             f'l2={hparams["l2"]}'.lower() in str(i) and
             f'dropout={hparams["dropout"]}'.lower() in str(i) and
             f'optimizer={hparams["optimizer"]}'.lower() in str(i) and
             f'loss={hparams["loss"]}'.lower() in str(i) and
             f'seed={hparams["seed"]}'.lower() in str(i)
             in str(i).lower()]

    if not files:
        return None
    else:
        print('Loading model from {}'.format(files[-1]))
        return files[-1]

def args_setup(cfg_path='configs/config.yaml'):
    cfg = yaml.safe_load(open(cfg_path))
    parser = ArgumentParser()
    # init
    parser.add_argument('--cfg', type=str, default=cfg_path, help='config file path')
    parser.add_argument('--seed', default=cfg["SEED"], type=int)

    # data
    parser.add_argument('--dataset', default=cfg["DATA"]["DATASET"], type=str)
    parser.add_argument('--data_dir', default=cfg["DATA"]["DATA_PATH"], type=str)
    parser.add_argument('--train_path', default=cfg["DATA"]["TRAIN_PATH"], type=str)
    parser.add_argument('--valid_path', default=cfg["DATA"]["VALID_PATH"], type=str)
    parser.add_argument('--test_path', default=cfg["DATA"]["TEST_PATH"], type=str)
    parser.add_argument('--train_batch_size', default=cfg["DATA"]["TRAIN_BATCH_SIZE"], type=int)
    parser.add_argument('--valid_batch_size', default=cfg["DATA"]["VALID_BATCH_SIZE"], type=int)
    parser.add_argument('--test_batch_size', default=cfg["DATA"]["TEST_BATCH_SIZE"], type=int)
    parser.add_argument('--train_size', default=cfg["DATA"]["TRAIN_SIZE"], type=int)
    parser.add_argument('--num_classes', default=cfg["DATA"]["NUM_CLASSES"], type=int)
    parser.add_argument('--num_workers', default=cfg["DATA"]["NUM_WORKERS"], type=int)

    # data augmentation
    parser.add_argument('--aug', default=cfg["DATA"]["AUG"], type=bool)
    parser.add_argument('--aug_prob', default=cfg["DATA"]["AUG_PROB"], type=float)
    parser.add_argument('--aug_specific', default="[0.0, 0.0]", type=str)

    # train
    parser.add_argument('--lr', default=cfg["OPTIMIZATION"]["LR"], type=float)

    # LR Scheduler
    parser.add_argument('--lr_scheduler', default=cfg["OPTIMIZATION"]["LR_SCHEDULER"],
                        choices=['step', 'multistep', 'cosine', 'constant', 'one_cycle', 'cyclic', 'plateau', 'nmos'], type=str)
    parser.add_argument('--lr_warmup_epochs', default=cfg["OPTIMIZATION"]["LR_WARMUP_EPOCHS"], type=int)
    parser.add_argument('--lr_decay_steps', default=cfg["OPTIMIZATION"]["LR_DECAY_STEPS"], type=int)
    parser.add_argument('--lr_decay_rate', default=cfg["OPTIMIZATION"]["LR_DECAY_RATE"], type=float)
    parser.add_argument('--lr_decay_min_lr', default=cfg["OPTIMIZATION"]["LR_DECAY_MIN_LR"], type=float)

    # Training Info
    parser.add_argument('--model_name', default=cfg["MODEL"]["NAME"], type=str)
    parser.add_argument('--loss', default=cfg["OPTIMIZATION"]["LOSS"],
                        choices=["cross_entropy", "binary_cross_entropy", "l1", "l2", "diy"], type=str)
    parser.add_argument('--margin', default=cfg["OPTIMIZATION"]["MARGIN"], type=float)
    parser.add_argument('--weight_decay', default=cfg["OPTIMIZATION"]["WEIGHT_DECAY"], type=float)
    parser.add_argument('--momentum', default=cfg["OPTIMIZATION"]["MOMENTUM"], type=float)
    parser.add_argument('--optimizer', default=cfg["OPTIMIZATION"]["OPTIMIZER"], choices=["Adam", "SGD", "RMSprop", "DIY"],
                        type=str)
    parser.add_argument('--l1', default=cfg["MODEL"]["L1"], type=float)
    parser.add_argument('--l2', default=cfg["MODEL"]["L2"], type=float)
    parser.add_argument('--patience', default=cfg["OPTIMIZATION"]["PATIENCE"], type=int)
    parser.add_argument('--log_dir', default=cfg["LOG"]["PATH"], type=str)
    parser.add_argument('--save_dir', default=cfg["MODEL"]["SAVE_DIR"], type=str)
    parser.add_argument('--exp_name', default=cfg["LOG"]["NAME"], type=str)
    parser.add_argument('--run', default=0, type=int)

    # Model Hyperparameters
    parser.add_argument('--input_size', default=cfg["MODEL"]["INPUT_SIZE"], type=int)
    parser.add_argument('--input_length', default=cfg["MODEL"]["INPUT_LENGTH"], type=int)
    parser.add_argument('--encoder_name', default=cfg["MODEL"]["ENCODER"]["NAME"], type=str)
    parser.add_argument('--encoder_num_layers', default=cfg["MODEL"]["ENCODER"]["NUM_LAYERS"], type=int)
    parser.add_argument('--encoder_hidden_size', default=cfg["MODEL"]["ENCODER"]["HIDDEN_SIZE"], type=str)
    parser.add_argument('--arg_comp_hidden_size', default=cfg["MODEL"]["ARG_COMP"]["HIDDEN_SIZE"], type=int)
    parser.add_argument('--arg_comp_output_size', default=cfg["MODEL"]["ARG_COMP"]["OUTPUT_SIZE"], type=int)
    parser.add_argument('--event_comp_hidden_size', default=cfg["MODEL"]["EVENT_COMP"]["HIDDEN_SIZE"], type=int)
    parser.add_argument('--event_comp_output_size', default=cfg["MODEL"]["EVENT_COMP"]["OUTPUT_SIZE"], type=int)
    parser.add_argument('--dropout', default=cfg["MODEL"]["DROPOUT"], type=float)

    # Add pytorch lightning's args to parser as a group.
    parser = Trainer.add_argparse_args(parser)

    # Reset Some Default Trainer Arguments' Default Values
    parser.set_defaults(max_epochs=cfg["OPTIMIZATION"]["MAX_EPOCHS"])
    parser.set_defaults(accumulate_grad_batches=cfg["OPTIMIZATION"]["ACC_GRADIENT_STEPS"])
    parser.set_defaults(accelerator="auto")
    parser.set_defaults(gpus=cfg["GPUS"])
    parser.set_defaults(strategy=cfg["STRATEGY"])
    parser.set_defaults(precision=cfg["PRECISION"])
    parser.set_defaults(limit_train_batches=cfg["DATA"]["NUM_TRAIN"], type=float)
    parser.set_defaults(limit_val_batches=cfg["DATA"]["NUM_VALID"], type=float)
    parser.set_defaults(limit_test_batches=cfg["DATA"]["NUM_TEST"], type=float)
    parser.set_defaults(limit_predict_batches=cfg["DATA"]["NUM_PREDICT"], type=float)
    parser.set_defaults(deterministic=True)

    args = parser.parse_args()
    # List Arguments
    args.img_mean = cfg["DATA"]["IMG_MEAN"]
    args.img_std = cfg["DATA"]["IMG_STD"]
    args.aug_specific = ast.literal_eval(args.aug_specific)
    args.aug_prob = args.aug_specific
    # parse the list/int arguments
    if isinstance(args.encoder_hidden_size, str):
        if '[' in args.encoder_hidden_size:
            args.encoder_hidden_size = [int(x) for x in ast.literal_eval(args.encoder_hidden_size)]
        else:
            args.encoder_hidden_size = int(args.encoder_hidden_size)
    # parse the strategy
    if str(args.strategy).lower() == "none":
        args.strategy = None
    # parse the continuous devices argument if it is a string
    if isinstance(args.gpus, str):
        if "[" in args.gpus:
            args.gpus = ast.literal_eval(args.gpus)
        else:
            args.gpus = [int(g) for g in args.gpus]
        args.devices = len(args.gpus)
    elif isinstance(args.gpus, int):
        args.gpus = [args.gpus]
        args.devices = 1
    elif isinstance(args.gpus, list):
        args.devices = len(args.gpus)

    # sync batch size with devices
    if args.devices > 1:
        args.sync_batchnorm = True
        args.auto_select_gpus = False
    else:
        args.auto_select_gpus = True

    return args

def configure_args(cfg_path, hparams):
    args = args_setup(cfg_path)
    # align other args with hparams args
    args = vars(args)
    args.update(hparams)
    args = argparse.Namespace(**args)

    return args
