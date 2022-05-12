import torch.nn as nn
import numpy as np
import torch
import os
import random

def fix_all_seeds(seed):
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# activation function parser
def configure_activation(activation):
    activation = activation.lower()
    if activation == "relu":
        activation = nn.ReLU()
    elif activation == "leaky_relu":
        activation = nn.LeakyReLU()
    elif activation == "tanh":
        activation = nn.Tanh()
    elif activation == "sigmoid":
        activation = nn.Sigmoid()
    elif activation == "none":
        activation = nn.Identity()
    return activation


