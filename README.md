# Could a Neural Network Understand Microprocessor?
A Machine Learning Project Discovering Causality in the NMOS6502.

<p align="center">
    <img width="1000" alt="causal_effect" src="https://github.com/CharonWangg/NMOS6502-Causality/blob/main/pics/causal_effect.png#gh-dark-mode-only">
</p>



## Table Of Contents
-  [Introduction](#introduction)
-  [Requirements](#requirements)
-  [Preparation](#preparation)
-  [Codebase Structure](#codebase-structure)
-  [Usage](#usage)
-  [Future Work](#future-work)

## Introduction  

## Requirements
- [Cython](https://cython.org/)
- [Pytorch-lightning](https://pytorch-lightning.readthedocs.io/en/latest/)

## Preparation

Install PyTorch and download the ImageNet dataset following the [official PyTorch ImageNet training code](https://github.com/pytorch/examples/tree/master/imagenet). Similar to [MoCo](https://github.com/facebookresearch/moco), the code release contains minimal modifications for both unsupervised pre-training and linear classification to that code. 

In addition, install [apex](https://github.com/NVIDIA/apex) for the [LARS](https://github.com/NVIDIA/apex/blob/master/apex/parallel/LARC.py) implementation needed for linear classification.

## Codebase Structure
```
├── README.md
├── nmos_inference
│   ├── pipeline
│   │   ├── configs
│   │   ├── data_preprocess.ipynb
│   │   ├── debug.ipynb
│   │   ├── make_data.py
│   │   ├── methods.ipynb
│   │   ├── sanity_check.py
│   │   ├── src
│   │   │   ├── data
│   │   │   ├── model
│   │   │   └── utils
│   │   ├── train.py
│   │   ├── train.sh
│   │   ├── train_by_cmd.py
│   │   ├── train_ds.sh
│   │   ├── train_ds_noise.sh
│   │   ├── train_noise.sh
│   │   ├── validation.py
│   │   └── visualization.ipynb
│   ├── plot_utils
│   └── tests
├── nmos_simulation
│   ├── build
│   ├── install_deps.sh
│   ├── open_dos.bat
│   ├── requirements.txt
│   ├── setup.py
│   ├── sim2600
│   ├── tests
│   │   ├── EDA.ipynb
│   │   ├── main.py
│   │   ├── make_effect_label.py
│   │   ├── test_compare_sims.py
│   │   └── test_compare_sims.pyc
│   └── venv

```


## Usage
### NMOS6502 Simulation
```
python main_simsiam.py \
  -a resnet50 \
  --dist-url 'tcp://localhost:10001' --multiprocessing-distributed --world-size 1 --rank 0 \
  --fix-pred-lr \
  [your imagenet-folder with train and val folders]
```
### NMOS6502 Inference
```
python main_simsiam.py \
  -a resnet50 \
  --dist-url 'tcp://localhost:10001' --multiprocessing-distributed --world-size 1 --rank 0 \
  --fix-pred-lr \
  [your imagenet-folder with train and val folders]
```

# Future Work
Any kind of enhancement or contribution is welcomed.
