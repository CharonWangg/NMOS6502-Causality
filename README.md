# Could a Neural Network Understand Microprocessor?
A Machine Learning Project Discovering Causality in the NMOS6502.

<p align="center">
    <img width="1000" alt="causal_effect" src="https://github.com/CharonWangg/NMOS6502-Causality/blob/main/pics/cause_effect.png#gh-dark-mode-only">
</p>

# Requirements
- [Pytorch-lightning](https://github.com/rbgirshick/yacs) (Yet Another Configuration System)
- [](https://pytorch.org/) (An open source deep learning platform) 
- [ignite](https://github.com/pytorch/ignite) (High-level library to help with training neural networks in PyTorch)

# Table Of Contents
-  [In a Nutshell](#in-a-nutshell)
-  [In Details](#in-details)
-  [Future Work](#future-work)
-  [Contributing](#contributing)
-  [Acknowledgments](#acknowledgments)

# In a Nutshell   
In a nutshell here's how to use this template, so **for example** assume you want to implement ResNet-18 to train mnist, so you should do the following:
- In `modeling`  folder create a python file named whatever you like, here we named it `example_model.py` . In `modeling/__init__.py` file, you can build a function named `build_model` to call your model

```python
from .example_model import ResNet18

def build_model(cfg):
    model = ResNet18(cfg.MODEL.NUM_CLASSES)
    return model
``` 

   
- In `engine`  folder create a model trainer function and inference function. In trainer function, you need to write the logic of the training process, you can use some third-party library to decrease the repeated stuff.

```python
# trainer
def do_train(cfg, model, train_loader, val_loader, optimizer, scheduler, loss_fn):
 """
 implement the logic of epoch:
 -loop on the number of iterations in the config and call the train step
 -add any summaries you want using the summary
 """
pass

# inference
def inference(cfg, model, val_loader):
"""
implement the logic of the train step
- run the tensorflow session
- return any metrics you need to summarize
 """
pass
```

- In `tools`  folder, you create the `train.py` .  In this file, you need to get the instances of the following objects "Model",  "DataLoader”, “Optimizer”, and config
```python
# create instance of the model you want
model = build_model(cfg)

# create your data generator
train_loader = make_data_loader(cfg, is_train=True)
val_loader = make_data_loader(cfg, is_train=False)

# create your model optimizer
optimizer = make_optimizer(cfg, model)
```

- Pass the all these objects to the function `do_train` , and start your training
```python
# here you train your model
do_train(cfg, model, train_loader, val_loader, optimizer, None, F.cross_entropy)
```

**You will find a template file and a simple example in the model and trainer folder that shows you how to try your first model simply.**


# In Details
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


# Future Work

# Contributing
Any kind of enhancement or contribution is welcomed.


# Acknowledgments




### Preparation

Install PyTorch and download the ImageNet dataset following the [official PyTorch ImageNet training code](https://github.com/pytorch/examples/tree/master/imagenet). Similar to [MoCo](https://github.com/facebookresearch/moco), the code release contains minimal modifications for both unsupervised pre-training and linear classification to that code. 

In addition, install [apex](https://github.com/NVIDIA/apex) for the [LARS](https://github.com/NVIDIA/apex/blob/master/apex/parallel/LARC.py) implementation needed for linear classification.

### Unsupervised Pre-Training

Only **multi-gpu**, **DistributedDataParallel** training is supported; single-gpu or DataParallel training is not supported.

To do unsupervised pre-training of a ResNet-50 model on ImageNet in an 8-gpu machine, run:
```
python main_simsiam.py \
  -a resnet50 \
  --dist-url 'tcp://localhost:10001' --multiprocessing-distributed --world-size 1 --rank 0 \
  --fix-pred-lr \
  [your imagenet-folder with train and val folders]
```
The script uses all the default hyper-parameters as described in the paper, and uses the default augmentation recipe from [MoCo v2](https://arxiv.org/abs/2003.04297). 

The above command performs pre-training with a non-decaying predictor learning rate for 100 epochs, corresponding to the last row of Table 1 in the paper. 

### Linear Classification

With a pre-trained model, to train a supervised linear classifier on frozen features/weights in an 8-gpu machine, run:
```
python main_lincls.py \
  -a resnet50 \
  --dist-url 'tcp://localhost:10001' --multiprocessing-distributed --world-size 1 --rank 0 \
  --pretrained [your checkpoint path]/checkpoint_0099.pth.tar \
  --lars \
  [your imagenet-folder with train and val folders]
```

The above command uses LARS optimizer and a default batch size of 4096.

### Models and Logs


### Transferring to Object Detection


