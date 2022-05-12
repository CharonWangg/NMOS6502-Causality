# Could a Neural Network Understand Microprocessor?
A Machine Learning Project Discovering Causality in the NMOS6502.  

<p align="center">
    <img width="1000" alt="causal_effect" src="https://github.com/CharonWangg/NMOS6502-Causality/blob/main/pics/causal_effect.png#gh-dark-mode-only">
</p>

## Table Of Contents
-  [Introduction](#introduction)
-  [Requirements](#requirements)
-  [Preparation](#preparation)
-  [Codebase](#codebase)
-  [Usage](#usage)
-  [Partial Result](#partial-result)
<!-- -  [Future Work](#future-work) -->
<!-- -  [Acknowledgement](#acknowledgement) -->

## Introduction  
Causal inference (CI) from time-varying data is an important issue in neuroscience, medicine, and machine learning. Techniques used for CI include regression, matching, and instrumental variables which are all based on human conceptual models. However, as we found in other areas of machine learning, human models may be worse than machine learning. Here we thus take a system with a large number of causal effects, the NMOS6502 microprocessor and meta-learn the causal inference procedure. By conducting single element perturbation, we acquire the causality relationship between every transistor and others, which lays a ground truth foundation for validation of causal inference algorithm. Here we introduce acquired supervised signal to generate a new causal inference algorithm. Experiments show that this procedure has robust generalization and far outperforms human designed inference procedures. We argue that causal inference should move towards a supervised mode, where causal inference procedures are chosen for their performance on large datasets where causal relations are known on top of the time-varying data. This promises a new approach towards the analysis of neural and medical data. 

## Requirements
- [Cython](https://cython.org/)
- [Pytorch-lightning](https://pytorch-lightning.readthedocs.io/en/latest/)

## Preparation
### For NMOS6502 Simulation (Modified from and inspired by [Sim2600](https://github.com/ericmjonas/Sim2600)):  
* Create a Python 2.7 env or adopt the existed venv.
* Install essential softwares and setup:
    ```
    bash install_deps.sh
    python setup.py
    ```
### For NMOS 6502 Inference
* Complete NMOS6502 simulation to acquire time series data and causal relationship of transistors. 
* Install requirements:
    ```
    pip install -r requirements.txt
    ```

## Codebase
```
├── README.md
├── nmos_inference  # Infer causal relationship by different algorithms
│   ├── pipeline  # Deep learning based training&inference pipeline
│   │   ├── configs  # Default pipeline config
│   │   ├── data_preprocess.ipynb  # Preprocess transistor state sequence like Screening, Stablization (has been integrated in make_data.py)
│   │   ├── debug.ipynb
│   │   ├── make_data.py  # Register preprocessed data and categorical label into dataframe, balance samples and split train/valid set
│   │   ├── methods.ipynb # Make plots for different experiments
│   │   ├── sanity_check.py # Sanity check on the inference result
│   │   ├── src  # Pipeline source code based on Pytorch Lightning
│   │   │   ├── data # Load NMOS6502 data into pipeline
│   │   │   ├── model # Load different models (LSTM, TCN, Transformers, FCN) into pipeline
│   │   │   └── utils # Utils for config, data, model
│   │   ├── train.py
│   │   ├── train.sh  # General training
│   │   ├── train_by_cmd.py  # Core script for running .sh script
│   │   ├── train_ds.sh  # Training under different down sample rate
│   │   ├── train_noise.sh  # Training under noise of different standard deviation
│   │   ├── train_ds_noise.sh  # Training under different combo of sample rate and noise
│   │   ├── train_ds_noise.sh  # Training under different combo of sample rate and noise
│   │   └── visualization.ipynb  # Visualization on the time series data and some circuts information
│   ├── plot_utils 
│   └── tests  # Test scripts of traditional methods (Pearson Correlation, Decorrelation, Mutual Information, Granger Causaltiy)
│   │   ├── all_tests.py  # Run all tests
│   │   ├── correlation.py  # Caculate accuracy based on the abs of Pearson correlation score
│   │   ├── decorrelation.py  # Caculate accuracy based on the sigmoided first eigenvector of covariance matrix
│   │   ├── mutual_info.py  # Caculate accuracy based on the sigmoided mutual information score
│   │   ├── granger_gc_test  # Caculate accuracy based on the Granger causality score based on ssr based F test after Bonferroni correction

├── nmos_simulation  # Generate regular and perturbed transistor-level simulation data
│   ├── build  
│   ├── install_deps.sh  # Setup 
│   ├── open_dos.bat
│   ├── setup.py  # Setup
│   ├── sim2600  # Core simulation for NMOS6502
│   ├── tests
│   │   ├── EDA.ipynb  # Some visualization and sanity check experiments on time series data
│   │   ├── main.py  # Generate regular simulation data or high/low voltage intervention data
│   │   ├── make_effect_label.py  # Convert raw perturbation data to pairwise voltage difference, and generate categorical label
│   │   ├── test_compare_sims.py  # Simulation interface called by main.py
│   └── venv

```


## Usage
### NMOS6502 Simulation
```
python main.py \  # Acquire High voltage intervention on each transistor and save result as (num_transistor, num_transistor, num_iter)
  --game All \
  --num_iter 512 \
  --action High
python main.py \  # Acquire Low voltage intervention on each transistor and save result as (num_transistor, num_transistor, num_iter)
  --game All \
  --num_iter 512 \
  --action Low
python main.py \  # Acquire Regular running state of each transistor and save result as (num_transistor, num_iter)
  --game All \
  --num_iter 512 \
  --action Regular
python make_effect_label.py \  # Get cause effect and raw categorical label
  -game All                    # (1: transistor A has cause effect on B; 0: transistor A doesn't have cause effect on B)
```
### NMOS6502 Inference
* Run make_dat.py to get structural dataframe and preprocessed data  
* Train different models:
```
bash train.sh  # Regular training without conditions
bash train_ds.sh  # Training under different sample rate (0.5x, 0.25x, 0.125x)
bash train_noise.sh  # Training under noise of different intensity (mean: 0; std: 0.1, 0.3, 0.5)
bash train_ds_noise.sh # Training under different noise and sample rate combinations
```
* Run different traditional tests:
```
python all_tests.py
```

## Partial Results
* Supervised procedure trained on Donkey Kong shows stable transferability on Pitfall and Space Invaders without retraining or finetuning, which still outperforms traditional methods. 
<p align="center">
    <img width="600"  alt="transfer_learning" src="https://github.com/CharonWangg/NMOS6502-Causality/blob/main/pics/transfer_learning.png#gh-dark-mode-only">
</p>

* Here we adopt different combinations of noise and down sample rate to simulate more and more severe reality condition. Compared to traditional methods, doing causal inference in a supervised mode can help algorithm keep model robustness and inference stability.
<p align="center">
     <img width="500" alt="double_aug" src="https://github.com/CharonWangg/NMOS6502-Causality/blob/main/pics/double_aug_chart.png#gh-dark-mode-only">
</p>
<!-- 
## Future Work
Any kind of enhancement or contribution is welcomed. -->

<!-- ## Acknowledgement -->


