<div align="center">

# Object Detection Framework
<a href="https://www.python.org/"><img alt="Python" src="https://img.shields.io/badge/-Python 3.7+-blue?style=for-the-badge&logo=python&logoColor=white"></a>
<a href="https://pytorchlightning.ai/"><img alt="Lightning" src="https://img.shields.io/badge/-Lightning 1.5+-792ee5?style=for-the-badge&logo=pytorchlightning&logoColor=white"></a>
<a href="https://hydra.cc/"><img alt="Config: hydra" src="https://img.shields.io/badge/config-hydra 1.1-89b8cd?style=for-the-badge&labelColor=gray"></a>
<a href="https://black.readthedocs.io/en/stable/"><img alt="Code style: black" src="https://img.shields.io/badge/code%20style-black-black.svg?style=for-the-badge&labelColor=gray"></a>
</div>

## Introduction 
This object detection framework was created for dental caries detection from X-ray images. Despite its purpose, it can be modified to serve as a general object detection framework. <br />
It was inspired by the IceVision (https://github.com/airctic/icevision) library, and part of their code was inherited. This framework supports self-implemented detection models as well as models from multiple external libraries such as MMDet or Ultralytics. We provide functions to convert data to the format required by those libraries. 


## How to install
This project relies on many external dependencies. The main dependencies are as follows: Pytorch-Lightning, Hydra, Optuna. You can install this project by following commands (conda required).

```{bash}
- conda env create -f environment.yaml
- conda activate detection
```
The other option is to use poetry. You simply do the following
```{bash}
poetry install
```

## How to run
- You can train a model with the default setting by `python train.py`. 
- To test a model specify the path to model weights either by command line or in module.yaml file and run `python test.py` this will run tests on the test dataset and report the metrics defined by the user (by default mean average precision)
- Model inference is defined for every module. We usually convert those into a .json file (this format is required to do the ensembling). The conversion and prediction can be done either by `scripts/make_and_save_preds.py`  or by one of the jupyter notebooks `predict.ipynb` or `predict_batches.ipynb`.
- Model evaulation is done by class in 'evaluation/prediction_evaluation.py'. We support evaulatuion for predictions in .json files or in COCO format. An example how to evaluate predictions is in `predict.ipynb` notebook.
- Ensembling is done by one of the following methods: Weighted Boxes Fusion, Non-Maximal Weighted, Non-Maximal suppression. First convert predictions of all models to .json files. Ensembling is handles by a class defined in `evaluation/ensembling.py`. For example of use see `ensemble.ipynb` file in `notebooks` folder.

## Project structure

```{bash}
MT
├── configs                       <- Folder with all Hydra configuration files
│   ├── train.yaml                <- Main project configuration file for training. Other config files are merged by this one
│   ├── test.yaml                 <- Main project configuration file for testing. Other config files are merged by this one
│   ├── callbacks                 <- Callbacks config, choose which callbacks you want to include and their settings
│   ├── datamodule                <- Datamodule related config (path, batch-size,...)
│   ├── experiment                <- Experiment config
│   ├── logger                    <- Logger configs
│   ├── module                    <- Configure models (learning rate, model related parameters, ...)
│   ├── transforms                <- List of all transformations to augment images
│   └── trainer                   <- Pytorch-Lightning trainer configuration
├── notebooks                     <- Jupyter notebooks (preprocessing and visualization)
├── scrips                        <- Python scripts for parameter search or other auxilary tasks
├── optimize_optuna.py            <- Optuna multi-process optimization
├── optuna_single_process.py      <- Optuna single-process optimization
├── README.md   
├── train.py                      <- Run current configuration specified by Hydra configs (train.yaml)
├── train.py                      <- Run current configuration specified by Hydra configs (test.yaml)
├── src                          
│   ├── core                      <- Classes for the composite design
│   ├── data                      <- Unified dataset, data parsers (from preprocessed format), splitters
│   ├── evaluation                <- Prediction evaluation and post-processing (ensembling)
│   ├── metrics                   <- MAP metrics based on PyCOCOtools
│   ├── datamodules               <- Pytorch lightning based datamodules and datasets
│   ├── modules                   <- Pytorch lightning based modules - training loop implementation.
│   ├── models                    <- Models and backbones are deffined here. Most of them are wrappers for extenal libraries
│   ├── transforms                <- Transforms composer to create transformation given by configuration yaml file 
│   ├── train.py                  <- Here all source files are combined, based on the configuration
│   └── utils                     <- Data conversion, loss functions, auxilary functions
└── tests                         <- Pytests for parts of the project (not everything is covered)

```

## How to change the program settings
This project relies on Hydra to manage the configuration. The configuration is divided into separate files for each part of the program, e.g., configuration for the model, datamodule, etc. For more details, read the project structure section and configurations structure.
Most of the settings can be changed from the command line. You can override any configuration by appending `<settings to change>=<reqiured settings>` to the `python train command`. For example, `python train.py module=yolov5`. You can also add configuration, that is not specified in .yaml files by `+<setting to change>=<required settings>`, eg. `+trainer.fast_dev_run=True`. You have to look into the documentation of the particular method to get a list of all parameters, or you can install Hydra's autocompletion by running the following command: `python train.py --hydra-help` and then following the manual, that will pop up. 

## Configurations structure
The configuration is governed by test.yaml or train.yaml. The test.yaml file is for inference, while the train.yaml is for the training of the model. For a list of all things that can be configured, check the project structure. The most important modules to be configured are the following:
#### Eperiments
Allow you to override default settings. The common use is to switch to semantic segmentation tasks. For this, you need to override multiple configurations setting at once, which is done by the corresponding experiment.yaml file.
#### Modules
Here you configure parameters of the model such as learning rate or weight decay.
#### Datamodules
Configuration of datamodules is done here. It consists of the path to the data, batch size, etc.
#### Transforms
This is used to select image augmentations. Currently, we support Albumentations only, but it is possible to use a different library, but you need to provide a transformation composer.
#### Trainer
Parameters of PyTorch-Lightning trainer are set here. The settings include the number of GPUs used, the maximum number of epochs, or the number of gradient batches to accumulate.
#### Callbacks
Here you define callbacks that will be passed to PyTorch-Lightning. You can PyTorch-Lightning callbacks or instantiate your own callbacks.

## Hyper-parameter search
This framework supports hyper-parameter search powered by Optuna. Optuna is an optimization toolbox that uses methods such as Tree-structured Parzen Estimator to propose hyper-parameters to use in the next trial. The history of runs is kept in SQL database. You can optimize multiple computers, even multiple nodes, if you specify the path to your SQL storage. 
There are multiple options for how to use optuna by this framework.
 ```{bash}
 pyhton run.py -m hprarams_search=<config from hparams_search folder>
 ```
 This is the fastest possible approach to setup, but Optuna will have limited capabilities. There will be no pruning available, and search-space configuration is limited. The search space is defined in the .yaml config file. 
 <br>
 ```{bash}
 python optimize_optuna.py
 ```
 Will launch multi-process Optuna search. You need to modify the content of this file (specify search-space and config overrides). This is a discouraged approach since there are situations when single or more processes freeze. I am working on fixing this issue.
 
 <br>
 
 ```{bash}
 python optuna_single_process.py
 ```
 In this setting, optuna launches a single optimization process. You can run this file multiple times to get faster optimization results. The scaling of the number of computers should have a near-linear impact on search time. If you run this on multiple nodes, you need to provide a database that is accessible to all nodes.

### Merge dataset from CVAT
Merging of multiple datset from CVAT is supported. To merge dataset from CVAT: Export  all cvat tasks in the COCO format. Tasks are merged by `merge_dataset.ipynb` script, where you select path to tasks and target location

### FiftyOne visualization
The dataset can be visualized by VoxelFiftyone. This is done by `fiftyone.ipynb` jupyter notebook. Please note, that fiftyone is not dependency (since it is not compatible with our environment on CMP server). To load prediction prepare them in .json format and load them by function in the notebook.
