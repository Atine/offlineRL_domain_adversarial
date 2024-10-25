# Offline Deep Reinforcement Learning for Visual Distractions via Domain Adversarial Training

<center>
Jen-Yen Chang<sup><sup>1</sup></sup>, Thomas Westfechtel <sup><sup>1</sup></sup>, Takayuki Osa <sup><sup>1,2</sup></sup>, Tatsuya Harada <sup><sup>1,2</sup></sup>
<br><br>
<sup><sup>1</sup></sup>The University of Tokyo &nbsp;&nbsp;<sup><sup>2</sup></sup>RIKEN AIP
</center>

<br>

#### Updates
- 2024.10.18 : Paper accepted by TMLR 2024
- 2024.10.22 : Code release


---


## Dataset

The data for V-D4RL can be found at their github: [V-D4RL](https://github.com/conglu1997/v-d4rl).    Our dataset can be found in at this [google drive](https://drive.google.com/drive/folders/1J58uGFI2qxTrEJ9LZv402iUOUtA3a3lN?usp=sharing).

Additionally, DAVIS17 dataset are needed to generate distracting backgrounds (videos) at [DAVIS2017](https://davischallenge.org/davis2017/code.html).   
Make sure to select the `2017 TrainVal - Images and Annotations (480p)`. 
For convenience, we also provide DAVIS17 dataset in our dataset directory above.     

It is recommended to copy the datasets to `~/.v_d4rl/`  as default location.    
The default dataset structure would look like this: 
```
~/.v_d4rl/
   └───vd4rl/
       └───main/
           └───walker_walk/
           └───cheetah_run/
           └───humanoid_walk/
           └───   ...
       └───distracting/
            ...
   └───vd4rl_extended/
       └───main/
           └───walker_walk/
           └───cheetah_run/
           └───humanoid_walk/
           └───   ...
       └───distracting

```

Alternatively, it is possible to change the dataset paths using either options below:

1. change in `train.py` and `train_dis.py` directly (as currently implemented)
2. adapt `cfg.offline_dir` and `cfg.offline_dir_dis` in cfg/config.yaml




## Install packages & setup
The helper scripts below expect to use cuda12+ (default pytorch cuda version), although cuda11 should work too with a bit of change to the pytorch install package inside `requirements.txt`.

virtualenv is used here for managing conflits but if you prefer conda you can play with the install scripts.


1. `source install.sh`
2. `source setup.sh`


## Usage

For baseline methods:
```
python train.py task=cheetah_run_expert_easy algo=drqv2 gpu=0 hydra.run.dir=/tmp/temp0 no_track=True
```

For proposed method:
```
python train_dis.py task=cheetah_run_expert_easy algo=drqv2_tog dropblock=0.3 gpu=0 hydra.run.dir=/tmp/temp0 no_track=True
```


#### algos
Baseline methods implemented, except DV2 (Dreamerv2), can be found in `cfgs/algos/`. Code for DV2 will come soon.


#### tasks
All the tasks that can be supplied to command line option `task={}` are listed in `cfgs/tasks/`


#### additional options:

- `no_track` disables wandb tracking
- `hydra.run.dir` will not create log directory in the current folder
- `cfg.offline_dir` and `cfg.offline_dir_dis` to change the directories to load dataset from.
- to train using our proposed method on two distracting dataset:
```
python train_two_dis.py task=cheetah_run_expert_easy algo=drqv2_tog dropblock=0.3 gpu=0 hydra.run.dir=/tmp/temp0 no_track=True
```
- to train using baseline method on one distracting dataset and one normal dataset:
```
python train_both.py task=cheetah_run_expert_easy algo=drqv2 gpu=0 hydra.run.dir=/tmp/temp0 no_track=True
```

## Citation
```
@inproceedings{polite-platinum-24,
author    = {Jen-Yen Chang and Thomas Westfechtel and Takayuki Osa and Tatsuya Harada},
title     = {Offline Deep Reinforcement Learning for Visual Distractions via Domain Adversarial Training},
booktitle = {Transactions on Machine Learning Research (TMLR)},
year      = {2024},
}
```
