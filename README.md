
# Offline Deep Reinforcement Learning for Visual Distractions via Domain Adversarial Training


## Dataset
The data for V-D4RL can be found at their github: https://github.com/conglu1997/v-d4rl   
Our dataset can be found in:
The dataset paths is expected to be found at ~/.v_d4rl/vd4rl   
Alternatively, it is possible to change the dataset paths using either options below:
  1. in `train.py` and `train_dis.py` directly (as currently implemented)
  2. `cfg.offline_dir` and `cfg.offline_dir_dis` in cfg/config.yaml


## install
expect to have cuda12+ on the system, although cuda11 should work

## Usage

For baseline methods:   
```bash
python train.py task=cheetah_run_expert_easy algo=drqv2 gpu=0 hydra.run.dir=/tmp/temp0 no_track=True
```

For proposed method:   
```bash
python train_dis.py task=cheetah_run_expert_easy algo=drqv2_tog dropblock=0.3 gpu=0 hydra.run.dir=/tmp/temp0 no_track=True 
```

`no_track` disables wandb tracking   
`hydra.run.dir` will not create log directory in the current folder 



## Tasks
A list of tasks can be found in cfg/tasks
