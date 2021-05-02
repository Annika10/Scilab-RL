# ideas_deep_rl2

This is the IDEAS / LeCAREbot deep RL repository focusing on hierarchical goal-conditioned reinforcement learning using the stable baselines 3 methods and OpenAI gym interface

## Requirements:
- Python 3.6+

## Getting started

1. Download MuJoCo (mujoco.org) and obtain a license
   (as student you can obtain a free one-year student license).
   Copy the mjpro200_linux folder from the downloaded archive
   as well as mjkey.txt that you will obtain from the registration
   to folders of your choice.

1. Set the environment variables in `set_paths.sh` according to the
   locations where you saved the mjpro200_linux folder and the mjkey.txt.
   If you are using an IDE, set the variables there as well.

1. generate a virtual python3 environment with

    `virtualenv -p python3 venv`

1. load the environment with

    `source venv/bin/activate`

1. upgrade the version of pip
    `pip install --upgrade pip`

1. set environment variables and paths with

    `source ./set_paths.sh`

1. install required python libraries with

    `pip install -r requirements.txt`

1. Add the mujoco binary path to the  `LD_LIBRARY_PATH` environment variable. If you first run the script without doing this it will automatically remind you and identify the path for you.


## Start training manually (hydra debugging)

```bash
python3 experiment/train.py env=FetchReach-v1 algorithm=hac algorithm.render_test=record algorithm.time_scales=[5,-1]
python experiment/train.py env=FetchReach-v1 algorithm=hac layer_classes=['sacvg','ddpg']
```

```bash
python experiment/train.py env=FetchReach-v1 algorithm=her2 layer_classes=['sac']
# also works with ddpg
```


## File structure
* The main script from which all algorithms are started is `train.py`.
* The root directory contains shell scripts for automated testing and data generation.
* The folder `experiment` contains all architectural stuff for evaluation.

    * `plot.py` is for plotting

    * `testing_algos.py` is for defining testing parameters

    * `testing_envs.py` is for defining testing environments

    * `click_options.py` is for setting the global (model-independent) command-line parameters

    * `check_error_logs.py`, `generate_testing_commands.py` and `validate_performance_testing.py` is for executing evaluations and tests.

* The folder `ideas_baselines` contains the new HAC implementation and an implementation of HER. Other new algorithms should be added here, too. For details on the specific algorithms, see below.
* The folder `ideas_envs` should contain new environments (but we may also choose to put environments in a completely different repository).
* The folder `interface` contains for each algorithm, both stable-baselines3 algorithms and the algorithms here, a file `config.py` and `click_options.py`. The click options file determines the kwargs passed on to the model (HAC, SAC, TD3, etc). These are specifyable as command-line options. The file `config.py` is right now just for determining the parameters to be used for generating a path name for the log directory.
* The folder `util` contains some misc utilities.
* The folder `hydra_plugins` contains some customized plugins for our hyperparameter management system.

## Testing

1. generate a file with all possible environment/algorithm combinations with:

    `./run_testing.sh -g <comma-separated list of gpu_ids in paremtheses, e.g. '(0,1)'> -p <max processes> -m <min. free GPU memory required to star a new process> -s <time to sleep before executing next command> -t <type of testing <function|performance>>`

    The defaults for these parameters are g=0, p=6, m=2500, t='performance', s=15

    The script will create a file `test_cmds.txt` in the base directory with all currently supported environment-algorithm combinations and useful hyperparameters. It will also execute them and perform two checks:

    1. A function check to determine whether a process crashed
    1. A performance test to determine whether certain predefined success rates are met. These success rates are defined in `experiment/testing_algos.py` (see Section on Testing below.)

## Algorithms

### HAC
TBD

## Limitations
Currently, only off-policy algorithms are supported: DQN, DDPG, TD3 and SAC. PPO is not supported

## Hyperparameter optimization and management
The framework has a sophisticated hyperparameter optimization management. It builds on 
* hydra to manage the command-line parameters and configuration options. 
* mlflow to collect studies and log data of alle runs
* optuna as a framework to perform the hyperparameter optimization algorithm (e.g. TPE)
* comet.ml to visualize the data.


To start the hyperparameter optimization start `experiment/train.py --multirun`. The `--multirun` flag starts the hyperparameter optimization mode. 

To set up the search space for the hyperparameter optimization look at the `search_space` section of the `conf/main.yaml` file

To upload the results to comet.ml, execute `comet_for_mlflow --api-key $COMET_API_KEY --rest-api-key $COMET_REST_API_KEY` in the root directory of this repository.
