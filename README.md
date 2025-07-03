# Giving AI agents a sense of control facilitates reinforcement learning in multitasking scenarios

This repo belongs to the paper *Giving AI agents a sense of control facilitates reinforcement learning in multitasking
scenarios*.
It is a fork of the [Scilab-RL](https://github.com/Scilab-RL/Scilab-RL) repository focusing on goal-conditioned
reinforcement learning using
the [stable baselines 3](https://stable-baselines3.readthedocs.io/en/master/) methods
and [Gymnasium](https://gymnasium.farama.org/) interface.
You can find their wiki here: [Scilab-RL wiki](https://scilab-rl.github.io/Scilab-RL/wiki/).

With this repository you can train and evaluate sub- and meta-agents in a multitasking scenario playing two *Collect
Asteroids* games.
For more information about the setup, please refer to the paper.
The statistical analysis of the agents can be found in a [jupyter notebook](https://osf.io/ctn79) uploaded on OSF.

## Installation

Please refer to
the [installation script](https://github.com/Scilab-RL/Scilab-RL?tab=readme-ov-file#getting-started-using-the-setup-script)
of the original repository.
Additionally, MISSING

## Relevant files

- [conf/algorithms/ppo_moonlander.yaml](conf/algorithm/ppo_moonlander.yaml): The configuration file for the PPO
  algorithm.
- [conf/performance/Moonlander](conf/performance/Moonlander): The folder containing the hyperparameter tuning
  configuration files.
- [conf/main.yaml](conf/main.yaml): The main configuration file for the training and evaluation process.


- [src/main.py](src/main.py): The main script to run the training and evaluation process.
- [src/custom_algorithms/ppo_moonlander](src/custom_algorithms/ppo_moonlander): The folder containing the PPO algorithm
  implementation.
- [src/custom_envs/moonlander](src/custom_envs/moonlander): The folder containing the environments used for training and
  evaluation.
    - [src/custom_envs/moonlander/heuristic_baseline_evaluation_new.py](src/custom_envs/moonlander/heuristic_baseline_evaluation_new.py):
      The script to evaluate the switch every frame baseline.
    - [src/custom_envs/moonlander/moonlander_env.py](src/custom_envs/moonlander/moonlander_env.py): The main environment
      for the *Collect Asteroids* game.
        - [src/custom_envs/moonlander/positions_wrapper.py](src/custom_envs/moonlander/positions_wrapper.py): The
          wrapper to use position observations instead of image observations.
    - [src/custom_envs/meta_env_pretrained_without_soc.py](src/custom_envs/moonlander/meta_env_pretrained_without_soc.py):
      The script to create the meta-environment for training and evaluating meta-agents.
        - [src/custom_envs/meta_env_pretrained_with_soc_wrapper.py](src/custom_envs/moonlander/meta_env_pretrained_with_soc_wrapper.py):
          The wrapper to use the Sense of Control (SoC) for training and evaluating meta-agents.
        - [src/custom_envs/meta_env_pretrained_with_soc_observation_only_wrapper.py](src/custom_envs/moonlander/meta_env_pretrained_with_soc_observation_only_wrapper.py):
          The wrapper to use the SoC only as the observation for training and evaluating meta-agents.
        - [src/custom_envs/meta_env_pretrained_with_soc_reward_only_wrapper.py](src/custom_envs/moonlander/meta_env_pretrained_with_soc_reward_only_wrapper.py):
          The wrapper to use the SoC only as the reward for training and evaluating meta-agents.
    - [src/custom_envs/register_envs.py](src/custom_envs/register_envs.py): The script to register the custom
      environments.
- [trained_agents](trained_agents): The folder containing the trained agents.
    - [trained_agents/subagents](trained_agents/subagents): The folder containing the pretrained sub-agents.
    - [trained_agents/meta-agents](trained_agents/meta-agents): The folder containing the pretrained meta-agents.

## Usage

### Training of Subagents

Each subagent is trained in one instance of a *Collect Asteroids* game.
With the following command you can train the sub-agents in the easy level configuration without input noise.

```bash
python src/main.py env=MoonlanderWorld-collect-gaussian_with_distance-easy-v0 algorithm=ppo_moonlander wandb=1 positions=1 +performance=Moonlander/collect-ppo-moonlander-opti.yaml --multirun
```

In the following, we explain the different parameters of the command:

- `env=MoonlanderWorld-collect-gaussian_with_distance-easy-v0`: The environment to train in. The environment is a
  *Collect Asteroids* game with Gaussian and distance-based rewards and in easy level difficulty.
- `algorithm=ppo_moonlander`: The algorithm to use for training. In this case, we use the Proximal Policy
  Optimization (PPO) algorithm.
- wandb=1: Whether to use Weights and Biases (WandB) for logging the training process. Set to 0 to disable WandB.
- `positions=1`: Indicates if the subagent is trained with positions or with image observation.
  In this case, we train the sub-agent with a position observation.
  For training with image observations, change `positions=1` to `positions=0`.
- `+performance=Moonlander/collect-ppo-moonlander-opti.yaml`: The configuration file for the performance
  evaluation of the trained agent. This file contains the hyperparameters for the training process.
- `--multirun`: This flag indicates that multiple agents are trained at the same time.

If you want to train the subagent with input noise, you can change the environment
to `MoonlanderWorld-collect-gaussian_with_distance-easy-input-noise-v0`.
Training for the hard level configuration can be done by changing `easy` to `hard` in the environment name.

### Training of Meta-agents

The meta-agents are trained in a multitasking scenario, where they have to play two *Collect Asteroids* games.
The meta-agents use already pretrained sub-agents.
We added the pretrained sub-agents to the [repository](trained_agents/subagents), so you can use them directly.
We also added the meta-agents used in the paper to the [repository](trained_agents/meta-agents).
To train the meta-agents, you can use the following command:

```bash
python src/main.py env=MetaEnv-pretrained-without-SoC-easy-easy-v0 algorithm=ppo_moonlander wandb=1 soc=1 n_epochs=5 +performance=Moonlander/meta-ppo-moonlander-opti.yaml --multirun
```

In the following, we explain the different parameters of the command:

- `env=MetaEnv-pretrained-without-SoC-easy-easy-v0`: The environment to train in. The environment is a multitasking
  scenario with two *Collect Asteroids* games in easy level difficulty.
  For changing the difficulty or applying input noise, change the name of the environment.
  You can find all available environments in [register_envs.py](src/custom_envs/register_envs.py).
- `algorithm=ppo_moonlander`: The algorithm to use for training. In this case, we use the Proximal Policy
  Optimization (PPO) algorithm.
- wandb=1: Whether to use Weights and Biases (WandB) for logging the training process. Set to 0 to disable WandB.
- `soc=1`: Whether to use the Sense of Control (SoC) for training the meta-agent. Set to 0 to disable SoC.
- `n_epochs=5`: The number of epochs to train the meta-agent. You can change this value to train the meta-agent
  for more or fewer epochs. You need a higher number of epochs for training without SoC.
- `+performance=Moonlander/meta-ppo-moonlander-opti.yaml`: The configuration file for the performance
  evaluation of the trained agent. This file contains the hyperparameters for the training process.
- `--multirun`: This flag indicates that multiple agents are trained at the same time.

### Evaluation of Subagents

To evaluate the sub-agents, you can use the following command:

```bash
python src/main.py env=MoonlanderWorld-collect-gaussian_with_distance-easy-benchmark-v0 algorithm=ppo_moonlander wandb=0 render=none n_epochs=1 eval_after_n_steps=1 positions=1 +restore_policy=/absolute/path/to/your/repo/Scilab-RL/trained_agents/subagents/collect_gaussian_with_distance_easy_positions_07_04_2025_best_model
```

In the following, we explain the different parameters of the command:

- `env=MoonlanderWorld-collect-gaussian_with_distance-easy-benchmark-v0`: The environment to evaluate the sub-agent in.
  The benchmark ensures that all subagents trained with an easy level difficulty are evaluated in the same environments.
  The environment is a *Collect Asteroids* game with Gaussian and distance-based rewards and in easy level difficulty.
- `algorithm=ppo_moonlander`: The algorithm to use for evaluation. In this case, we use the Proximal Policy
  Optimization (PPO) algorithm.
- `wandb=0`: Whether to use Weights and Biases (WandB) for logging the evaluation process. Set to 1 to enable WandB.
- `render=none`: The rendering mode for the evaluation. Set to `none` to disable rendering.
  You can also set it to `display` to render the environment in a window or `record` to record the evaluation as a
  video.
- `n_epochs=1`: The number of epochs to evaluate the sub-agent.
- `eval_after_n_steps=1`: The number of steps after which to evaluate the sub-agent.
- `positions=1`: Indicates if the subagent is evaluated with positions or with image observation.
  In this case, we evaluate the sub-agent with a position observation.
  For evaluation with image observations, change `positions=1` to `positions=0`.
- `+restore_policy=/absolute/path/to/your/repo/Scilab-RL/policies/collect_gaussian_with_distance_easy_positions_07_04_2025_best_model`:
  The path to the
  pretrained sub-agent policy. You can find the pretrained sub-agents in the [repository](trained_agents/subagents).
  Make sure to change the path to the absolute path of your repository.

It will output the evaluation results to the console.

### Evaluation of Meta-agents

To evaluate the meta-agents, you can use the following command:

```bash
python src/main.py env=MetaEnv-pretrained-without-SoC-easy-hard-input-noise-in-hard-task-v0 algorithm=ppo_moonlander wandb=0 render=none n_epochs=1 eval_after_n_steps=1 n_test_rollouts=100 +env_kwargs.list_of_object_dict_lists_collect_task_one_filename=collect_easy_object_list_30_times_40_0.csv +env_kwargs.list_of_object_dict_lists_collect_task_two_filename=collect_hard_object_list_30_times_40_1.csv +restore_policy=/absolute/path/to/your/repo/Scilab-RL/trained_agents/meta-agents/meta_easy_hard_yes_in_hard_task_without_soc_20_05_2025_best_model.zip
```

In the following, we explain the different parameters of the command if different than the evaluation of sub-agents:

- `env=MetaEnv-pretrained-without-SoC-easy-hard-input-noise-in-hard-task-v0`: The environment to evaluate the
  meta-agent in. The environment is a multitasking scenario with two *Collect Asteroids* games in easy and hard level
  difficulty.
  For changing the difficulty or applying input noise, change the name of the environment.
  You can find all available environments in [register_envs.py](src/custom_envs/register_envs.py).
- `n_test_rollouts=100`: The number of test rollouts to evaluate the meta-agent.
- `+env_kwargs.list_of_object_dict_lists_collect_task_one_filename=collect_easy_object_list_30_times_40_0.csv`:
  The path to the object list for the first *Collect Asteroids* game. This file contains the objects that are used in
  the
  first game. With this file we can ensure that all meta-agents are evaluated in the exact same 100 test rollouts.
- `+env_kwargs.list_of_object_dict_lists_collect_task_two_filename=collect_hard_object_list_30_times_40_1.csv`:
  The path to the object list for the second *Collect Asteroids* game. This file contains the objects that are used in
  the
  second game. With this file we can ensure that all meta-agents are evaluated in the exact same 100 test rollouts.
- `+restore_policy=/home/annika/coding_projects/Scilab-RL-github/Scilab-RL/policies/meta_easy_hard_yes_in_hard_task_without_soc_20_05_2025_best_model.zip`:
  The path to the pretrained meta-agent policy. You can find the pretrained meta-agents in
  the [repository](trained_agents/meta-agents).
  Make sure to change the path to the absolute path of your repository.

It will output the evaluation results to the console and save the results in the [logs](src/custom_envs/logs) folder.