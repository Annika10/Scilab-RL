from itertools import chain
import numpy as np
from gym.core import Wrapper
import gym.spaces as spaces
from gym.wrappers import TimeLimit
import rlbench.gym  # unused, but do not remove. It registers the RL Bench environments.
from rlbench.backend.conditions import DetectedCondition, NothingGrasped
from pyrep.objects.shape import Shape
from pyrep.const import PrimitiveShape


class RLBenchWrapper(Wrapper):
    """
    Wrapper for RL Bench tasks that provides the methods necessary to work with our framework.
    RL Bench tasks consist of "conditions". This wrapper extracts goals from these "conditions",
    so that we can use them for goal-conditioned reinforcement learning.
    """
    def __init__(self, env):
        env = TimeLimit(env, 1000)
        super().__init__(env)
        self._max_episode_steps = self.env._max_episode_steps
        self.env.reset()
        obs, _, _, _ = self.step(self.env.unwrapped.action_space.sample())
        self.obs = obs.copy()  # We cache the obs because RL Bench Environments do not have a _get_obs() method.
        obs_space = spaces.Dict(dict(
            desired_goal=spaces.Box(-np.inf, np.inf, shape=obs['achieved_goal'].shape, dtype='float32'),
            achieved_goal=spaces.Box(-np.inf, np.inf, shape=obs['achieved_goal'].shape, dtype='float32'),
            observation=spaces.Box(-np.inf, np.inf, shape=obs['observation'].shape, dtype='float32'),
        ))
        obs_space.spaces['desired_goal'] = self._guess_goal_space(obs_space.spaces['desired_goal'])
        self.observation_space = obs_space
        self.vis = {}

        cond = self.env.unwrapped.task._task._success_conditions[0]
        if isinstance(cond, DetectedCondition):
            bb = (cond._detector.get_bounding_box())
            # the bounding box is ordered like this: min_x, max_x, min_y, max_y, min_z, max_z
            # but we want to order it like this: min_x, min_y, min_z, max_x, max_y, max_z
            bb = np.array([bb[0], bb[2], bb[4], bb[1], bb[3], bb[5]])
            self.bb = bb
            # TODO check for negation

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        achieved_goal, desired_goal = self._get_goals()
        obs = {
            'observation': obs,
            'achieved_goal': achieved_goal,
            'desired_goal': desired_goal
        }
        self.obs = obs.copy()
        info['is_success'] = int(reward)
        reward = reward - 1
        return obs, reward, done, info

    def reset(self, **kwargs):
        # max_episode_steps is updated after __init__(), so we apply the change to the TimeLimit env here
        self.env._max_episode_steps = self._max_episode_steps
        obs = self.env.reset()
        achieved_goal, desired_goal = self._get_goals()
        obs = {
            'observation': obs,
            'achieved_goal': achieved_goal,
            'desired_goal': desired_goal
        }
        self.obs = obs.copy()
        return obs

    def _get_obs(self):
        return self.obs

    def _sample_goal(self):
        self.env.unwrapped.reset()
        _, goal = self._get_goals()
        return goal

    def _guess_goal_space(self, goal_space):  # TODO proposition: always set the goal space = space above the table
        n_samples = 10# FIXME 00
        goal_space.high = -goal_space.high
        goal_space.low = -goal_space.low
        for _ in range(n_samples):
            goal = self._sample_goal()
            goal_space.high = np.maximum(goal, goal_space.high)
            goal_space.low = np.minimum(goal, goal_space.low)
        # Add some small extra margin.
        goal_space.high += np.abs(goal_space.high - goal_space.low) * 0.01
        goal_space.low -= np.abs(goal_space.high - goal_space.low) * 0.01
        return goal_space

    def _get_goals(self):
        achieved_goal = []
        desired_goal = []
        for cond in self.env.unwrapped.task._task._success_conditions:
            if isinstance(cond, DetectedCondition):
                desired_goal.append(cond._detector.get_position())
                achieved_goal.append(cond._obj.get_position())
            elif isinstance(cond, NothingGrasped):
                pass  # ignore NothingGrasped
            else:
                raise NotImplementedError("Converting this condition-type to a goal is not supported yet.")
        achieved_goal = np.array(list(chain.from_iterable(achieved_goal)))
        desired_goal = np.array(list(chain.from_iterable(desired_goal)))
        self.goal = desired_goal
        return achieved_goal, desired_goal

    def render(self, **kwargs):
        pass  # CoppeliaSim environments are not explicitly rendered

    def compute_reward(self, achieved_goal, desired_goal, info):
        # The reward/success computation in an RLBench environment uses different types of conditions to determine the
        # sparse rewards. To compute the reward manually, we would have to set the objects in the scene and then query
        # whether the conditioned was met. This would take too much time.
        # A workaround for DetectedConditions is to use the bounding box of the detector.
        if len(achieved_goal.shape) == 1:
            achieved_goal = [achieved_goal]
            desired_goal = [desired_goal]
        rewards = []
        for ag, dg in zip(achieved_goal, desired_goal):
            distance = ag - dg
            if np.all(distance >= self.bb[:3]) and np.all(distance <= self.bb[3:]):
                rewards.append(0.0)
            else:
                rewards.append(-1.0)
        return np.array(rewards)

    def _is_success(self, achieved_goal, desired_goal):
        return self.compute_reward(achieved_goal, desired_goal, []) == 0

    def display_subgoals(self, subgoals):
        # receives the subgoals in a dict of the form
        # 'subgoal_SHAPEX' : (position_array[x, y, z], float size, rgba[r, g, b, a])
        # where SHAPE is the visualization-shape and X is the number of the subgoal.
        to_visualize = {}
        for k in subgoals:
            to_visualize[k] = {'pos': subgoals[k][0], 'col': subgoals[k][2][:3], 'shape': k[8:-1]}
        self.visualize(to_visualize)

    def visualize(self, names_pos_col={}):
        """
        Takes a dictionary with names, positions, shapes and colors that is structured as follows:
        {'name': {'pos': [0.8, -0.1, 1.1], 'col': [.0, .9, .0], 'shape': 'sphere'},
        'name2': {'pos': [1.0, 0.1, 1.3], 'col': [.0, .0, .9], 'shape': 'box'}}
        Then PrimitiveShapes with the name are created in the specified shape and color and moved to the position.
        """
        for name in names_pos_col:
            if names_pos_col[name]['shape'] == 'box':
                shape = PrimitiveShape.CUBOID
            elif names_pos_col[name]['shape'] == 'sphere':
                shape = PrimitiveShape.SPHERE
            elif names_pos_col[name]['shape'] == 'cylinder':
                shape = PrimitiveShape.CYLINDER
            if name not in self.vis:
                self.vis[name] = Shape.create(shape, [0.04]*3, mass=0, respondable=False, static=True,
                                              position=names_pos_col[name]['pos'], color=names_pos_col[name]['col'])
            else:
                self.vis[name].set_position(names_pos_col[name]['pos'])
                self.vis[name].set_color(names_pos_col[name]['col'])

    @property
    def unwrapped(self):
        # This breaks the unwrapped property because it does not return the pure env.
        # We do this here because the RL Bench environments do not provide methods like _get_obs().
        return self
