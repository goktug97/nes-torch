from typing import Callable, List, Optional
from dataclasses import field
from itertools import tee

import gym
from pipcs import Config
import torch
from torch import nn
import numpy as np

from nes import NES, Policy, default_config


def pairwise(iterable):
    "s -> (s0,s1), (s1,s2), (s2, s3), ..."
    a, b = tee(iterable)
    next(b, None)
    return zip(a, b)


class Agent(Policy):
    def __init__(self, input_size, hidden_layers, output_size, output_func,
                 hidden_act, output_act=None, bias=True):
        super().__init__()
        layers: List[nn.Module] = []
        layer_sizes = [input_size] + hidden_layers + [output_size]
        for idx, (layer1, layer2) in enumerate(pairwise(layer_sizes)):
            layers.append(nn.Linear(layer1, layer2, bias=bias))
            if idx < len(layer_sizes) - 2:
                layers.append(hidden_act())
        if output_act is not None:
            layers.append(output_act())
        self.output_func = output_func
        self.seq = nn.Sequential(*layers)

    def rollout(self, env):
        done = False
        obs = env.reset()
        total_reward = 0
        while not done:
            obs_tensor = torch.from_numpy(obs).float()
            action = self.output_func(self.seq(obs_tensor))
            obs, reward, done, _ = env.step(action)
            total_reward += reward
        return total_reward


def after_optimize_hook(self):
    total_reward = 0
    n_rollout = self.config.nes.n_rollout
    for _ in range(n_rollout):
        total_reward += self.eval_policy(self.policy)
    total_reward /= n_rollout
    if total_reward > self.config.user_variables.best_reward:
        self.config.user_variables.best_reward = total_reward
    best_reward = self.config.user_variables.best_reward
    print(f'Gen: {self.gen} Test Reward: {total_reward} Best Reward: {best_reward}')


config = Config(default_config)


@config('environment')
class EnvironmentConfig():
    id = 'LunarLander-v2'


@config('policy')
class PolicyConfig():
    policy = Agent
    env = gym.make(config.environment.id)
    input_size: int = env.observation_space.shape[0]
    if isinstance(env.action_space, gym.spaces.Discrete):
        output_size: int = env.action_space.n
        output_func: Callable[[int], int] = lambda x: x.argmax().item()
    else:
        output_size: int = env.action_space.shape[0]
        output_func: Callable[[int], int] = lambda x: x.detach().numpy()
    hidden_layers: List[int] = field(default_factory=lambda: [64, 32])
    hidden_act: nn.Module = nn.ReLU
    output_act: Optional[nn.Module] = nn.Tanh
    bias: bool = True


@config('optimizer')
class OptimizerConfig():
    lr = 0.02
    optim_type = torch.optim.Adam


@config('nes')
class NESConfig():
    n_rollout = 5
    n_step = 500
    l2_decay = 0.005
    population_size = 256
    sigma = 0.02
    seed = 123123
    after_optimize_hook = after_optimize_hook


@config('user_variables')
class UserVariables():
    """Users can initialize variables as config to use in the hook function."""
    best_reward: float = -np.inf


if __name__ == '__main__':
    nes = NES(config)
    nes.train()

    # Evaluate trained policy
    print(f'Reward: {nes.eval_policy(nes.policy)}')
