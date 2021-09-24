from typing import Callable, List, Optional
from itertools import tee
from functools import lru_cache

from nes import Policy

import gym
import torch
from torch import nn
import numpy as np


def pairwise(iterable):
    a, b = tee(iterable)
    next(b, None)
    return zip(a, b)


@lru_cache(maxsize=1)
def make_env(env_id):
    return gym.make(env_id)


class Agent(Policy):
    def __init__(self, env_id: str, hidden_layers: List[int],
                 hidden_act: Optional[nn.Module] = None,
                 output_act: Optional[nn.Module] = None,
                 bias: bool = True, n_rollout: int = 1,
                 seed: Optional[int] = None):
        super().__init__()

        self.n_rollout = n_rollout
        self.env: gym.Env = make_env(env_id)
        self.env.seed(seed)
        self.env.action_space.seed(seed)
        self.output_func: Callable[[int], int]
        output_size: int
        input_size: int = self.env.observation_space.shape[0]

        if isinstance(self.env.action_space, gym.spaces.Discrete):
            output_size: int = self.env.action_space.n
            self.output_func = lambda x: x.argmax().item()
        else:
            output_size: int = self.env.action_space.shape[0]
            self.output_func = lambda x: x.detach().numpy()

        layers: List[nn.Module] = []
        layer_sizes = [input_size] + hidden_layers + [output_size]
        for idx, (layer1, layer2) in enumerate(pairwise(layer_sizes)):
            layers.append(nn.Linear(layer1, layer2, bias=bias))
            if hidden_act is not None:
                if idx < len(layer_sizes) - 2:
                    layers.append(hidden_act())
        if output_act is not None:
            layers.append(output_act())
        self.seq = nn.Sequential(*layers)

    def evaluate(self):
        total_reward = 0
        for _ in range(self.n_rollout):
            done = False
            obs = self.env.reset()
            while not done:
                obs_tensor = torch.from_numpy(obs).float()
                action = self.seq(obs_tensor).argmax().item()
                obs, reward, done, _ = self.env.step(action)
                total_reward += reward
        return total_reward / self.n_rollout
