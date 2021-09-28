from typing import Callable, List, Optional, Type
from dataclasses import field
from itertools import tee

import gym
import torch
from torch import nn
import numpy as np

from nes import NES, Policy, hook
from nes.config import Config, default_config

from modules import Agent


config = Config(default_config)


@config('nes')
class NESConfig():
    n_step: int = 300
    l2_decay: float = 0.005
    population_size: int = 256
    sigma: float = 0.02
    seed: int = 123123


@config('policy')
class PolicyConfig():
    policy: Type[Policy] = Agent

    # Below variables are user settings
    # they are passed to Agent as kwargs
    # env_id: str = 'LunarLander-v2'
    env_id: str = 'CartPole-v1'
    n_rollout: int = 2
    hidden_layers: List[int] = field(default_factory=lambda: [16])
    hidden_act: Optional[nn.Module] = nn.ReLU
    # output_act: Optional[nn.Module] = nn.Tanh
    bias: bool = True
    seed: int = config.nes.seed


@config('optimizer')
class OptimizerConfig():
    lr: float = 0.02
    optim_type: torch.optim.Optimizer = torch.optim.Adam


if __name__ == '__main__':
    @NES.__init__.add_hook()
    def init(self, *args, **kwargs):
        self.best_reward = -np.inf

    nes = NES(config)

    @nes.optimize.add_hook()
    def after_optimize(self, *args, **kwargs):
        reward = self.policy.evaluate()
        if reward > self.best_reward:
            self.best_reward = reward
        print(f'Gen: {self.gen} Test Reward: {reward} Best Reward: {self.best_reward}')

    nes.train()

    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    if comm.Get_rank() == 0:

        nes.env = gym.wrappers.Monitor(nes.env, './videos', force = True)

        setattr(nes, 'env.step', hook(nes.env.step.__func__))
        @nes.env.step.add_hook(after=False)
        def render(self, *args, **kwargs):
            self.render()

        print(f'Reward: {nes.policy.evaluate()}')
