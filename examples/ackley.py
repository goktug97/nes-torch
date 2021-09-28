from typing import Type

import torch
from torch import nn
import numpy as np

from nes import NES, Policy, default_config
from nes.config import default_config, Config


config = Config(default_config)


class Ackley(Policy):
    def __init__(self):
        super().__init__()
        self.params = nn.Parameter(torch.rand(2), requires_grad=False)

    def evaluate(self):
        x = self.params[0]
        y = self.params[1]
        first_term = -20 * torch.exp(-0.2*torch.sqrt(0.5*(x**2+y**2)))
        second_term = -torch.exp(0.5*(torch.cos(2*np.pi*x)+np.cos(2*np.pi*y)))+np.e + 20
        return -(second_term + first_term).item()


@config('policy')
class PolicyConfig():
    policy: Type[Policy] = Ackley


@config('optimizer')
class OptimizerConfig():
    lr: float = 0.02
    optim_type: Type[torch.optim.Optimizer] = torch.optim.Adam


@config('nes')
class NESConfig():
    n_step: int = 300
    l2_decay: float = 0.0
    population_size: int = 256
    sigma: float = 0.2
    seed: int = 123123


if __name__ == '__main__':
    nes = NES(config)

    @nes.optimize.add_hook()
    def after_optimize(self, *args, **kwargs):
        reward = self.policy.evaluate()
        print(f'Generation: {self.gen} Reward: {reward}')

    nes.train()
