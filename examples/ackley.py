from pipcs import Config
import torch
from torch import nn
import numpy as np

from nes import NES, Policy, default_config


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
    policy = Ackley


@config('optimizer')
class OptimizerConfig():
    lr = 0.02
    optim_type = torch.optim.Adam


@config('nes')
class NESConfig():
    n_step = 300
    l2_decay = 0.0
    population_size = 256
    sigma = 0.2
    seed = 123123


if __name__ == '__main__':
    nes = NES(config)

    @nes.optimize.add_hook()
    def after_optimize(self, *args, **kwargs):
        reward = self.policy.evaluate()
        print(f'Generation: {self.gen} Reward: {reward}')

    nes.train()
