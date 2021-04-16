from typing import Optional, Type, Callable

import torch
from pipcs import Config, Required, required

from .nes import Policy, NES


def after_optimize_hook(self):
    pass


default_config = Config()


@default_config('environment')
class EnvironmentConfig():
    id: Required['str'] = required


@default_config('policy')
class PolicyConfig():
    policy: Required[Type[Policy]] = required
    device: 'str' = 'cpu'


@default_config('optimizer')
class OptimizerConfig():
    lr: float = 0.02
    optim_type: Required[Type[torch.optim.Optimizer]] = required


@default_config('nes')
class NESConfig():
    n_rollout: int = 1
    n_step: Required[int] = required
    l2_decay: float = 0.005
    evolution_population: Required[int] = required
    sigma: float = 0.02
    seed: Optional[int] = None
    after_optimize_hook: Callable[[NES], None] = after_optimize_hook
