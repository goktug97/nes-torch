from typing import Optional, Type, Callable

import torch
from pipcs import Config, Required, required

from .nes import Policy, NES


def after_optimize_hook(self):
    pass


default_config = Config()


@default_config('environment')
class EnvironmentConfig():
    """**name: environment**

    :ivar str id: Gym environment id
    """
    id: Required['str'] = required


@default_config('policy')
class PolicyConfig():
    """**name: policy**

    :ivar Required[Type[Policy]] policy: torch.nn.Module with a rollout method
    :ivar str device: torch device
    """
    policy: Required[Type[Policy]] = required
    device: str = 'cpu'


@default_config('optimizer')
class OptimizerConfig():
    """**name: optimizer**

    :ivar float lr: Learning rate
    :ivar Required[Type[torch.optim.Optimizer]] optim_type: torch optim module
    """
    lr: float = 0.02
    optim_type: Required[Type[torch.optim.Optimizer]] = required


@default_config('nes')
class NESConfig():
    """**name: nes**

    :ivar int population_size: Population Size, higher means lower variance in gradient calculation but higher memory consumption.
    :ivar int n_step: Number of training steps
    :ivar float sigma: Standart deviation for population sampling
    :ivar int n_rollout: Number of episodes per sampled policy.
    :ivar Optional[int] seed: Random seed
    :ivar Callable[[NES],None] after_optimize_hook: Executed after optim.step()
    """
    n_rollout: int = 1
    n_step: Required[int] = required
    l2_decay: float = 0.005
    population_size: Required[int] = required
    sigma: float = 0.02
    seed: Optional[int] = None
    after_optimize_hook: Callable[[NES], None] = after_optimize_hook
