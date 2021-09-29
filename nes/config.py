from typing import Optional, Type, Callable, Any

import torch

from .nes import Policy, NES
from .dataclass_config import *


default_config = Config()


@default_config('policy')
class PolicyConfig():
    """**name: policy**

    :ivar Required[Type[Policy]] policy: torch.nn.Module with a rollout method
    """
    policy: Required[Type[Policy]] = Required()


@default_config('optimizer')
class OptimizerConfig():
    """**name: optimizer**

    :ivar float lr: Learning rate
    :ivar Required[Type[torch.optim.Optimizer]] optim_type: torch optim module
    """
    lr: float = 0.02
    optim_type: Required[Type[torch.optim.Optimizer]] = Required()


@default_config('nes')
class NESConfig():
    """**name: nes**

    :ivar int population_size: Population Size, higher means lower variance in
        gradient calculation but higher memory consumption and longer training time.
    :ivar int n_step: Number of training steps
    :ivar float sigma: Standart deviation for population sampling
    :ivar Optional[int] seed: Random seed
    """
    n_step: Required[int] = Required()
    l2_decay: float = 0.005
    population_size: Required[int] = Required()
    sigma: float = 0.02
    seed: Optional[int] = None
