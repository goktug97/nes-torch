import os
import sys
import random
from abc import ABC, abstractmethod

import torch
from torch import nn
import numpy as np
from pipcs import Config

try:
    disable_mpi = os.environ.get('NESTORCH_DISABLE_MPI')
    if disable_mpi and disable_mpi != '0':
        raise ImportError
    from mpi4py import MPI
except ImportError:
    from .MPI import MPI
    MPI = MPI()

from .utils import *


class Policy(nn.Module, ABC):
    """Abstract subclass of nn.Module."""
    def __init__(self):
        super().__init__()

    @abstractmethod
    def evaluate(self, env):
        """This function should be implemented by the user and it
        should evaluate the model and return reward or negative loss."""
        pass


class NES():
    """
    :ivar int gen: Current generation
    :ivar Policy policy: Trained policy
    :ivar torch.optim.Optimizer optim: Optimizer of the policy
    """
    @hook
    def __init__(self, config: Config):
        config.check_config()
        self.config = config

        comm = MPI.COMM_WORLD
        self.n_workers = comm.Get_size()
        self.rank = comm.Get_rank()

        if config.nes.seed is not None:
            seed = config.nes.seed
        else:
            seed = random.randint(0, 1000)
            seed = comm.bcast(seed, root=0)

        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

        self.gen = 0

        self.policy = self.make_policy(**config.policy.to_dict())
        self.optim = self.make_optimizer(policy=self.policy, **config.optimizer.to_dict())
        self.dummy_policy = self.make_policy(**self.config.policy.to_dict())

    @staticmethod
    def make_policy(policy, **kwargs):
        """Helper function to create a policy."""
        assert issubclass(policy, Policy)
        return policy(**kwargs)

    @staticmethod
    def make_optimizer(policy, optim_type, **kwargs):
        """Helper function to create an optimizer."""
        return optim_type(policy.parameters(), **kwargs)

    @hook
    def sample(self, mean):
        normal = torch.distributions.normal.Normal(0, self.config.nes.sigma)
        epsilon = normal.sample([int(self.config.nes.population_size/2), mean.shape[0]])
        population_params = torch.cat((mean + epsilon, mean - epsilon))
        epsilons = torch.cat((epsilon, -epsilon))
        return population_params, epsilons

    @hook
    def evaluate(self, population_params):
        comm = MPI.COMM_WORLD
        rewards = []
        reward_array = np.zeros(self.config.nes.population_size, dtype=np.float32)
        batch = np.array_split(population_params, self.n_workers)[self.rank]
        for param in batch:
            torch.nn.utils.vector_to_parameters(param, self.dummy_policy.parameters())
            rewards.append(self.dummy_policy.evaluate())
        comm.Allgatherv([np.array(rewards, dtype=np.float32), MPI.FLOAT], [reward_array, MPI.FLOAT])
        return reward_array

    @hook
    def calculate_gradients(self, rewards, mean, epsilons):
        ranked_rewards = torch.from_numpy(rank_transformation(rewards)).unsqueeze(0)
        grad = -(torch.mm(ranked_rewards, epsilons) / (len(rewards) * self.config.nes.sigma))
        grad = (grad + mean * self.config.nes.l2_decay).squeeze()  # L2 Decay
        return grad

    @hook
    def optimize(self, grad):
        index = 0
        for parameter in self.policy.parameters():
            size = np.prod(parameter.shape)
            parameter.grad = grad[index:index+size].view(parameter.shape)
            # Limit gradient update to increase stability.
            parameter.grad.data.clamp_(-1.0, 1.0)
            index += size
        self.optim.step()

    def train(self):
        """Train ``self.policy`` for ``self.config.nes.n_steps`` to increase reward returns
        from the ``self.env`` using Natural Evolution Strategy gradient estimation."""
        torch.set_grad_enabled(False)
        if not self.rank == 0:
            f = open(os.devnull, 'w')
            sys.stdout = f

        for gen in range(self.config.nes.n_step):
            self.gen = gen

            # Sample
            mean = torch.nn.utils.parameters_to_vector(self.policy.parameters())
            population_params, epsilons = self.sample(mean)

            # Evaluate Population
            rewards = self.evaluate(population_params)

            # Calculate Gradients
            grad = self.calculate_gradients(rewards, mean, epsilons)

            self.optimize(grad)

        sys.stdout = sys.__stdout__
        torch.set_grad_enabled(True)
