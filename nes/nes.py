import os
import sys
import random
from functools import lru_cache
from abc import ABC, abstractmethod

import torch
from torch import nn
import numpy as np
from mpi4py import MPI
import gym
from pipcs import Config


@lru_cache(maxsize=1)
def _center_function(population_size):
    centers = np.arange(0, population_size, dtype=np.float32)
    centers = centers / (population_size - 1)
    centers -= 0.5
    return centers


def _compute_ranks(rewards):
    rewards = np.array(rewards)
    ranks = np.empty(rewards.size, dtype=int)
    ranks[rewards.argsort()] = np.arange(rewards.size)
    return ranks


def rank_transformation(rewards):
    ranks = _compute_ranks(rewards)
    values = _center_function(rewards.size)
    return values[ranks]


class Policy(nn.Module, ABC):
    """Abstract subclass of nn.Module."""
    def __init__(self):
        super().__init__()

    @abstractmethod
    def rollout(self, env):
        """This function should be implemented by the user and it should evaluate the model in the given environment and return the total reward."""
        pass


class NES():
    """
    :ivar int gen: Current generation
    :ivar Policy policy: Trained policy
    :ivar torch.optim.Optimizer optim: Optimizer of the policy
    """
    def __init__(self, config: Config):
        config.check_config()
        self.config = config

        torch.manual_seed(config.nes.seed)
        np.random.seed(config.nes.seed)
        random.seed(config.nes.seed)
        torch.cuda.manual_seed(config.nes.seed)
        torch.cuda.manual_seed_all(config.nes.seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        self.env = gym.make(**config.environment.to_dict())
        self.env.seed(config.nes.seed)
        self.env.action_space.seed(config.nes.seed)

        self.gen = 0

        self.policy = self.make_policy(**config.policy.to_dict())
        self.optim = self.make_optimizer(policy=self.policy, **config.optimizer.to_dict())

    def make_policy(self, policy, device, **kwargs):
        """Helper function to create a policy."""
        assert issubclass(policy, Policy)
        return policy(**kwargs).to(device)

    def make_optimizer(self, policy, optim_type, lr, **kwargs):
        """Helper function to create an optimizer."""
        return optim_type(policy.parameters(), lr=lr, **kwargs)

    def eval_policy(self, policy):
        """Evaluate policy on the ``self.env`` for ``self.config.nes.n_rollout times``"""
        total_reward = 0
        for _ in range(self.config.nes.n_rollout):
            total_reward += policy.rollout(self.env)
        return total_reward / self.config.nes.n_rollout

    def train(self):
        """Train ``self.policy`` for ``self.config.nes.n_steps`` to increase reward returns from the ``self.env`` using Natural Evolution Strategy gradient estimation."""
        torch.set_grad_enabled(False)
        comm = MPI.COMM_WORLD
        n_workers = comm.Get_size()
        rank = comm.Get_rank()
        if not rank == 0:
            f = open(os.devnull, 'w')
            sys.stdout = f
        device = self.config.policy.device
        dummy_policy = self.make_policy(**self.config.policy.to_dict())
        for gen in range(self.config.nes.n_step):
            self.gen = gen

            # Sample
            mean = torch.nn.utils.parameters_to_vector(self.policy.parameters())
            normal = torch.distributions.normal.Normal(0, self.config.nes.sigma)
            epsilon = normal.sample([int(self.config.nes.population_size/2), mean.shape[0]])
            population_params = torch.cat((mean + epsilon, mean - epsilon))
            epsilons = torch.cat((epsilon, -epsilon))

            # Evaluate
            rewards = []
            reward_array = np.zeros(self.config.nes.population_size, dtype=np.float32)
            batch = np.array_split(population_params, n_workers)[rank]
            for param in batch:
                torch.nn.utils.vector_to_parameters(param.to(device), dummy_policy.parameters())
                rewards.append(self.eval_policy(dummy_policy))
            comm.Allgatherv([np.array(rewards, dtype=np.float32), MPI.FLOAT], [reward_array, MPI.FLOAT])

            # Calculate gradients
            ranked_rewards = torch.from_numpy(rank_transformation(reward_array)).unsqueeze(0).to(device)
            grad = -(torch.mm(ranked_rewards, epsilons) / (len(reward_array) * self.config.nes.sigma))
            grad = (grad + mean * self.config.nes.l2_decay).squeeze()  # L2 Decay

            # Optimize
            self.optim.zero_grad()
            index = 0
            for parameter in self.policy.parameters():
                size = np.prod(parameter.shape)
                parameter.grad = grad[index:index+size].view(parameter.shape).to(device)
                # Limit gradient update to increase stability.
                parameter.grad.data.clamp_(-1.0, 1.0)
                index += size
            self.optim.step()

            self.config.nes.after_optimize_hook(self)

        sys.stdout = sys.__stdout__
        torch.set_grad_enabled(True)
