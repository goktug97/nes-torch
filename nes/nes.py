import os
import sys
import random
from abc import ABC, abstractmethod

import torch
from torch import nn
import numpy as np
from mpi4py import MPI
import gym
from pipcs import Config

from .utils import *


class Policy(nn.Module, ABC):
    """Abstract subclass of nn.Module."""
    def __init__(self):
        super().__init__()

    @abstractmethod
    def rollout(self, env):
        """This function should be implemented by the user and
        it should evaluate the model in the given environment and
        return the total reward."""
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

        if config.nes.seed is not None:
            torch.manual_seed(config.nes.seed)
            np.random.seed(config.nes.seed)
            random.seed(config.nes.seed)
            torch.cuda.manual_seed(config.nes.seed)
            torch.cuda.manual_seed_all(config.nes.seed)
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
            self.env = self.make_env(**config.environment.to_dict())
            self.env.seed(config.nes.seed)
            self.env.action_space.seed(config.nes.seed)

        self.gen = 0

        comm = MPI.COMM_WORLD
        self.n_workers = comm.Get_size()
        self.rank = comm.Get_rank()

        self.policy = self.make_policy(**config.policy.to_dict())
        self.optim = self.make_optimizer(policy=self.policy, **config.optimizer.to_dict())
        self.dummy_policy = self.make_policy(**self.config.policy.to_dict())

    @staticmethod
    def make_env(make_env, **kwargs):
        """Helper function to create a gym environment."""
        return make_env(**kwargs)

    @staticmethod
    def make_policy(policy, **kwargs):
        """Helper function to create a policy."""
        assert issubclass(policy, Policy)
        return policy(**kwargs)

    @staticmethod
    def make_optimizer(policy, optim_type, **kwargs):
        """Helper function to create an optimizer."""
        return optim_type(policy.parameters(), **kwargs)

    def eval_policy(self, policy):
        """Evaluate policy on the ``self.env`` for ``self.config.nes.n_rollout times``"""
        total_reward = 0
        for _ in range(self.config.nes.n_rollout):
            total_reward += policy.rollout(self.env)
        return total_reward / self.config.nes.n_rollout

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
            rewards.append(self.eval_policy(self.dummy_policy))
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
