import gym
from pipcs import Config
import torch
from torch import nn

from nes import NES, Policy, default_config


class Agent(Policy):
    def __init__(self):
        super().__init__()
        self.seq = nn.Sequential(
            nn.Linear(4, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 2))

    def rollout(self, env):
        done = False
        obs = env.reset()
        total_reward = 0
        while not done:
            obs_tensor = torch.from_numpy(obs).float()
            action = self.seq(obs_tensor).argmax().item()
            obs, reward, done, _ = env.step(action)
            total_reward += reward
        return total_reward


config = Config(default_config)


@config('environment')
class EnvironmentConfig():
    make_env = gym.make
    id: str = 'CartPole-v1'


@config('policy')
class PolicyConfig():
    policy = Agent


@config('optimizer')
class OptimizerConfig():
    lr = 0.02
    optim_type = torch.optim.Adam


@config('nes')
class NESConfig():
    n_rollout = 5
    n_step = 500
    population_size = 256
    seed = 123123


if __name__ == '__main__':
    nes = NES(config)

    @nes.optimize.add_hook()
    def after_optimize(self, *args, **kwargs):
        reward = self.eval_policy(self.policy)
        print(reward)

    nes.train()
