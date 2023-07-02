import gym
import torch
from stable_baselines3 import PPO
import torch.nn as nn
# import numpy as np
# from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize, SubprocVecEnv
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from QuadrupedRobotEnv_4 import QuadrupedRobotEnv
import os

qp = QuadrupedRobotEnv(render = True, max_episode_steps=5000)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


gym.envs.register(
    id='MyCustomEnv-v0',
    entry_point='environment:QuadrupedRobotEnv',
    kwargs={}
)


def make_env():
    env = gym.make('MyCustomEnv-v0')
    return env

class CustomNN(BaseFeaturesExtractor):
    """
    :param observation_space: (gym.Space)
    :param features_dim: (int) Number of features extracted.
        This corresponds to the number of unit for the last layer.
    """

    def __init__(self, observation_space: qp.observation_space, features_dim: int = 16):
        super().__init__(observation_space, features_dim)
        # self.features_extractor.to(device)
        self.l1 = nn.Linear(71, 72)
        self.a1 = nn.ReLU()
        self.l2 = nn.Linear(72, 64)
        self.a2 = nn.ReLU()

        self.l3 = nn.Linear(194, 256)
        self.a3 = nn.ReLU()
        self.l4 = nn.Linear(256, 128)
        self.a4 = nn.ReLU()
        self.l5 = nn.Linear(128, 64)
        self.a5 = nn.ReLU()
        self.l6 = nn.Linear(64, features_dim)

        if torch.cuda.is_available():
            self.cuda()

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        observations = observations.to(device)
        obs = observations.tolist()[0]
        # print("obs", obs)
        proprioception = obs[:130]
        extrinsics = obs[130:]
        # print("len extrincics", len(extrinsics))
        proprioception = torch.Tensor(proprioception)
        extrinsics = torch.Tensor(extrinsics)
        extrinsics = extrinsics.to(device)
        proprioception = proprioception.to(device)

        extrinsics = self.l1(extrinsics)
        extrinsics = self.a1(extrinsics)
        extrinsics = self.l2(extrinsics)
        extrinsics = self.a2(extrinsics)

        state = torch.cat((proprioception, extrinsics), dim=0)
        state.to(device)
        state = self.l3(state)
        state = self.a3(state)
        state = self.l4(state)
        state = self.a4(state)
        state = self.l5(state)
        state = self.a5(state)
        state = self.l6(state)

        return state


# num_env = 1
# env_fns = [make_env for _ in range(num_env)]
# # envs = [VecNormalize(env, norm_reward=True, norm_obs=True) for env in env_fns]
# env = SubprocVecEnv(env_fns)
# v_env = VecNormalize(env, norm_obs=True, norm_reward=True)

# # v_env = DummyVecEnv([lambda: env])
# # v_env = VecNormalize(env, norm_reward=True, norm_obs=True)

# v_env.reset()

policy_kwargs = dict(
features_extractor_class=CustomNN,
features_extractor_kwargs=dict(features_dim=16),
log_std_init=-1.6094
)
policy_kwargs = dict(net_arch=[201, 256, 128, 64, 32, 16], log_std_init=-1.6094)

timesteps = 100000
model = PPO("MlpPolicy", qp, verbose=1, learning_rate=0.001, policy_kwargs=policy_kwargs,
            batch_size=100000, gae_lambda=0.95, gamma=0.998, n_steps=25000, clip_range=0.2,
            clip_range_vf=0.2, device=device)
ppo_agent = PPO.load("./PPO_2/5100000.zip", env=qp)

n_episode = 10
total_reward = 0
for i in range(n_episode):
    state = qp.reset()
    done = False
    epi_reward = 0
    print("true")
    while not done:
        action,_ = ppo_agent.predict(state)
        state, reward, done, _ = qp.step(action)
        epi_reward+=reward
    
    total_reward+=epi_reward
print("Average Reward::", total_reward/n_episode) 

qp.close()

    
    
    
    
    
    
    
