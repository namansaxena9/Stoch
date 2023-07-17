import gym
import torch
from stable_baselines3 import PPO
import torch.nn as nn
# import numpy as np
# from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize, SubprocVecEnv
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from environment import QuadrupedRobotEnv
import os

print("RUNNING agent_ppo....")
qp = QuadrupedRobotEnv()
models_dir = "."
log_dir = "."
env_dir = "."

NUM = "20500000"
model_dir = os.path.join(".", "backup1", "PPO",  NUM + ".zip")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ppo_path = os.path.join(models_dir, 'PPO')
env_path = os.path.join(env_dir, "ENV")

# model_dir = os.path.join(".", "PPO", "17600000")

# ------------------------------------------------------------
# model_dir = os.path.join(".", "PPO1", "1400000")
# # print(model_dir)
# gym.envs.register(
#     id='MyCustomEnv-v0',
#     entry_point='environment:QuadrupedRobotEnv',
#     kwargs={}
# )
# env = gym.make('MyCustomEnv-v0')
# v_env = DummyVecEnv([lambda: env])
# v_env = VecNormalize(v_env, norm_reward=True)
# v_env.reset()
# model = PPO.load(model_dir, env=v_env)
# print("Model Loaded!")
# ---------------------------------------------------------------------------

if not os.path.exists(models_dir):
    os.makedirs(models_dir)

if not os.path.exists(log_dir):
    os.makedirs(log_dir)

if not os.path.exists(env_path):
    os.makedirs(env_path)

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

    def __init__(self, observation_space: gym.spaces.Space, features_dim: int = 35):
        super().__init__(observation_space, features_dim)
        # self.features_extractor.to(device)
        self.l1 = nn.Linear(59, 32)
        self.a1 = nn.ReLU()
        self.l2 = nn.Linear(62, 32)
        self.a2 = nn.ReLU()
        self.l3 = nn.Linear(68, 32)
        self.a3 = nn.ReLU()
        # self.a4 = nn.ReLU()
        # self.l5 = nn.Linear(128, 64)
        # self.a5 = nn.ReLU()
        # self.l6 = nn.Linear(64, features_dim)

        if torch.cuda.is_available():
            self.cuda()

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        # observations = observations.to(device)
        # print("obs", observations)
        history = observations[:, :36]
        proprioception = observations[:, 36:66]
        command = observations[:, 66:69]
        extrinsics = observations[:, 69:]
        # print("len extrincics", len(extrinsics))

        extrinsics = self.l1(extrinsics)
        extrinsics = self.a1(extrinsics)

        state = torch.cat((extrinsics, proprioception), dim=1)
        state = self.l2(state)
        state = self.a2(state)

        state = torch.cat((state, history), dim=1)
        state = self.l3(state)
        state = self.a3(state)

        state = torch.cat((state, command), dim=1)
        # print(state)

        return state


if __name__ == '__main__':
    num_env = 10
    env_fns = [make_env for _ in range(num_env)]
    # envs = [VecNormalize(env, norm_reward=True, norm_obs=True) for env in env_fns]
    env = SubprocVecEnv(env_fns)
    v_env = VecNormalize(env, norm_obs=True, norm_reward=True)

    # v_env = DummyVecEnv([lambda: env])
    # v_env = VecNormalize(env, norm_reward=True, norm_obs=True)

    v_env.reset()

    policy_kwargs = dict(
    features_extractor_class=CustomNN,
    features_extractor_kwargs=dict(features_dim=35),
    log_std_init=-1.6094
)
    # policy_kwargs = dict(net_arch=[201, 256, 128, 64, 32, 16], log_std_init=-1.6094)

    timesteps = 100000

    # model = PPO.load(model_dir, env=v_env)
    # v_env = VecNormalize.load("./ENV/" + NUM, v_env)

    model = PPO("MlpPolicy", v_env, verbose=1, learning_rate=0.001, tensorboard_log=log_dir, policy_kwargs=policy_kwargs,
                batch_size=100000, gae_lambda=0.95, gamma=0.998, n_steps=25000, clip_range=0.2,
                clip_range_vf=0.2, device=device)

    for i in range(1, 15000):
        # model.learn(total_timesteps=timesteps, tb_log_name='PPO_R1')
        model.learn(total_timesteps=timesteps * num_env, reset_num_timesteps=False, tb_log_name='PPO_R1')
        model.save(f"{ppo_path}/{timesteps * i}")
        v_env.save(f"{env_path}/{timesteps * i}")

