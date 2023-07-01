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

qp = QuadrupedRobotEnv()
models_dir = "."
log_dir = "."

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ppo_path = os.path.join(models_dir, 'PPO')
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
    features_extractor_kwargs=dict(features_dim=16),
    log_std_init=-1.6094
)
    # policy_kwargs = dict(net_arch=[201, 256, 128, 64, 32, 16], log_std_init=-1.6094)

    timesteps = 100000
    # model = PPO.load(model_dir, env=v_env)
    model = PPO("MlpPolicy", v_env, verbose=1, learning_rate=0.001, tensorboard_log=log_dir, policy_kwargs=policy_kwargs,
                batch_size=100000 * num_env, gae_lambda=0.95, gamma=0.998, n_steps=25000 * num_env, clip_range=0.2,
                clip_range_vf=0.2, device=device)

    for i in range(1, 15000):
        # model.learn(total_timesteps=timesteps, tb_log_name='PPO_R1')
        model.learn(total_timesteps=timesteps * num_env, reset_num_timesteps=False, tb_log_name='PPO_R1')
        model.save(f"{ppo_path}/{timesteps * i}")

