import csv

import gym
import time
from environment import QuadrupedRobotEnv
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
import os

# models_dir = os.path.join(".", "PPO")  # "./PPO"

# d = os.listdir(".")[4].strip()
# print(models_dir)
# models_dir = os.path.join(".", "PPO")
# models = os.listdir(models_dir)
# print(models)

NUM = "500000"
model_dir = os.path.join("./PPO/", NUM + ".zip")

gym.envs.register(
    id='MyCustomEnv-v0',
    entry_point='environment:QuadrupedRobotEnv',
    kwargs={}
)
env = gym.make('MyCustomEnv-v0')
v_env = DummyVecEnv([lambda: env])
# v_env = VecNormalize(v_env, norm_reward=True, norm_obs=True)
v_env = VecNormalize.load("./ENV/" + NUM, v_env)

# v_env.reset()

model = PPO.load(model_dir, env=v_env, device='cpu')
# v_env.to('cpu')

episodes = 20


for i in range(episodes):
    obs = v_env.reset()
    done = False

    while not done:
        actions, _states = model.predict(obs)
        time.sleep(0.085)
        obs, reward, done, info = v_env.step(actions)





