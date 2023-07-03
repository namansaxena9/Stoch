import random

import numpy
import numpy as np
import torch
import torch.nn as nn


STATE_PATH = r"/Users/navakrish/Desktop/IISC/state_buffer.npy"
NEXT_STATE_PATH = r"/Users/navakrish/Desktop/IISC/next_state_buffer.npy"
ACTION_PATH = r"/Users/navakrish/Desktop/IISC/action_buffer.npy"
NEXT_ACTION_PATH = r"/Users/navakrish/Desktop/IISC/next_action_buffer.npy"
REWARD_PATH = r"/Users/navakrish/Desktop/IISC/reward_buffer.npy"


class Datagen(object):
    def __init__(self):
        self.present_states = self.get_states()
        self.present_action = self.get_action()
        self.next_state = self.get_next_states()
        self.next_action = self.get_next_action()
        self.reward = self.get_rewards()

        obs_lower_limits = np.array(
            [
                -1, -1, -1, -3.14, -3.14, -3.14, -5, -5, -5, -3.14, -3.14, -3.14, -3.14, -3.14, -3.14, -3.14, -3.14,
                -3.14, -3.14, -3.14,
                -3.14, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -1, -1, -1, -1, -1, -1,
                -1, -1, -2.5, -2.5, -2.5, -2.5, 0, -3.14, -3.14, -3.14, -3.14, -3.14, -3.14, -3.14, -3.14, -3.14,
                -3.14, -3.14, -3.14, -3.14, -3.14, -3.14, -3.14, -3.14, -3.14, -3.14, -3.14, -3.14, -3.14, -3.14, -3.14,
                -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,
                -100, -100, -100, -100, -100, -100, -100, -2.5, -2.5, -2.5, -2.5, -2.5, -2.5, -2.5, -2.5, -2.5, -2.5,
                -2.5, -2.5, -2.5, -2.5, -2.5,
                -2.5, -2.5, -2.5, -2.5, -2.5, -2.5, -2.5, -2.5, -2.5, -2.5, -2.5, -2.5, -2.5, -2.5, -2.5, -2.5, -2.5,
                -2.5, -2.5, -2.5, -2.5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -10, -10, -10, -10, -10, -10,
                -10, -10, -10, -10, -10, -10, -10, -10, -10, -10, -10, -10, -10, -10, -10, -10, -10, -10, -10, -10,
                -10, -10, -10, -10, -10, -10, -10, -10, -10, -10, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0,
            ]
        )
        obs_upper_limits = np.array(
            [
                1, 1, 1, 3.14, 3.14, 3.14, 5, 5, 5, 3.14, 3.14, 3.14, 3.14, 3.14, 3.14, 3.14, 3.14, 3.14, 3.14, 3.14,
                3.14, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 1, 1, 1, 1, 1, 1, 1, 1, 2.5, 2.5, 2.5,
                2.5, 2.5, 3.14, 3.14, 3.14, 3.14, 3.14, 3.14, 3.14, 3.14, 3.14, 3.14, 3.14, 3.14, 3.14, 3.14, 3.14,
                3.14, 3.14, 3.14, 3.14, 3.14, 3.14, 3.14, 3.14, 3.14, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100,
                100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5,
                2.5, 2.5, 2.5, 2.5,
                2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5,
                2.5, 2.5, 2.5, 2.5, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 10, 10, 10, 10, 10, 10, 10,
                10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10,
                10, 10, 10, 1000, 1000,
                1000, 1000, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 50, 50, 50
            ])

        self.observation_factor = []
        for low, high in zip(list(obs_lower_limits), list(obs_upper_limits)):
            mini = abs((low - high)/2)
            maxi = abs((low + high)/2)
            self.observation_factor.append(max(mini, maxi))

        self.observation_factor = numpy.array(self.observation_factor)
        return

    def get_states(self):
        states = np.load(STATE_PATH)

        return states

    def get_next_states(self):
        next_states = np.load(NEXT_STATE_PATH)

        return next_states

    def get_rewards(self):
        rewards = np.load(REWARD_PATH)

        return rewards

    def get_action(self):
        action = np.load(ACTION_PATH)

        return action

    def get_next_action(self):
        next_action = np.load(NEXT_ACTION_PATH)

        return next_action

    def normalize_states(self, state):
        norm_state = state / self.observation_factor

        return norm_state

    def normalize_rewards(self, reward):
        norm_factor = 3
        norm_reward = reward / norm_factor

        return norm_reward

    def get_batch(self,BATCH_SIZE=32,):
        states = []
        actions = []
        rewards = []
        next_states = []
        next_actions = []

        indices = random.sample(range(15000), BATCH_SIZE)
        for index in indices:
            state = self.present_states[index]
            action = self.present_action[index]
            reward = self.reward[index]
            next_state = self.next_state[index]
            next_action = self.next_action[index]

            norm_state = self.normalize_states(state)
            norm_reward = self.normalize_rewards(reward)

            states.append(norm_state)
            actions.append(action)
            rewards.append(norm_reward)
            next_states.append(next_state)
            next_actions.append(next_action)

        states = torch.tensor(states)
        actions = torch.tensor(actions)
        rewards = torch.tensor(rewards)
        next_states = torch.tensor(next_states)
        next_actions = torch.tensor(next_actions)

        return states, actions, rewards, next_states, next_actions


if __name__ == '__main__':
    dg = Datagen()
    states, actions, rewards, next_states, next_actions = dg.get_batch(32)
    print("vals", states[0], actions[0], rewards[0], next_actions[0], next_states[0])
