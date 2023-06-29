import math
import os
import random
import time

import gym
import numpy
from gym import spaces
import pybullet as p
import numpy as np

from QuadrupedRobotEnv_3 import QuadrupedRobotEnv
"""
env = QuadrupedRobotEnv(render = True)
env.reset()
done = False
count = 0
try:
    while(not done):
        count+=1
        action = np.zeros(16)
        _, _, done, _ = env.step(action)
        done = False
        print(count)
    env.close()    
except:
    env.close()

"""

env = QuadrupedRobotEnv(render = True)
env.reset()
done = False
count = 0
try:
    while(not done):
        phase0 = env.phase_leg0
        phase1 = env.phase_leg1
        phase2 = env.phase_leg2
        phase3 = env.phase_leg3
        
        # print("phases", phase0, phase1, phase2, phase3)
        if 0 <= phase0 <= math.pi:
            x0 = 0.1
            y0 = 0.005
        elif math.pi <= phase0 <= 2 * math.pi:
            x0 = -0.1
            y0 = 0
        
        if 0 <= phase1 <= math.pi:
            x1 = 0.1
            y1 = -0.005
        elif math.pi <= phase1 <= 2 * math.pi:
            x1 = -0.1
            y1 = 0
        
        if 0 <= phase2 <= math.pi:
            x2 = 0.1
            y2 = -0.005
        elif math.pi <= phase2 <= 2 * math.pi:
            x2 = -0.1
            y2 = 0
        
        if 0 <= phase3 <= math.pi:
            x3 = 0.1
            y3 = 0.005
        elif math.pi <= phase3 <= 2 * math.pi:
            x3 = -0.1
            y3 = 0
        
        if env.counter <= 16:
            x0 = -0.05
            y0 = 0
        
            x1 = 0
            y1 = 0.005
        
            x2 = 0
            y2 = 0.005
        
            x3 = -0.05
            y3 = 0.005
        
        print("x0", x0)
        print("x3", x3)
        
        action = np.array([0, 0, 0, 0, x0, 0, 0, x1, 0, 0, x2, 0, 0, x3, 0, 0])
        # if env.counter % 25 == 0:
        #     force = [random.uniform(-100, 100) for i in range(3)]  # Specify the force vector [X, Y, Z]
        #     position = [0, 0, 0]  # Specify the position where the force is applied [X, Y, Z]
        #     link_index = -1
        #     p.applyExternalForce(env.robot_id, link_index, force, position, flags=p.WORLD_FRAME)
        #     print(force)
        
        env.step(action)    
    env.close()    
except:
    env.close()
