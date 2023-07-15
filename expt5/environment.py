"""
This version supports CMU model with 1 environment
curriculum modified
reward coeff 1 and gain = 0.7
"""

import math
import os
import random
import time

import gym
import numpy
from gym import spaces
import pybullet as p
import numpy as np
import torch

BASEDIR = "../"
URDFPATH = os.path.join(BASEDIR, "convex_mpc", "mpc_controller", "stoch3_description", "urdf", "stoch3.urdf")
# print(URDFPATH)
SEARCHPATH = os.path.join(BASEDIR, "convex_mpc", "mpc_controller", "stoch3_description", "meshes")
BULLETPATH = os.path.join(BASEDIR, "convex_mpc", "mpc_controller", "stoch3_description", "urdf", "stoch3_env2.bullet")
num_env = 1


class QuadrupedRobotEnv(gym.Env):
    def __init__(self, max_episode_steps=2000):

        self.physics_client = p.connect(p.DIRECT)
        #self.physics_client = p.connect(p.GUI)

        self.max_episode_steps = max_episode_steps

        self._motor_offset = np.array([0] * 12)
        self.friction = 0.7
        self.fx = 0
        self.fy = 0
        self.fz = 0
        # self._motor_offset = np.array([0, -0.8, -1.5, 0, 0.8, -1.5, 0, -0.8, -1.5, 0, 0.8, -1.5])

        self._motor_direction = np.array([1, 1, 1,
                                          1, 1, 1,
                                          1, 1, 1,
                                          1, 1, 1])

        # Loading the URDF file with meshes
        p.setAdditionalSearchPath(SEARCHPATH)  # Path to the meshes folder
        self.robot_id = p.loadURDF(URDFPATH, [0, 0, 0.52])
        self.NUM_LEGS = 4

        self.time_interval = 0.05
        self.two_pi = 2 * math.pi

        # Definining terrain parameters
        self.terrain_size = 256
        self.terrain_scale = 1

        self.heights = np.random.uniform(0, 0, size=(self.terrain_size, self.terrain_size))

        # Creating a PyBullet heightfield shape from the height values
        terrain_shape = p.createCollisionShape(shapeType=p.GEOM_HEIGHTFIELD,
                                               meshScale=[self.terrain_scale, self.terrain_scale, 1],
                                               heightfieldTextureScaling=(self.terrain_size - 1) / 2,
                                               heightfieldData=self.heights.flatten().tolist(),
                                               numHeightfieldRows=self.terrain_size,
                                               numHeightfieldColumns=self.terrain_size,
                                               )

        self.terrain_body = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=terrain_shape)
        p.changeVisualShape(self.terrain_body, -1, textureUniqueId=-1)
        p.changeVisualShape(self.terrain_body, -1, rgbaColor=[0.545, 0.271, 0.075, 1])

        # Setting the position of the terrain to be at the center of the world
        p.resetBasePositionAndOrientation(self.terrain_body, [0, 0, 0], [0, 0, 0, 1])
        p.resetBasePositionAndOrientation(self.robot_id, [0, 0, 0.69], [0, 0, 0, 1])

        p.setGravity(0, 0, -9.81)
        # p.setPhysicsEngineParameter(contactStiffness=100000.0)
        p.setTimeStep(self.time_interval, self.physics_client)
        ids = [3, 7, 11, 15]
        for id in ids:
            p.changeDynamics(
                self.robot_id,
                id,
                lateralFriction=self.friction,
                spinningFriction=self.friction,
                rollingFriction=0.014,
                physicsClientId=0,
                restitution=0.35,
                contactStiffness=6500,
                contactDamping=0.05
            )
        p.changeDynamics(
            self.terrain_body,
            -1,
            lateralFriction=self.friction,
            spinningFriction=self.friction,
            rollingFriction=0.014,
            physicsClientId=0,
            restitution=0.35,
            contactStiffness=6500,
            contactDamping=0.05
        )

        self.state_id = p.saveState()
        p.saveBullet(BULLETPATH)
        # num_joints = p.getNumJoints(self.robot_id)

        self.desired_speed = 0.6
        self.desired_twisting_speed = 0.5

        self.twist_dir = 0
        self.angle = 0
        self.vx = self.desired_speed * math.cos(self.angle)
        self.vy = self.desired_speed * math.sin(self.angle)

        # self.valid_indices = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
        self.leg_valid = [0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14]
        self.leg_indices = [0, 1, 2, 3]
        self.action_tminus1 = [0.0 for i in range(12)]

        self.action_factors = np.array([0.15, 0.4, 0.4, 0.15, 0.4, 0.4, 0.15, 0.4, 0.4, 0.15, 0.4, 0.4])

        self.action_space = spaces.Box(low=-1, high=1, shape=(12,))

        self.obs_lower_limits = np.array(
            [
                -0.15, -0.4, -0.4, -0.15, -0.4, -0.4, -0.15, -0.4, -0.4, -0.15, -0.4, -0.4,  # joint action history
                0, 0.3, -3.14159, -1, 0.3, -3.14159, 0, 0.3, -3.14159, -1, 0.3, -3.14159,  # joint state history
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  # joint velocities

                0, 0.3, -3.14159, -1, 0.3, -3.14159, 0, 0.3, -3.14159, -1, 0.3, -3.14159,  # present joint states
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  # present joint velocities

                -3.14, -3.14, -3.14,  # base orientation
                -5, -5, -5,  # base velocites
                -1, -1, -1,  # desired direction turning direction

                0, 0, 0, 0,  # contact states
                -10, -10, -10, -10, -10, -10,  # terrain height
                -10, -10, -10, -10, -10, -10, -10, -10, -10, -10, -10, -10, -10, -10, -10, -10, -10, -10, -10, -10,
                -10, -10, -10, -10, -10, -10,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  # terrain normal
                0, 0, 0, 0, 0, 0, 0,  # friction and ext force
                0, 0, 0, 0,  # contact forces

            ]
        )

        self.obs_upper_limits = np.array(
            [
                0.15, 0.4, 0.4, 0.15, 0.4, 0.4, 0.15, 0.4, 0.4, 0.15, 0.4, 0.4,
                1, 1.5708, -1, 0, 1.5708, -1, 1, 1.5708, -1, 0, 1.5708, -1,
                100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100,

                1, 1.5708, -1, 0, 1.5708, -1, 1, 1.5708, -1, 0, 1.5708, -1,
                100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100,

                3.14, 3.14, 3.14,
                5, 5, 5,
                1, 1, 1,

                1, 1, 1, 1,
                10, 10, 10, 10, 10, 10, 10,
                10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10,
                1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                1.5, 1.5, 1.5, 1.5, 150, 150, 150,
                1000, 1000, 1000, 1000,
            ]
        )
        # print("len low", len(self.obs_lower_limits))

        self.observation_space = spaces.Box(low=self.obs_lower_limits, high=self.obs_upper_limits, shape=(128,),
                                            dtype=np.float32)

        angle_1 = 0.007
        angle_2 = 1.035
        angle_3 = 1.6  # 1.794
        self.initial_action = np.array(
            [-angle_1, angle_2, -angle_3, angle_1, angle_2, -angle_3, -angle_1, angle_2, -angle_3, angle_1, angle_2,
             -angle_3])

        self.act_lower_lim = np.array([0, 0.3, -3.14159, -1, 0.3, -3.14159, 0, 0.3, -3.14159, -1, 0.3, -3.14159])
        self.act_upper_lim = np.array([1, 1.5708, -1, 0, 1.5708, -1, 1, 1.5708, -1, 0, 1.5708, -1])

        self.present_state = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        self.present_vels = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])

        self.tminus1_state = self.present_state

        self.tminus1_vels = self.present_vels

        self.action_tminus1 = [0.0 for i in range(12)]
        self.action = [0.0 for i in range(12)]
        self.clipped_action = self.initial_action

        self.torques = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        self._foot_link_ids = [3, 7, 11, 15]
        self.num_motors = 12
        self.num_legs = 4
        # self.initial_copy = [0, -1, -1, 0, 1, -1, 0, -1, -1, 0, 1, -1]
        # self.initial_copy = [-1.5708, -1.5708, -3.14159, 1.5708, 1.5708, -3.14159, 1.5708, -1.5708, -3.14159, -1.5708, 1.5708, -3.14159]
        self.set_initial_stance(self.robot_id, self.leg_valid, list(self.initial_action))
        self.vx_target = 0.35  # 0.5 #m/s
        self.timesteps = 0
        self.counter = 0
        self.sim_steps = 0
        # self.prev_torq = [0.0 for i in range(12)]
        # self.reward_sum = 0.0
        # self.epi_count = 0
        self.done = False
        # self.data = 0.0
        self.lin_vel_coeff = 7
        self.lateral_coeff = 0.01
        # smooth_coeff = 0.00001
        self.ener_coeff = 0.00001
        self.z_vel_coeff = 0.001
        self.action_rate_coeff = 0.0001
        self.survival_bonus = 0.01



    def step(self, action_orig):
        # print("inside step")
        # print("time-steps=", self.timesteps)
        # print("counter=", self.counter)
        # print("base vels=", p.getBaseVelocity(self.robot_id)[0])

        action = np.multiply(action_orig, self.action_factors)
        self.action_tminus1 = self.action
        self.action = list(action)
        # print(action)
        act = np.add(self.initial_action, action)
        # print(act)
        clipped_action = np.clip(act, self.act_lower_lim, self.act_upper_lim)
        self.clipped_action = clipped_action
        action_joints = list(clipped_action)

        for i, joint_index in enumerate(self.leg_valid):
            p.setJointMotorControl2(self.robot_id, joint_index, controlMode=p.POSITION_CONTROL,
                                    targetPosition=action_joints[i], positionGain=0.7)

        p.stepSimulation()
        self.timesteps += 1
        self.counter += 1
        # time.sleep(0.035)

        # for i in range(4):
        #     for i, joint_index in enumerate(self.leg_valid):
        #         p.setJointMotorControl2(self.robot_id, joint_index, controlMode=p.POSITION_CONTROL,
        #                                 targetPosition=action_joints[i], positionGain=0.5)
        #
        #     p.stepSimulation()
        #     self.timesteps += 1
        #     self.counter += 1

        obs = self.get_obs()
        self.tminus1_state = self.present_state
        self.present_state = obs[36:48]
        # print(self.present_state, clipped_action)

        self.tminus1_vels = self.present_vels
        self.present_vels = obs[48:60]

        # self.action_tminus1 = action_orig
        reward, info = self.calculate_reward()
        done = self._is_done()

        return obs, reward, done, info

    def reset(self):
        # Reset the simulation and the robot's state
        # print("reset")
        p.resetSimulation()

        cameraTargetPosition = [0, 0, 0]  # Example target position [x, y, z]
        p.resetDebugVisualizerCamera(cameraDistance=3, cameraYaw=0, cameraPitch=-30,
                                     cameraTargetPosition=cameraTargetPosition)
        self.fx = 0
        self.fy = 0
        self.fz = 0
        # self.timesteps = 0
        self.done = False
        p.setAdditionalSearchPath(SEARCHPATH)  # Path to the meshes folder
        self.robot_id = p.loadURDF(URDFPATH, [0, 0, 0.52])
        self.terrain_size = 256
        self.terrain_scale = 1

        if self.counter < 45000000:
            self.heights = np.random.uniform(0, 0, size=(self.terrain_size, self.terrain_size))

            # Creating a PyBullet heightfield shape from the height values
            terrain_shape = p.createCollisionShape(shapeType=p.GEOM_HEIGHTFIELD,
                                                   meshScale=[self.terrain_scale, self.terrain_scale, 1],
                                                   heightfieldTextureScaling=(self.terrain_size - 1) / 2,
                                                   heightfieldData=self.heights.flatten().tolist(),
                                                   numHeightfieldRows=self.terrain_size,
                                                   numHeightfieldColumns=self.terrain_size,
                                                   )

            self.terrain_body = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=terrain_shape)
            p.changeVisualShape(self.terrain_body, -1, textureUniqueId=-1)

        else:
            # Definining terrain parameters
            frequency = 10.0  # Frequency of the fractal
            amplitude = 0.23  # Amplitude of the fractal

            # Generate the heights using fractal noise
            self.heights = np.zeros((self.terrain_size, self.terrain_size))

            # Generate fractal noise
            for octave in range(1, 3):  # Number of octaves
                lacunarity = 2.0  # Lacunarity of the fractal
                gain = 0.25  # Gain of the fractal
                octave_frequency = frequency * lacunarity ** octave
                octave_amplitude = amplitude * gain ** octave

                for i in range(self.terrain_size):
                    for j in range(self.terrain_size):
                        self.heights[i, j] += octave_amplitude * (2 * np.random.rand() - 1) * np.sin(
                            (i / self.terrain_size) * octave_frequency * np.pi
                        ) * np.sin((j / self.terrain_size) * octave_frequency * np.pi)

            # Creating a PyBullet heightfield shape from the height values
            terrain_shape = p.createCollisionShape(
                shapeType=p.GEOM_HEIGHTFIELD,
                meshScale=[self.terrain_scale, self.terrain_scale, 1],
                heightfieldTextureScaling=(self.terrain_size - 1) / 2,
                heightfieldData=self.heights.flatten().tolist(),
                numHeightfieldRows=self.terrain_size,
                numHeightfieldColumns=self.terrain_size,
            )

            # Create the terrain object
            self.terrain_body = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=terrain_shape)
            p.changeVisualShape(self.terrain_body, -1, textureUniqueId=-1)

        p.changeVisualShape(self.terrain_body, -1, rgbaColor=[0.545, 0.271, 0.075, 1])

        # Setting the position of the terrain to be at the center of the world
        p.resetBasePositionAndOrientation(self.terrain_body, [0, 0, 0], [0, 0, 0, 1])
        p.resetBasePositionAndOrientation(self.robot_id, [0, 0, 0.49], [0, 0, 0, 1])

        p.setGravity(0, 0, -9.81)
        # p.setPhysicsEngineParameter(contactStiffness=100000.0)
        p.setTimeStep(self.time_interval, self.physics_client)
        ids = [3, 7, 11, 15]
        for id in ids:
            p.changeDynamics(
                self.robot_id,
                id,
                lateralFriction=self.friction,
                spinningFriction=self.friction,
                rollingFriction=0.014,
                physicsClientId=0,
                restitution=0.35,
                contactStiffness=6500,
                contactDamping=0.05
            )
        p.changeDynamics(
            self.terrain_body,
            -1,
            lateralFriction=self.friction,
            spinningFriction=self.friction,
            rollingFriction=0.014,
            physicsClientId=0,
            restitution=0.35,
            contactStiffness=6500,
            contactDamping=0.05
        )

        self.present_state = self.initial_action
        self.torques = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        self.set_initial_stance(self.robot_id, self.leg_valid, list(self.initial_action))
        # p.setJointMotorControlArray(self.robot_id, range(p.getNumJoints(self.robot_id)))
        num_timesteps = 20
        for i in range(num_timesteps):
            p.stepSimulation()
            # time.sleep(0.01)

        states = self.get_obs()
        # obs = self.get_obs()

        present = states[36:48]
        # print("present ANGLES", present)
        vels_present = states[48:60]

        self.action_tminus1 = [0.0 for i in range(12)]
        self.action = [0.0 for i in range(12)]
        self.clipped_action = self.initial_action

        self.present_state = present
        self.tminus1_state = self.present_state

        self.tminus1_vels = vels_present

        self.timesteps = 0
        self.sim_steps = 0

        return np.array(states)

    def get_obs(self):
        dirs = self.calculate_desired_direction_turning_direction()
        twist = [dirs[-1]]  # discrete
        lin_dir = dirs[:-1]
        base_angular_vel = self.get_base_angular_vels()
        base_linear_vel = self.get_base_linear_vels()
        joint_pose = self.get_joint_positions()
        joint_vels = self.get_joint_velocity()
        pose_history = self.get_joint_pose_history()
        vels_history = self.get_joint_vels_history()
        action_history = self.action_tminus1

        terrain_normal = self.get_terrain_normal()
        terrain_height = self.get_height_scan()
        foot_contact_forces = self.get_foot_contact_forces()
        foot_contact_states = self.find_binary_foot_contact()  # discrete
        thigh_contact_states = self.get_thigh_contact_states()  # discrete
        shank_contact_states = self.get_shank_contact()  # discrete
        ground_fric = self.get_ground_friction_coeff()
        # print("ground_fric=", ground_fric[0])
        ext_force = self.get_ext_force()
        # print("ext force=", ext_force)

        observation = np.array(
            action_history + pose_history + vels_history + joint_pose + joint_vels + base_angular_vel + base_linear_vel + lin_dir + twist + foot_contact_states + \
            terrain_height + terrain_normal + ground_fric + ext_force + foot_contact_forces
        )  # print("len obs", len(observation))
        # print(observation)
        return observation

    def set_initial_stance(self, robot_id, joint_indices, initial_positions):
        # Set the initial position for each joint
        for joint_index, position in zip(joint_indices, initial_positions):
            p.resetJointState(robot_id, joint_index, position)

        # print("set initial stance")

    def find_binary_foot_contact(self):
        binary_foot_contact = [0 for i in range(4)]
        contact_points = list(p.getContactPoints(self.robot_id))
        for i in range(len(contact_points)):
            point = contact_points[i]
            if point[3] == 3:
                binary_foot_contact[0] = 1

            elif point[3] == 7:
                binary_foot_contact[1] = 1

            elif point[3] == 11:
                binary_foot_contact[2] = 1

            elif point[3] == 15:
                binary_foot_contact[3] = 1

        # print("binary foot contact:", binary_foot_contact)

        return binary_foot_contact

    def calculate_desired_direction_turning_direction(self, ):
        horizontal_twist_dirs = []
        if self.counter >= 30000000 and self.counter != 0:
            if self.counter < 35000000:
                if self.counter % 1000000 == 0:
                    self.twist_dir = random.choice([-1, 0, 1])
                    if self.twist_dir == 0:
                        self.angle = random.uniform(0, 3.14 / 4)

                    else:
                        self.angle = 0

            elif 35000000 <= self.counter < 40000000:
                if self.counter % 100000 == 0:
                    self.twist_dir = random.choice([-1, 0, 1])
                    if self.twist_dir == 0:
                        self.angle = random.uniform(0, 3.14 / 4)

                    else:
                        self.angle = 0

                    self.angle = random.uniform(0, 3.14 / 4)

            elif 40000000 <= self.counter <= 45000000:
                if self.counter % 10000 == 0:
                    self.twist_dir = random.choice([-1, 0, 1])
                    if self.twist_dir == 0:
                        self.angle = random.uniform(0, 3.14 / 4)

                    else:
                        self.angle = 0

                    self.angle = random.uniform(0, 3.14 / 4)

            elif 45000000 <= self.counter <= 50000000:
                if self.counter % 1000 == 0:
                    self.twist_dir = random.choice([-1, 0, 1])
                    if self.twist_dir == 0:
                        self.angle = random.uniform(0, 3.14 / 4)

                    else:
                        self.angle = 0

                    self.angle = random.uniform(0, 3.14 / 4)

            else:
                if self.counter % 100 == 0:
                    self.twist_dir = random.choice([-1, 0, 1])
                    if self.twist_dir == 0:
                        self.angle = random.uniform(0, 3.14 / 4)

                    else:
                        self.angle = 0

                    self.angle = random.uniform(0, 3.14 / 4)

        horizontal_twist_dirs.extend([math.cos(0), math.sin(0)])
        horizontal_twist_dirs.append(self.twist_dir)

        return horizontal_twist_dirs

    def get_base_angular_vels(self):
        vels = p.getBaseVelocity(self.robot_id)
        angular = list(vels[1])

        return angular

    def get_base_linear_vels(self):
        vels = p.getBaseVelocity(self.robot_id)
        linear = list(vels[0])
        # print("linear vels", linear)

        return linear

    def get_joint_positions(self):
        pose = []
        for i in self.leg_valid:
            pose.append(p.getJointState(self.robot_id, i)[0])

        return pose

    def get_joint_velocity(self):
        states = []
        for i in self.leg_valid:
            states.append(p.getJointState(self.robot_id, i)[1])

        return states

    # def get_foot_locations(self):
    #     foot_indices = [3, 7, 11, 15]
    #     locs = []
    #     for ind in foot_indices:
    #         loc = list(self.link_position_in_base_frame(ind))
    #         locs.extend(loc)
    #
    #     return locs

    def get_joint_pose_history(self):
        poses = []
        poses.extend(list(self.tminus1_state))

        return poses

    def get_joint_vels_history(self):
        vels = []
        vels.extend(list(self.tminus1_vels))

        return vels

    def get_terrain_normal(self):
        contact_info = p.getContactPoints(self.robot_id)
        d = {}
        normals = []
        valid_foot = [3, 7, 11, 15]
        for f in valid_foot:
            d[f] = ()

        for contact in contact_info:
            d[contact[3]] = contact[7]

        for v in valid_foot:
            if d[v] == ():
                normals.extend([0.0, 0.0, 0.0])
            else:
                normals.extend(list(d[v]))

        return normals

    def get_foot_pose(self):
        index_pos = {}
        foot_indices = [3, 7, 11, 15]
        contact_states = p.getContactPoints(self.robot_id)
        for c in foot_indices:
            # print(index_pos)
            index_pos[c] = ()

        for c in contact_states:
            index_pos[c[3]] = c[5]

        tar = []
        for ind in foot_indices:
            if index_pos[ind] == ():
                tar.append([0.0, 0.0, 0.0])
            else:
                tar.append(list(index_pos[ind]))

        return tar

    def get_height_scan(self):
        tar = self.get_foot_pose()
        [foot0_pose, foot1_pose, foot2_pose, foot3_pose] = tar
        foot_poses = [foot0_pose, foot1_pose, foot2_pose, foot3_pose]

        for i, foot in enumerate(foot_poses):
            foot_poses[i] = self.transform_foot_poses(foot)

        r = 0.1
        angles = [random.uniform(-math.pi, math.pi) for i in range(8)]
        x = []
        y = []
        for a in angles:
            x.append(r * math.cos(a))
            y.append(r * math.sin(a))

        coords = []

        for foot in foot_poses:
            for i in range(len(x)):
                foot = [foot[0], foot[1]]
                coords.append(foot + [x[i], y[i]])

        heights = []
        for coord in coords:
            (x1, y1) = (coord[0], coord[1])
            (i, j) = (int(128 + x1 / self.terrain_scale), int(128 + y1 / self.terrain_scale))
            # print(i,j)
            heights.append(self.heights[i][j])
        # print("len heights", len(heights))

        return heights

    def transform_foot_poses(self, foot_pose):
        footx = foot_pose[0]
        footy = foot_pose[1]

        return [footy * (-1), footx]

    def get_foot_contact_forces(self):
        contact_info = p.getContactPoints(self.robot_id)
        # print("contact info", contact_info)
        d = {}
        forces = []
        valid_foot = [3, 7, 11, 15]
        for f in valid_foot:
            d[f] = 0.0

        for contact in list(contact_info):
            d[contact[3]] = contact[9]

        for f in valid_foot:
            forces.append(d[f])

        return forces

    def get_thigh_contact_states(self):
        binary_thigh_contact = [0 for i in range(4)]
        contact_points = list(p.getContactPoints(self.robot_id))
        for i in range(len(contact_points)):
            point = contact_points[i]
            if point[3] == 2:
                binary_thigh_contact[0] = 1

            elif point[3] == 6:
                binary_thigh_contact[1] = 1

            elif point[3] == 10:
                binary_thigh_contact[2] = 1

            elif point[3] == 14:
                binary_thigh_contact[3] = 1

        # print("binary foot contact:", binary_thigh_contact)
        return binary_thigh_contact

    def get_shank_contact(self):
        binary_shank_contact = [0 for i in range(4)]
        contact_points = list(p.getContactPoints(self.robot_id))
        for i in range(len(contact_points)):
            point = contact_points[i]
            if point[3] == 1:
                binary_shank_contact[0] = 1

            elif point[3] == 5:
                binary_shank_contact[1] = 1

            elif point[3] == 9:
                binary_shank_contact[2] = 1

            elif point[3] == 13:
                binary_shank_contact[3] = 1

        # print("binary foot contact:", binary_thigh_contact)
        return binary_shank_contact

    def get_ground_friction_coeff(self):
        if self.counter < 6000000:
            self.friction = 0.7

        elif 6000000 <= self.counter < 10000000:
            if self.counter % 1000000 == 0:
                self.friction = random.uniform(0.65, 0.75)
            p.changeDynamics(self.robot_id,
                             -1,
                             lateralFriction=self.friction,
                             spinningFriction=self.friction
                             )
            p.changeDynamics(self.terrain_body,
                             -1,
                             lateralFriction=self.friction,
                             spinningFriction=self.friction
                             )

        elif 10000000 <= self.counter < 15000000:
            if self.counter % 1000000 == 0:
                self.friction = random.uniform(0.55, 0.75)
                p.changeDynamics(self.robot_id,
                                 -1,
                                 lateralFriction=self.friction,
                                 spinningFriction=self.friction
                                 )
                p.changeDynamics(self.terrain_body,
                                 -1,
                                 lateralFriction=self.friction,
                                 spinningFriction=self.friction
                                 )
        elif 15000000 <= self.counter < 20000000:
            if self.counter % 1000000 == 0:
                self.friction = random.uniform(0.45, 0.8)
                p.changeDynamics(self.robot_id,
                                 -1,
                                 lateralFriction=self.friction,
                                 spinningFriction=self.friction
                                 )
                p.changeDynamics(self.terrain_body,
                                 -1,
                                 lateralFriction=self.friction,
                                 spinningFriction=self.friction
                                 )

        else:
            if self.counter % 200000 == 0:
                self.friction = random.uniform(0.25, 0.8)
                p.changeDynamics(self.robot_id,
                                 -1,
                                 lateralFriction=self.friction,
                                 spinningFriction=self.friction
                                 )
                p.changeDynamics(self.terrain_body,
                                 -1,
                                 lateralFriction=self.friction,
                                 spinningFriction=self.friction
                                 )

        fric_coeff = [self.friction for i in range(4)]

        return fric_coeff

    def get_ext_force(self):
        if self.counter < int(3e6):
            self.fx = 0
            self.fy = 0
            self.fz = 0

        elif int(3e6) <= self.counter < int(6e6):
            if self.counter % 100 == 0:
                self.fx = random.uniform(-5, 5)
                self.fy = random.uniform(-5, 5)
                self.fz = random.uniform(-5, 5)
                force = [self.fx, self.fy, self.fz]
                position = [0, 0, 0]  # Specify the position where the force is applied [X, Y, Z]
                link_index = -1  # Specify the link index (-1 for base/root link)
                p.applyExternalForce(self.robot_id, link_index, force, position, flags=p.WORLD_FRAME)

            else:
                self.fx = 0
                self.fy = 0
                self.fz = 0

        elif int(6e6) <= self.counter < int(10e6):
            if self.counter % 100 == 0:
                self.fx = random.uniform(-10, 10)
                self.fy = random.uniform(-10, 10)
                self.fz = random.uniform(-10, 10)
                force = [self.fx, self.fy, self.fz]
                position = [0, 0, 0]  # Specify the position where the force is applied [X, Y, Z]
                link_index = -1  # Specify the link index (-1 for base/root link)
                p.applyExternalForce(self.robot_id, link_index, force, position, flags=p.WORLD_FRAME)
            else:
                self.fx = 0
                self.fy = 0
                self.fz = 0

        elif int(10e6) <= self.counter < int(15e6):
            if self.counter % 100 == 0:
                self.fx = random.uniform(-20, 20)
                self.fy = random.uniform(-20, 20)
                self.fz = random.uniform(-20, 20)
                force = [self.fx, self.fy, self.fz]
                position = [0, 0, 0]  # Specify the position where the force is applied [X, Y, Z]
                link_index = -1  # Specify the link index (-1 for base/root link)
                p.applyExternalForce(self.robot_id, link_index, force, position, flags=p.WORLD_FRAME)

            else:
                self.fx = 0
                self.fy = 0
                self.fz = 0

        elif int(15e6) <= self.counter < int(20e6):
            if self.counter % 100 == 0:
                self.fx = random.uniform(-40, 40)
                self.fy = random.uniform(-40, 40)
                self.fz = random.uniform(-40, 40)
                force = [self.fx, self.fy, self.fz]
                position = [0, 0, 0]  # Specify the position where the force is applied [X, Y, Z]
                link_index = -1  # Specify the link index (-1 for base/root link)
                p.applyExternalForce(self.robot_id, link_index, force, position, flags=p.WORLD_FRAME)

            else:
                self.fx = 0
                self.fy = 0
                self.fz = 0

        elif int(20e6) <= self.counter < int(25e6):
            if self.counter % 100 == 0:
                self.fx = random.uniform(-60, 60)
                self.fy = random.uniform(-60, 60)
                self.fz = random.uniform(-60, 60)
                force = [self.fx, self.fy, self.fz]
                position = [0, 0, 0]  # Specify the position where the force is applied [X, Y, Z]
                link_index = -1  # Specify the link index (-1 for base/root link)
                p.applyExternalForce(self.robot_id, link_index, force, position, flags=p.WORLD_FRAME)

            else:
                self.fx = 0
                self.fy = 0
                self.fz = 0


        else:
            if self.counter % 100 == 0:
                self.fx = random.uniform(-80, 80)
                self.fy = random.uniform(-80, 80)
                self.fz = random.uniform(-80, 80)
                force = [self.fx, self.fy, self.fz]
                position = [0, 0, 0]  # Specify the position where the force is applied [X, Y, Z]
                link_index = -1  # Specify the link index (-1 for base/root link)
                p.applyExternalForce(self.robot_id, link_index, force, position, flags=p.WORLD_FRAME)

            else:
                self.fx = 0
                self.fy = 0
                self.fz = 0

        force = [self.fx, self.fy, self.fz]

        return force


    def calculate_reward(self):

        # smooth_lim = 0.025
        ener_lim = 0.002
        z_vel_lim = 0.5
        action_rate_lim = 0.05
        lateral_lim = 4
        vx, vy, vz = p.getBaseVelocity(self.robot_id)[0]
        wz = self.get_base_angular_vels()[-1]

        if self.counter % 800000 * num_env == 0 and self.counter != 0:
            # smooth_coeff = min(smooth_coeff * 2, smooth_lim)
            self.ener_coeff = min(self.ener_coeff * 2, ener_lim)
            self.z_vel_coeff = min(self.z_vel_coeff * 2, z_vel_lim)
            self.action_rate_coeff = min(self.action_rate_coeff * 2, action_rate_lim)
            self.lateral_coeff = min(self.lateral_coeff * 2, lateral_lim)

        torques = []
        for i in self.leg_valid:
            joint_state = p.getJointState(self.robot_id, i)
            torques.append(joint_state[3])
        torques = np.array(torques)

        vels = np.array(self.get_joint_velocity())
        ener = np.multiply(vels, torques)
        energy_reward = abs(sum(list(ener))) * (-1)

        self.vx = self.desired_speed * math.cos(self.angle)
        self.vy = self.desired_speed * math.sin(self.angle)
        # forward_vel_reward = self.vx - abs(vx - self.vx_target) - abs(self.desired_twisting_speed * self.twist_dir - wz)
        forward_vel_reward = min(self.vx, vx)
        lateral_reward = (pow(self.vy - vy, 2) + pow(self.desired_twisting_speed * self.twist_dir - wz, 2)) * (-1)

        # smooth_reward_list = np.array(self.leg0_tar_t) - 2 * np.array(self.leg0_tar_tminus1) + np.array(
        #     self.leg0_tar_tminus2) + np.array(self.leg1_tar_t) - 2 * np.array(self.leg1_tar_tminus1) + np.array(
        #     self.leg1_tar_tminus2) + np.array(self.leg2_tar_t) - 2 * np.array(self.leg2_tar_tminus1) + np.array(
        #     self.leg2_tar_tminus2) + np.array(self.leg3_tar_t) - 2 * np.array(self.leg3_tar_tminus1) + np.array(
        #     self.leg3_tar_tminus2)
        # smooth_reward = (-1) * math.sqrt(
        #     pow(smooth_reward_list[0], 2) + pow(smooth_reward_list[1], 2) + pow(smooth_reward_list[2], 2))
        z_vel_reward = (-1) * pow(vz, 2)

        action_rate_list = np.array(self.action) - np.array(self.action_tminus1)

        action_rate_reward = (-1) * math.sqrt(sum([pow(action_rate_list[i], 2) for i in range(12)]))

        # print("line vel", forward_vel_reward * lin_vel_coeff)
        # print("lateral", lateral_reward * lateral_lim)
        # print("action rate", action_rate_reward * action_rate_lim)
        # # print("smoothness", smooth_reward)
        # print("z vel", z_vel_reward * z_vel_lim)
        # print("energy", energy_reward * ener_lim)

        reward = energy_reward * self.ener_coeff + forward_vel_reward * self.lin_vel_coeff + lateral_reward * self.lateral_coeff + \
                 z_vel_reward * self.z_vel_coeff + action_rate_reward * self.action_rate_coeff + self.survival_bonus

        # print("reward= ", reward)
        # print("Energy Reward = ", energy_reward * ener_lim)

        info = {}

        info['energy_reward'] = energy_reward * self.ener_coeff
        info['vel_reward'] = self.lin_vel_coeff * forward_vel_reward
        info['lateral_vel_reward'] = lateral_reward * self.lateral_coeff
        info['z-vel_reward'] = z_vel_reward * self.z_vel_coeff
        # info['smooth_reward'] = smooth_reward * smooth_coeff
        info['action_rate_reward'] = action_rate_reward * self.action_rate_coeff
        info['survival_reward'] = self.survival_bonus

        return reward, info


    def _is_done(self):
        base_pose, body_orien = p.getBasePositionAndOrientation(self.robot_id)
        # print("base pose", base_pose[2])
        if self.timesteps == self.max_episode_steps:
            # print("1000 ts")
            self.done = True
            # print("pose = ", base_pose)
            # self.data = math.sqrt(math.pow(base_pose[0], 2) + math.pow(base_pose[1], 2))

        if abs(body_orien[0]) > 0.4:
            # print("roll_done")
            self.done = True

        if abs(body_orien[1]) > 0.2:
            # print("pitch done")
            self.done = True

        if base_pose[2] < 0.327:
            # print("height done")
            self.done = True

        else:
            pass

        done = self.done
        return done

    def close(self):
        # Disconnecting from the physics server
        p.disconnect()


if __name__ == '__main__':
    qp = QuadrupedRobotEnv()
    qp.reset()
    i = 1
    while True:
        action = qp.action_space.sample()
        # action = [0, 0.4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        # a = time.time()
        qp.step(action)
        # b = time.time()
        # print(a-b)
        # print(action)
