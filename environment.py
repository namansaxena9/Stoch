import math
import os
import random
import time

import gym
import numpy
from gym import spaces
import pybullet as p
import numpy as np

BASEDIR = "/Users/navakrish"
URDFPATH = os.path.join(BASEDIR, "convex_mpc", "mpc_controller", "stoch3_description", "urdf", "stoch3.urdf")
# print(URDFPATH)
SEARCHPATH = os.path.join(BASEDIR, "convex_mpc", "mpc_controller", "stoch3_description", "meshes")
BULLETPATH = os.path.join(BASEDIR, "convex_mpc", "mpc_controller", "stoch3_description", "urdf", "stoch3_env2.bullet")


class QuadrupedRobotEnv(gym.Env):
    def __init__(self, max_episode_steps=1000):

        self.physics_client = p.connect(p.DIRECT)
        # self.physics_client = p.connect(p.GUI)

        self.max_episode_steps = max_episode_steps

        self._motor_offset = np.array([0] * 12)
        # self._motor_offset = np.array([0, -0.8, -1.5, 0, 0.8, -1.5, 0, -0.8, -1.5, 0, 0.8, -1.5])

        self._motor_direction = np.array([1, 1, 1,
                                          1, 1, 1,
                                          1, 1, 1,
                                          1, 1, 1])

        # Loading the URDF file with meshes
        p.setAdditionalSearchPath(SEARCHPATH)  # Path to the meshes folder
        self.robot_id = p.loadURDF(URDFPATH, [0, 0, 0.58])
        self.NUM_LEGS = 4

        self.time_interval = 0.01
        self.two_pi = 2 * math.pi

        # Definining terrain parameters
        self.terrain_size = 200
        self.terrain_scale = 4

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
        p.changeVisualShape(self.terrain_body, -1, rgbaColor=[0.529, 0.808, 0.922, 1])

        # Setting the position of the terrain to be at the center of the world
        p.resetBasePositionAndOrientation(self.terrain_body, [0, 0, 0], [0, 0, 0, 1])
        p.resetBasePositionAndOrientation(self.robot_id, [0, 0, 0.69], [0, 0, 0, 1])

        p.setGravity(0, 0, -9.81)
        # p.setPhysicsEngineParameter(contactStiffness=100000.0)
        p.setTimeStep(self.time_interval, self.physics_client)
        p.changeDynamics(
            self.robot_id,
            -1,
            lateralFriction=0.8,
            spinningFriction=0.8,
            rollingFriction=0.05,
            physicsClientId=0,
            restitution=0.15,
            contactStiffness=6500,
            contactDamping=0.05
        )
        p.changeDynamics(
            self.terrain_body,
            -1,
            lateralFriction=0.8,
            spinningFriction=0.8,
            rollingFriction=0.05,
            physicsClientId=0,
        )

        self.state_id = p.saveState()
        p.saveBullet(BULLETPATH)
        # num_joints = p.getNumJoints(self.robot_id)
        self.base_freq = 2.5

        self.desired_speed = 0.4
        self.desired_twisting_speed = 0.4

        self.twist_dir = 0
        self.angle = 0
        self.vx = self.desired_speed * math.cos(self.angle)
        self.vy = self.desired_speed * math.sin(self.angle)

        # self.valid_indices = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
        self.leg_valid = [0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14]
        self.leg_indices = [0, 1, 2, 3]
        # print(p.getContactPoints(self.robot_id))
        # print(p.getLinkState(self.robot_id, 2))
        self.phase_leg0 = 0.0
        self.phase_leg1 = math.pi
        self.phase_leg2 = math.pi
        self.phase_leg3 = 0.0

        self.phase_offset0 = self.phase_leg0
        self.phase_offset1 = self.phase_leg1
        self.phase_offset2 = self.phase_leg2
        self.phase_offset3 = self.phase_leg3

        self.freq_leg0 = self.base_freq
        self.freq_leg1 = self.base_freq
        self.freq_leg2 = self.base_freq
        self.freq_leg3 = self.base_freq

        self.leg0_tar_t = [0.0, 0.0, 0.0]
        self.leg1_tar_t = [0.0, 0.0, 0.0]
        self.leg2_tar_t = [0.0, 0.0, 0.0]
        self.leg3_tar_t = [0.0, 0.0, 0.0]
        self.leg0_tar_tminus1 = [0.0, 0.0, 0.0]
        self.leg0_tar_tminus2 = [0.0, 0.0, 0.0]
        self.leg1_tar_tminus1 = [0.0, 0.0, 0.0]
        self.leg1_tar_tminus2 = [0.0, 0.0, 0.0]
        self.leg2_tar_tminus1 = [0.0, 0.0, 0.0]
        self.leg2_tar_tminus2 = [0.0, 0.0, 0.0]
        self.leg3_tar_tminus1 = [0.0, 0.0, 0.0]
        self.leg3_tar_tminus2 = [0.0, 0.0, 0.0]

        self.action_factor = 0.01
        self.frequency_factor = 0.5

        self.action_space = spaces.Box(low=-1, high=1, shape=(16,))

        self.obs_lower_limits = np.array(
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
        self.obs_upper_limits = np.array(
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
        # print("len low", len(self.obs_lower_limits))

        self.observation_space = spaces.Box(low=self.obs_lower_limits, high=self.obs_upper_limits, shape=(201,),
                                            dtype=np.float32)

        angle_1 = 0.007
        angle_2 = 1.035
        angle_3 = 1.6  # 1.794
        self.initial_action = np.array(
            [-angle_1, angle_2, -angle_3, angle_1, angle_2, -angle_3, -angle_1, angle_2, -angle_3, angle_1, angle_2,
             -angle_3])

        self.present_state = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        self.present_vels = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])

        self.tminus1_state = self.present_state
        self.tminus2_state = self.present_state

        self.tminus1_vels = self.present_vels
        self.tminus2_vels = self.present_vels

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
        # self.prev_torq = [0.0 for i in range(12)]
        # self.reward_sum = 0.0
        # self.epi_count = 0
        self.done = False

        foot_clearance = 0.01
        height = 0.45
        self.desired_height = height - foot_clearance
        # self.data = 0.0

    def step(self, action):
        action = list(action)
        freq_offsets = action[:4]
        for i, f in enumerate(freq_offsets):
            freq_offsets[i] = f * self.frequency_factor

        leg_tars = action[4:]
        for i, f in enumerate(leg_tars):
            leg_tars[i] = f * self.action_factor

        self.freq_leg0 = self.base_freq + freq_offsets[0]
        self.freq_leg1 = self.base_freq + freq_offsets[1]
        self.freq_leg2 = self.base_freq + freq_offsets[2]
        self.freq_leg3 = self.base_freq + freq_offsets[3]

        self.phase_leg0 = (self.phase_offset0 + self.freq_leg0 * self.time_interval * self.two_pi * self.timesteps) % (
                2 * math.pi)
        self.phase_leg1 = (self.phase_offset1 + self.freq_leg1 * self.time_interval * self.two_pi * self.timesteps) % (
                2 * math.pi)
        self.phase_leg2 = (self.phase_offset2 + self.freq_leg2 * self.time_interval * self.two_pi * self.timesteps) % (
                2 * math.pi)
        self.phase_leg3 = (self.phase_offset3 + self.freq_leg3 * self.time_interval * self.two_pi * self.timesteps) % (
                2 * math.pi)

        ftg_z0 = self.FTG(self.phase_leg0)
        ftg_z1 = self.FTG(self.phase_leg1)
        ftg_z2 = self.FTG(self.phase_leg2)
        ftg_z3 = self.FTG(self.phase_leg3)

        leg0_tar = list(np.add(ftg_z0, np.array(leg_tars[:3])))
        leg1_tar = list(np.add(ftg_z1, np.array(leg_tars[3:6])))
        leg2_tar = list(np.add(ftg_z2, np.array(leg_tars[6:9])))
        leg3_tar = list(np.add(ftg_z3, np.array(leg_tars[9:])))

        self.leg0_tar_tminus2 = self.leg0_tar_tminus1
        self.leg0_tar_tminus1 = self.leg0_tar_t
        self.leg0_tar_t = list(np.add(np.array(self.leg0_tar_tminus1[:2] + [0.0]), np.array(leg0_tar)))

        self.leg1_tar_tminus2 = self.leg1_tar_tminus1
        self.leg1_tar_tminus1 = self.leg1_tar_t
        self.leg1_tar_t = list(np.add(np.array(self.leg1_tar_tminus1[:2] + [0.0]), np.array(leg1_tar)))

        self.leg2_tar_tminus2 = self.leg2_tar_tminus1
        self.leg2_tar_tminus1 = self.leg2_tar_t
        self.leg2_tar_t = list(np.add(np.array(self.leg2_tar_tminus1[:2] + [0.0]), np.array(leg2_tar)))

        self.leg3_tar_tminus2 = self.leg3_tar_tminus1
        self.leg3_tar_tminus1 = self.leg3_tar_t
        self.leg3_tar_t = list(np.add(np.array(self.leg3_tar_tminus1[:2] + [0.0]), np.array(leg3_tar)))

        joint_ids_0, joint_angles_0 = (
            self.ComputeMotorAnglesFromFootLocalPosition(
                0, tuple(self.leg0_tar_t)))

        joint_ids_1, joint_angles_1 = (
            self.ComputeMotorAnglesFromFootLocalPosition(
                1, tuple(self.leg1_tar_t)))

        joint_ids_2, joint_angles_2 = (
            self.ComputeMotorAnglesFromFootLocalPosition(
                2, tuple(self.leg2_tar_t)))

        joint_ids_3, joint_angles_3 = (
            self.ComputeMotorAnglesFromFootLocalPosition(
                3, tuple(self.leg3_tar_t)))

        joint_ids = []
        joint_angles = []

        joint_ids.extend(list(joint_ids_0))
        joint_ids.extend(list(joint_ids_1))
        joint_ids.extend(list(joint_ids_2))
        joint_ids.extend(list(joint_ids_3))
        joint_angles.extend(list(joint_angles_0))
        joint_angles.extend(list(joint_angles_1))
        joint_angles.extend(list(joint_angles_2))
        joint_angles.extend(list(joint_angles_3))

        d = {}
        for i, id in enumerate(joint_ids):
            d[id] = joint_angles[i]

        for joint_index in self.leg_valid:
            p.setJointMotorControl2(self.robot_id, joint_index, controlMode=p.POSITION_CONTROL,
                                    targetPosition=d[joint_index], positionGain=1.0, force=50)

        # Step the simulation forward by one time step
        p.stepSimulation()
        self.timesteps += 1
        self.counter += 1

        obs = self.get_obs()
        self.tminus2_state = self.tminus1_state
        self.tminus1_state = self.present_state
        self.present_state = obs[9:21]

        self.tminus2_vels = self.tminus1_vels
        self.tminus1_vels = self.present_vels
        self.present_vels = obs[21:33]

        reward, info = self.calculate_reward()
        done = self._is_done()
        # print("timesteps", self.timesteps)

        return obs, reward, done, info

    def reset(self):
        # Reset the simulation and the robot's state
        # print("reset")
        p.resetSimulation()

        cameraTargetPosition = [0, 0, 0]  # Example target position [x, y, z]
        p.resetDebugVisualizerCamera(cameraDistance=3, cameraYaw=0, cameraPitch=-30,
                                     cameraTargetPosition=cameraTargetPosition)

        # self.timesteps = 0
        self.done = False
        p.setAdditionalSearchPath(SEARCHPATH)  # Path to the meshes folder
        self.robot_id = p.loadURDF(URDFPATH, [0, 0, 0.58])
        self.terrain_size = 200
        self.terrain_scale = 1

        self.heights = np.random.uniform(0, 0, size=(self.terrain_size, self.terrain_size))
        # heights[150][100] = 2
        # print("heights", heights[100][100])
        # self.get_height_scan()

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
        p.changeVisualShape(self.terrain_body, -1, rgbaColor=[0.529, 0.808, 0.922, 1])

        # Setting the position of the terrain to be at the center of the world
        p.resetBasePositionAndOrientation(self.terrain_body, [0, 0, 0], [0, 0, 0, 1])
        p.resetBasePositionAndOrientation(self.robot_id, [0, 0, 0.65], [0, 0, 0, 1])

        p.setGravity(0, 0, -9.81)
        # p.setPhysicsEngineParameter(contactStiffness=100000.0)
        p.setTimeStep(self.time_interval, self.physics_client)
        p.changeDynamics(
            self.robot_id,
            -1,
            lateralFriction=0.8,
            spinningFriction=0.8,
            rollingFriction=0.1,
            physicsClientId=0,
            restitution=0.15,
            contactStiffness=6500,
            contactDamping=0.05
        )
        p.changeDynamics(
            self.terrain_body,
            -1,
            lateralFriction=0.8,
            spinningFriction=0.8,
            rollingFriction=0.1,
            physicsClientId=0,
        )

        self.present_state = self.initial_action
        self.torques = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        self.set_initial_stance(self.robot_id, self.leg_valid, list(self.initial_action))
        # p.setJointMotorControlArray(self.robot_id, range(p.getNumJoints(self.robot_id)))
        num_timesteps = 20
        for i in range(num_timesteps):
            p.stepSimulation()
            time.sleep(0.02)

        states = self.get_obs()
        # obs = self.get_obs()

        present = states[9:21]
        # print("present ANGLES", present)
        vels_present = states[21:33]

        self.present_state = present
        self.tminus1_state = self.present_state
        self.tminus2_state = self.present_state

        self.tminus1_vels = vels_present
        self.tminus2_vels = vels_present

        self.phase_leg0 = math.pi
        self.phase_leg1 = 0.0
        self.phase_leg2 = 0.0
        self.phase_leg3 = math.pi

        self.phase_offset0 = self.phase_leg0
        self.phase_offset1 = self.phase_leg1
        self.phase_offset2 = self.phase_leg2
        self.phase_offset3 = self.phase_leg3

        self.freq_leg0 = self.base_freq
        self.freq_leg1 = self.base_freq
        self.freq_leg2 = self.base_freq
        self.freq_leg3 = self.base_freq

        leg0_tar = list(self.link_position_in_base_frame(3))
        leg1_tar = list(self.link_position_in_base_frame(7))
        leg2_tar = list(self.link_position_in_base_frame(11))
        leg3_tar = list(self.link_position_in_base_frame(15))

        # print("leg0 pose in world frame", p.getLinkState(self.robot_id, 3)[0])
        # print("leg0, leg1, leg2, leg3", leg0_tar, leg1_tar, leg2_tar, leg3_tar)

        self.leg0_tar_t = leg0_tar
        self.leg1_tar_t = leg1_tar
        self.leg2_tar_t = leg2_tar
        self.leg3_tar_t = leg3_tar
        self.leg0_tar_tminus1 = leg0_tar
        self.leg0_tar_tminus2 = leg0_tar
        self.leg1_tar_tminus1 = leg1_tar
        self.leg1_tar_tminus2 = leg1_tar
        self.leg2_tar_tminus1 = leg2_tar
        self.leg2_tar_tminus2 = leg2_tar
        self.leg3_tar_tminus1 = leg3_tar
        self.leg3_tar_tminus2 = leg3_tar

        self.timesteps = 0

        return np.array(states)

    def ComputeMotorAnglesFromFootLocalPosition(self, leg_id,
                                                foot_local_position):
        """Use IK to compute the motor angles, given the foot link's local position.

        Args:
          leg_id: The leg index.
          foot_local_position: The foot link's position in the base frame.

        Returns:
          A tuple. The position indices and the angles for all joints along the
          leg. The position indices is consistent with the joint orders as returned
          by GetMotorAngles API.
        """
        return self._EndEffectorIK(
            leg_id, foot_local_position, position_in_world_frame=False)

    def joint_angles_from_link_position(
            self,
            leg_id,
            link_position,
            link_id,
            joint_ids,
            position_in_world_frame,
            base_translation=(0, 0, 0),
            base_rotation=(0, 0, 0, 1)):
        """Uses Inverse Kinematics to calculate joint angles.

        Args:
          robot: A robot instance.
          link_position: The (x, y, z) of the link in the body or the world frame,
            depending on whether the argument position_in_world_frame is true.
          link_id: The link id as returned from loadURDF.
          joint_ids: The positional index of the joints. This can be different from
            the joint unique ids.
          position_in_world_frame: Whether the input link_position is specified
            in the world frame or the robot's base frame.
          base_translation: Additional base translation.
          base_rotation: Additional base rotation.

        Returns:
          A list of joint angles.
        """
        # print("joint_ids", joint_ids)
        _IDENTITY_ORIENTATION = [0, 0, 0, 1]
        if not position_in_world_frame:
            # Projects to local frame.
            base_position, base_orientation = p.getBasePositionAndOrientation(
                self.robot_id)  # robot.GetBasePosition(), robot.GetBaseOrientation()
            base_position, base_orientation = p.multiplyTransforms(
                base_position, base_orientation, base_translation, base_rotation)

            # Projects to world space.
            world_link_pos, _ = p.multiplyTransforms(
                base_position, base_orientation, link_position, _IDENTITY_ORIENTATION)

            # print("world pose", world_link_pos)
        else:
            world_link_pos = link_position

        ik_solver = 0
        all_joint_angles = p.calculateInverseKinematics(
            self.robot_id, link_id, world_link_pos, solver=ik_solver)
        # print("all_joint_angles", all_joint_angles)
        # print("leg index", leg_id)
        # print("len of all joint angles", len(all_joint_angles))
        # Extract the relevant joint angles.
        if leg_id == 0:
            joint_angles = [all_joint_angles[i] for i in joint_ids]
        elif leg_id == 1:
            joint_angles = [all_joint_angles[i - 1] for i in joint_ids]
        elif leg_id == 2:
            joint_angles = [all_joint_angles[i - 2] for i in joint_ids]
        else:
            joint_angles = [all_joint_angles[i - 3] for i in joint_ids]

        # joint_angles = [all_joint_angles[i] for i in joint_ids]
        # print("joiny angles", joint_angles)

        return joint_angles

    def _EndEffectorIK(self, leg_id, position, position_in_world_frame):
        """Calculate the joint positions from the end effector position."""
        assert len(self._foot_link_ids) == self.num_legs
        toe_id = self._foot_link_ids[leg_id]
        motors_per_leg = self.num_motors // self.num_legs
        joint_position_idxs = [
            i for i in range(leg_id * motors_per_leg + leg_id, leg_id * motors_per_leg +
                             motors_per_leg + leg_id)
        ]
        joint_angles = self.joint_angles_from_link_position(
            leg_id=leg_id,
            link_position=position,
            link_id=toe_id,
            joint_ids=joint_position_idxs,
            position_in_world_frame=position_in_world_frame)
        # Joint offset is necessary for Laikago.
        # print("angles",             np.asarray(joint_angles) -
        #     np.asarray(self._motor_offset)[joint_position_idxs])
        # print("joint ids", joint_position_idxs)
        if leg_id == 0:
            pose = [0, 1, 2]
        elif leg_id == 1:
            pose = [3, 4, 5]
        elif leg_id == 2:
            pose = [6, 7, 8]
        else:
            pose = [9, 10, 11]

        joint_angles = np.multiply(
            np.asarray(joint_angles) -
            np.asarray(self._motor_offset)[pose],
            self._motor_direction[pose])
        # Return the joint index (the same as when calling GetMotorAngles) as well
        # as the angles.
        return joint_position_idxs, joint_angles.tolist()

    def get_obs(self):
        dirs = self.calculate_desired_direction_turning_direction()
        twist = [dirs[-1]]  # discrete
        lin_dir = dirs[:-1]
        base_angular_vel = self.get_base_angular_vels()
        base_linear_vel = self.get_base_linear_vels()
        joint_pose = self.get_joint_positions()
        joint_vels = self.get_joint_velocity()
        ftg_phases = self.get_FTG_phases()
        ftg_freq = self.get_FTG_frequency()
        base_freq = self.get_base_frequency()
        pose_history = self.get_joint_pose_history()
        vels_history = self.get_joint_vels_history()
        foot_tar_history = self.get_foot_target_history()
        foot_locs = self.get_foot_locations()

        terrain_normal = self.get_terrain_normal()
        # print("terrain normal", terrain_normal)
        terrain_height = self.get_height_scan()
        # print("terrain height", terrain_height)
        foot_contact_forces = self.get_foot_contact_forces()
        # print("foot contat forces", foot_contact_forces)
        foot_contact_states = self.find_binary_foot_contact()  # discrete
        thigh_contact_states = self.get_thigh_contact_states()  # discrete
        shank_contact_states = self.get_shank_contact()  # discrete
        ground_fric = self.get_ground_friction_coeff()
        ext_force = self.get_ext_force()

        observation = np.array(
            twist + lin_dir + base_angular_vel + base_linear_vel + joint_pose + joint_vels + foot_locs + ftg_phases + ftg_freq + base_freq + pose_history + vels_history + foot_tar_history + terrain_normal + terrain_height + foot_contact_forces + foot_contact_states + thigh_contact_states + shank_contact_states + ground_fric + ext_force)
        # print("len obs", len(observation))
        return observation

    def FTG(self, phase):
        # print("phase", phase)
        k = 2 * (phase - math.pi) / math.pi
        h = 0.12
        ftg = [0, 0, 0]
        if 0 <= k <= 1:
            ftg[2] = (h * ((-2) * pow(k, 3) + 3 * pow(k, 2)) - self.desired_height)
            # print(self.desired_height)
            # print("ftg[2]", h * ((-2) * pow(k, 3) + 3 * pow(k, 2)))
            # print("inside")

        elif 1 <= k <= 2:
            ftg[2] = (h * (2 * pow(k, 3) - 9 * pow(k, 2) + 12 * k - 4) - self.desired_height)

        else:
            ftg[2] = (-1) * self.desired_height
            # print("z", ftg[2])
            # print((-1) * self.desired_height)

        return np.array(ftg)

    def num_legs(self):
        return self.NUM_LEGS

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
        if self.counter % 6000000 == 0 and self.counter != 0:
            self.twist_dir = random.choice([-1, 0, 1])
            self.angle = random.uniform(0, 3.14 / 2)

        horizontal_twist_dirs.extend([math.cos(self.angle), math.sin(self.angle)])
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

    def get_FTG_phases(self):
        phases = []
        phases.extend([math.sin(self.phase_leg0), math.cos(self.phase_leg0)])
        phases.extend([math.sin(self.phase_leg1), math.cos(self.phase_leg1)])
        phases.extend([math.sin(self.phase_leg2), math.cos(self.phase_leg2)])
        phases.extend([math.sin(self.phase_leg3), math.cos(self.phase_leg3)])

        return phases

    def get_foot_locations(self):
        foot_indices = [3, 7, 11, 15]
        locs = []
        for ind in foot_indices:
            loc = list(self.link_position_in_base_frame(ind))
            locs.extend(loc)

        return locs

    def get_base_frequency(self):

        return [self.base_freq]

    def get_FTG_frequency(self):
        freq = []
        freq.extend([self.freq_leg0, self.freq_leg1, self.freq_leg2, self.freq_leg3])

        return freq

    def get_joint_pose_history(self):
        poses = []
        poses.extend(list(self.tminus1_state))
        poses.extend(list(self.tminus2_state))

        return poses

    def get_joint_vels_history(self):
        vels = []
        vels.extend(list(self.tminus1_vels))
        vels.extend(list(self.tminus2_vels))

        return vels

    def get_foot_target_history(self):
        foot_target = self.leg0_tar_tminus2 + self.leg1_tar_tminus2 + self.leg2_tar_tminus2 + self.leg3_tar_tminus2 + \
                      self.leg0_tar_tminus1 + self.leg1_tar_tminus1 + self.leg2_tar_tminus1 + self.leg3_tar_tminus1

        return foot_target

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
        angles = [random.uniform(-math.pi, math.pi) for i in range(9)]
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
            (i, j) = (int(100 + x1 / self.terrain_scale), int(100 + y1 / self.terrain_scale))
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
        fric_coeff = [0.6, 0.6, 0.6, 0.6]
        if self.counter % 3000000 == 0 and self.counter != 0:
            fric_coeff = [random.uniform(0.1, 1) for i in range(4)]
            p.changeDynamics(self.robot_id,
                             -1,
                             lateralFriction=fric_coeff[0],
                             )
            p.changeDynamics(self.terrain_body,
                             -1,
                             lateralFriction=fric_coeff[0],
                             )

        return fric_coeff

    def get_ext_force(self):
        force = [0.0, 0.0, 0.0]
        if self.counter > 5000000 and self.counter != 0:
            force = [random.uniform(-80, 80) for i in range(3)]  # Specify the force vector [X, Y, Z]
            position = [0, 0, 0]  # Specify the position where the force is applied [X, Y, Z]
            link_index = -1  # Specify the link index (-1 for base/root link)

            if (self.counter % 1000):
                p.applyExternalForce(self.robot_id, link_index, force, position, flags=p.WORLD_FRAME)

        return force

    def link_position_in_base_frame(self, link_id):
        """Computes the link's local position in the robot frame.

        Args:
          robot: A robot instance.
          link_id: The link to calculate its relative position.

        Returns:
          The relative position of the link.
        """
        base_position, base_orientation = p.getBasePositionAndOrientation(self.robot_id)
        inverse_translation, inverse_rotation = p.invertTransform(
            base_position, base_orientation)

        # print("link_id", link_id)
        link_state = p.getLinkState(self.robot_id, link_id)
        link_position = link_state[0]
        # print("link state", link_position)
        link_local_position, _ = p.multiplyTransforms(
            inverse_translation, inverse_rotation, link_position, (0, 0, 0, 1))

        return np.array(link_local_position)

    # def reward_function(self):
    #     global lin_vel_reward, ang_vel_reward
    #     lin_vel_coeff = 1
    #     ang_vel_coeff = 0.001
    #     base_pen_coeff = 0.0001
    #     smooth_coeff = 0.00001
    #     ener_coeff = 0.000001
    #
    #     ang_vel_lim = 1
    #     base_pen_limit = 0.4
    #     smooth_lim = 0.025
    #     ener_lim = 0.0004
    #
    #     if self.counter % 100000 == 0 and self.counter != 1:
    #         base_pen_coeff = min(base_pen_coeff * 2, base_pen_limit)
    #         smooth_coeff = min(smooth_coeff * 2, smooth_lim)
    #         ener_coeff = min(ener_coeff * 2, ener_lim)
    #         ang_vel_coeff = min(ang_vel_coeff * 2, ang_vel_lim)
    #
    #     torques = []
    #     for i in self.leg_valid:
    #         joint_state = p.getJointState(self.robot_id, i)
    #         torques.append(joint_state[3])
    #     torques = np.array(torques)
    #
    #     vels = np.array(self.get_joint_velocity())
    #     ener = np.multiply(vels, torques)
    #     energy_reward = abs(sum(list(ener))) * (-1)
    #
    #     vpr = sum(list(np.multiply(np.array(self.calculate_desired_direction_turning_direction()[:2]),
    #                                np.array(self.get_base_linear_vels()[:2]))))
    #     v_0 = math.sqrt(pow(self.get_base_linear_vels()[:2][0], 2) + pow(self.get_base_linear_vels()[:2][1], 2))
    #     if vpr == 0:
    #         lin_vel_reward = 0.0
    #     if vpr < self.desired_speed:
    #         lin_vel_reward = math.exp((-2) * pow((vpr - self.desired_speed), 2))
    #     elif vpr >= self.desired_speed:
    #         lin_vel_reward = 1.0
    #
    #     wpr = self.twist_dir * self.get_base_angular_vels()[-1]
    #     w_0 = abs(self.get_base_angular_vels()[-1])
    #     if wpr == 0:
    #         ang_vel_reward = math.exp((-1.5) * pow(w_0, 2))
    #
    #     if wpr < self.desired_twisting_speed:
    #         ang_vel_reward = math.exp((-1.5) * pow((wpr - self.desired_twisting_speed), 2))
    #
    #     elif wpr > self.desired_twisting_speed:
    #         ang_vel_reward = 1.0
    #
    #     # print("lin vel reward", lin_vel_reward * lin_vel_coeff)
    #     v0 = np.array(self.get_base_linear_vels()[:2]) - vpr * np.array(
    #         self.calculate_desired_direction_turning_direction()[:2])
    #     v0 = math.sqrt(pow(v0[0], 2) + pow(v0[1], 2))
    #     wxy = math.sqrt(pow(self.get_base_angular_vels()[0], 2) + pow(self.get_base_angular_vels()[1], 2))
    #
    #     if vpr == 0:
    #         base_reward = math.exp((-1.5) * pow(v_0, 2)) + math.exp((-1.5) * pow(wxy, 2))
    #
    #     else:
    #         base_reward = math.exp((-1.5) * pow(v0, 2)) + math.exp((-1.5) * pow(wxy, 2))
    #
    #     smooth_reward_list = np.array(self.leg0_tar_t) - 2 * np.array(self.leg0_tar_tminus1) + np.array(
    #         self.leg0_tar_tminus2) + np.array(self.leg1_tar_t) - 2 * np.array(self.leg1_tar_tminus1) + np.array(
    #         self.leg1_tar_tminus2) + np.array(self.leg2_tar_t) - 2 * np.array(self.leg2_tar_tminus1) + np.array(
    #         self.leg2_tar_tminus2) + np.array(self.leg3_tar_t) - 2 * np.array(self.leg3_tar_tminus1) + np.array(
    #         self.leg3_tar_tminus2)
    #     smooth_reward = (-1) * math.sqrt(
    #         pow(smooth_reward_list[0], 2) + pow(smooth_reward_list[1], 2) + pow(smooth_reward_list[2], 2))
    #
    #     reward = energy_reward * ener_coeff + lin_vel_coeff * lin_vel_reward + ang_vel_reward * ang_vel_coeff + base_reward * base_pen_coeff + smooth_reward * smooth_coeff
    #
    #     info = {}
    #
    #     info['energy_reward'] = energy_reward * ener_coeff
    #     info['vel_reward'] = lin_vel_coeff * lin_vel_reward
    #     info['ang_vel_reward'] = ang_vel_reward * ang_vel_coeff
    #     info['base_reward'] = base_reward * base_pen_coeff
    #     info['smooth_reward'] = smooth_reward * smooth_coeff
    #
    #     return reward, info

    def calculate_reward(self):
        lin_vel_coeff = 7
        lateral_coeff = 0.01
        smooth_coeff = 0.00001
        ener_coeff = 0.000001
        z_vel_coeff = 0.001
        action_rate_coeff = 0.001
        survival_bonus = 0.01

        smooth_lim = 0.025
        ener_lim = 0.0004
        z_vel_lim = 0.5
        action_rate_lim = 1
        lateral_lim = 3
        vx, vy, vz = p.getBaseVelocity(self.robot_id)[0]
        wz = self.get_base_angular_vels()[-1]

        if self.counter % 500000 == 0 and self.counter != 1:
            smooth_coeff = min(smooth_coeff * 2, smooth_lim)
            ener_coeff = min(ener_coeff * 2, ener_lim)
            z_vel_coeff = min(z_vel_coeff * 2, z_vel_lim)
            action_rate_coeff = min(action_rate_coeff * 2, action_rate_lim)
            lateral_coeff = min(lateral_coeff * 2, lateral_lim)

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
        forward_vel_reward = min(self.vx, self.desired_speed)
        lateral_reward = (pow(self.vy - vy, 2) + pow(self.desired_twisting_speed * self.twist_dir - wz, 2)) * (-1)

        smooth_reward_list = np.array(self.leg0_tar_t) - 2 * np.array(self.leg0_tar_tminus1) + np.array(
            self.leg0_tar_tminus2) + np.array(self.leg1_tar_t) - 2 * np.array(self.leg1_tar_tminus1) + np.array(
            self.leg1_tar_tminus2) + np.array(self.leg2_tar_t) - 2 * np.array(self.leg2_tar_tminus1) + np.array(
            self.leg2_tar_tminus2) + np.array(self.leg3_tar_t) - 2 * np.array(self.leg3_tar_tminus1) + np.array(
            self.leg3_tar_tminus2)
        smooth_reward = (-1) * math.sqrt(
            pow(smooth_reward_list[0], 2) + pow(smooth_reward_list[1], 2) + pow(smooth_reward_list[2], 2))
        z_vel_reward = (-1) * pow(vz, 2)

        action_rate_list = np.array(self.leg3_tar_t) - np.array(self.leg3_tar_tminus1) + np.array(
            self.leg2_tar_t) - np.array(self.leg2_tar_tminus1) + \
                           np.array(self.leg1_tar_t) - np.array(self.leg1_tar_tminus1) + np.array(
            self.leg0_tar_t) - np.array(self.leg0_tar_tminus1)

        action_rate_reward = (-1) * math.sqrt(
            pow(action_rate_list[0], 2) + pow(action_rate_list[1], 2) + pow(action_rate_list[2], 2))

        # print("line vel", forward_vel_reward)
        # print("lateral", lateral_reward)
        # print("action rate", action_rate_reward)
        # print("smoothness", smooth_reward)
        # print("z vel", z_vel_reward)
        # print("energy", energy_reward)

        reward = energy_reward * ener_coeff + smooth_reward * smooth_coeff + forward_vel_reward * lin_vel_coeff + lateral_reward * lateral_coeff + \
                 z_vel_reward * z_vel_coeff + action_rate_reward * action_rate_coeff + survival_bonus

        info = {}

        info['energy_reward'] = energy_reward * ener_coeff
        info['vel_reward'] = lin_vel_coeff * forward_vel_reward
        info['lateral_vel_reward'] = lateral_reward * lateral_coeff
        info['z-vel_reward'] = z_vel_reward * z_vel_coeff
        info['smooth_reward'] = smooth_reward * smooth_coeff
        info['action_rate_reward'] = action_rate_reward * action_rate_coeff
        info['survival_reward'] = survival_bonus

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
    obs = qp.get_obs()
    # print("obserbations", len(obs))
    # val = qp.get_swing_foot_tgt(0.3, [0.25, 0.25])
    # print("val", val)
    while True:
        # qp.reset()
        # reward = qp.reward_function()
        done = qp._is_done()
        # print("done", done)
        # print("rewards", reward)
        phase0 = qp.phase_leg0
        phase1 = qp.phase_leg1
        phase2 = qp.phase_leg2
        phase3 = qp.phase_leg3

        # print("phases", phase0, phase1, phase2, phase3)
        if 0 <= phase0 <= math.pi:
            x0 = 0.2
            y0 = 0.005
        elif math.pi <= phase0 <= 2 * math.pi:
            x0 = -0.2
            y0 = 0

        if 0 <= phase1 <= math.pi:
            x1 = 0.2
            y1 = -0.005
        elif math.pi <= phase1 <= 2 * math.pi:
            x1 = -0.2
            y1 = 0

        if 0 <= phase2 <= math.pi:
            x2 = 0.2
            y2 = -0.005
        elif math.pi <= phase2 <= 2 * math.pi:
            x2 = -0.2
            y2 = 0

        if 0 <= phase3 <= math.pi:
            x3 = 0.2
            y3 = 0.005
        elif math.pi <= phase3 <= 2 * math.pi:
            x3 = -0.2
            y3 = 0

        if qp.counter <= 20:
            x0 = 0.1
            y0 = 0

            x1 = 0
            y1 = 0.005

            x2 = 0
            y2 = 0.005

            x3 = 0.1
            y3 = 0.005

        # print("x0", x0)
        # print("x3", x3)

        action = np.array([0, 0, 0, 0, -x0, 0, 0, -x1, 0, 0, -x2, 0, 0, -x3, 0, 0])
        reward = qp.calculate_reward()
        # if qp.counter % 25 == 0:
        #     force = [random.uniform(-100, 100) for i in range(3)]  # Specify the force vector [X, Y, Z]
        #     position = [0, 0, 0]  # Specify the position where the force is applied [X, Y, Z]
        #     link_index = -1
        #     p.applyExternalForce(qp.robot_id, link_index, force, position, flags=p.WORLD_FRAME)
        #     print(force)

        qp.step(action)
    #     print(action)
