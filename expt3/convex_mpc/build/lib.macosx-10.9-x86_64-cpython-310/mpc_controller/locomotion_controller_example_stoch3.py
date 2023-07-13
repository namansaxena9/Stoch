from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import inspect

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
os.sys.path.insert(0, parentdir)

from absl import app
from absl import flags
import scipy.interpolate
import numpy as np
import pybullet_data as pd
from pybullet_utils import bullet_client

import pybullet
import random

from mpc_controller import com_velocity_estimator
from mpc_controller import gait_generator as gait_generator_lib
from mpc_controller import locomotion_controller
from mpc_controller import openloop_gait_generator
from mpc_controller import raibert_swing_leg_controller
from mpc_controller import torque_stance_leg_controller

from mpc_controller import stoch3_sim as robot_sim

np.set_printoptions(linewidth=200, precision = 3, suppress=True)

FLAGS = flags.FLAGS

_NUM_SIMULATION_ITERATION_STEPS = 300

# For faster trotting (v > 1.5 ms reduce this to 0.13s).
_STANCE_DURATION_SECONDS = [
    0.3
] * 4

# Trotting
_DUTY_FACTOR = [0.6] * 4
_INIT_PHASE_FULL_CYCLE = [0.9, 0., 0., 0.9] # [0.9, 0, 0, 0.9]
_MAX_TIME_SECONDS = 50

_INIT_LEG_STATE = (
    gait_generator_lib.LegState.SWING,
    gait_generator_lib.LegState.STANCE,
    gait_generator_lib.LegState.STANCE,
    gait_generator_lib.LegState.SWING,
)

ADD_INIT_WEIGHT = True

def _generate_example_linear_angular_speed(t):
  """Creates an example speed profile based on time for demo purpose."""
  vx = 0.6 * robot_sim.MPC_VELOCITY_MULTIPLIER
  vy = 0.6 * robot_sim.MPC_VELOCITY_MULTIPLIER
  wz = 0.8 * robot_sim.MPC_VELOCITY_MULTIPLIER
  
  time_points = (0, 5, 10, 15, 20, 25,30)
  speed_points = ((vx, 0, 0, 0), (0, 0, 0, wz), (vx, 0, 0, wz), (0, vy, 0, 0), (-vx, 0, 0, wz),
                  (0, 0, -vy, 0), (-vx, 0, 0, -wz))

  speed = scipy.interpolate.interp1d(
      time_points,
      speed_points,
      kind="previous",
      fill_value="extrapolate",
      axis=0)(
          t)

  return speed[0:3], speed[3]

def _setup_controller(robot):
  """Demonstrates how to create a locomotion controller."""
  desired_speed = (0, 0)
  desired_twisting_speed = 0

  gait_generator = openloop_gait_generator.OpenloopGaitGenerator(
      robot,
      stance_duration=_STANCE_DURATION_SECONDS,
      duty_factor=_DUTY_FACTOR,
      initial_leg_phase=_INIT_PHASE_FULL_CYCLE,
      initial_leg_state=_INIT_LEG_STATE)
  state_estimator = com_velocity_estimator.COMVelocityEstimator(robot,
                                                                window_size=20)
  sw_controller = raibert_swing_leg_controller.RaibertSwingLegController(
      robot,
      gait_generator,
      state_estimator,
      desired_speed=desired_speed,
      desired_twisting_speed=desired_twisting_speed,
      desired_height=robot_sim.MPC_BODY_HEIGHT,
      foot_clearance=0.01)

  st_controller = torque_stance_leg_controller.TorqueStanceLegController(
      robot,
      gait_generator,
      state_estimator,
      desired_speed=desired_speed,
      desired_twisting_speed=desired_twisting_speed,
      desired_body_height=robot_sim.MPC_BODY_HEIGHT,
      body_mass=robot_sim.MPC_BODY_MASS,
      body_inertia=robot_sim.MPC_BODY_INERTIA)

  controller = locomotion_controller.LocomotionController(
      robot=robot,
      gait_generator=gait_generator,
      state_estimator=state_estimator,
      swing_leg_controller=sw_controller,
      stance_leg_controller=st_controller,
      clock=robot.GetTimeSinceReset)
  return controller


def _update_controller_params(controller, lin_speed, ang_speed):
  controller.swing_leg_controller.desired_speed = lin_speed
  controller.swing_leg_controller.desired_twisting_speed = ang_speed
  controller.stance_leg_controller.desired_speed = lin_speed
  controller.stance_leg_controller.desired_twisting_speed = ang_speed


def _run_example(max_time=_MAX_TIME_SECONDS):
  """Runs the locomotion controller example."""
  
  #recording video requires ffmpeg in the path
  record_video = False
  if record_video:
    p = pybullet
    p.connect(p.GUI, options="--width=1280 --height=720 --mp4=\"test.mp4\" --mp4fps=100")
    p.configureDebugVisualizer(p.COV_ENABLE_SINGLE_STEP_RENDERING,1)
  else:
     p = bullet_client.BulletClient(connection_mode=pybullet.GUI)    
         
  p.configureDebugVisualizer(p.COV_ENABLE_GUI,0)
  p.configureDebugVisualizer(p.COV_ENABLE_RENDERING,0)
  # p.startStateLogging(pybullet.STATE_LOGGING_VIDEO_MP4, "zeroHD.mp4")  
  p.setAdditionalSearchPath(pd.getDataPath())
  
  num_bullet_solver_iterations = 30

  p.setPhysicsEngineParameter(numSolverIterations=num_bullet_solver_iterations)

  p.setPhysicsEngineParameter(enableConeFriction=0)
  p.setPhysicsEngineParameter(numSolverIterations=30)
  simulation_time_step = 0.001

  p.setTimeStep(simulation_time_step)
 
  p.setGravity(0, 0, -9.8)
  p.setPhysicsEngineParameter(enableConeFriction=0)
  p.setAdditionalSearchPath(pd.getDataPath())
  
  #random.seed(10)
  #p.configureDebugVisualizer(p.COV_ENABLE_RENDERING,0)
  heightPerturbationRange = 0.06
  
  plane = True
  if plane:
    p.loadURDF("plane.urdf")
    #planeShape = p.createCollisionShape(shapeType = p.GEOM_PLANE)
    #ground_id  = p.createMultiBody(0, planeShape)
  else:
    numHeightfieldRows = 256
    numHeightfieldColumns = 256
    heightfieldData = [0]*numHeightfieldRows*numHeightfieldColumns 
    for j in range (int(numHeightfieldColumns/2)):
      for i in range (int(numHeightfieldRows/2) ):
        height = random.uniform(0,heightPerturbationRange)
        heightfieldData[2*i+2*j*numHeightfieldRows]=height
        heightfieldData[2*i+1+2*j*numHeightfieldRows]=height
        heightfieldData[2*i+(2*j+1)*numHeightfieldRows]=height
        heightfieldData[2*i+1+(2*j+1)*numHeightfieldRows]=height
    
    terrainShape = p.createCollisionShape(shapeType = p.GEOM_HEIGHTFIELD, meshScale=[.05,.05,1], heightfieldTextureScaling=(numHeightfieldRows-1)/2, heightfieldData=heightfieldData, numHeightfieldRows=numHeightfieldRows, numHeightfieldColumns=numHeightfieldColumns)
    ground_id  = p.createMultiBody(0, terrainShape)

  #p.resetBasePositionAndOrientation(ground_id,[0,0,0], [0,0,0,1])
  
  #p.changeDynamics(ground_id, -1, lateralFriction=1.0)
  
  robot_uid = p.loadURDF(robot_sim.URDF_NAME, robot_sim.START_POS)

  robot = robot_sim.SimpleRobot(p, robot_uid, simulation_time_step=simulation_time_step,
                                add_init_weight=ADD_INIT_WEIGHT)
  
  controller = _setup_controller(robot)
  controller.reset()
  
  p.configureDebugVisualizer(p.COV_ENABLE_RENDERING,1)

  current_time = robot.GetTimeSinceReset()
  #logId = p.startStateLogging(p.STATE_LOGGING_PROFILE_TIMINGS, "mpc.json")

  data = []
  
  while current_time < max_time:
    p.submitProfileTiming("loop")
    
    # Updates the controller behavior parameters.
    lin_speed, ang_speed = _generate_example_linear_angular_speed(current_time)
    _update_controller_params(controller, lin_speed, ang_speed)

    # Needed before every call to get_action().
    controller.update()
    hybrid_action, info, foot_contact = controller.get_action()

    contact_forces = [info['qp_sol'][i] for i in range(4)]
    contact_forces = np.vstack(contact_forces)
    
    robot.Step(hybrid_action)

    if ADD_INIT_WEIGHT:
        weight_id = robot.weight
        mass = p.getDynamicsInfo(weight_id, -1)[0]
        new_mass = mass + 0.5
        if new_mass > 25:
            new_mass = 25
        robot.pybullet_client.changeDynamics(weight_id, -1, mass=new_mass)

    pos = robot.GetBasePosition()
    pos_dot = robot.GetBaseVelocity()
    theta = robot.GetBaseRollPitchYaw()
    ang_vel = robot.pybullet_client.getBaseVelocity(robot.quadruped)[1]
    foot_pos = robot.GetFootPositionsInBaseFrame()
    # foot_contact = robot.GetFootContacts()

    data_dict = {}
    data_dict['p'] = np.array(pos)
    data_dict['p_dot'] = np.array(pos_dot)
    data_dict['theta'] = np.array(theta)
    data_dict['ang_vel'] = np.array(ang_vel)
    data_dict['foot_pos'] = foot_pos
    data_dict['foot_contact'] = np.array(foot_contact)
    data_dict['contact_force'] = contact_forces

    data.append(data_dict)

    if record_video:
      p.configureDebugVisualizer(p.COV_ENABLE_SINGLE_STEP_RENDERING,1)

    #time.sleep(0.003)
    current_time = robot.GetTimeSinceReset()
    p.submitProfileTiming()

  # import pickle
  # pickle.dump(data, open('data1.pkl', 'wb'))

  #p.stopStateLogging(logId)
  #while p.isConnected():
  #  time.sleep(0.1)

def main(argv):
  del argv
  _run_example()

if __name__ == "__main__":
  app.run(main)