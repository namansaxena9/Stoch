import numpy as np
import pinocchio
import pickle

dt = 0.025
MPC_BODY_MASS = 25
MPC_BODY_INERTIA = np.array((0.05922, 0, 0, 0, 0.06835, 0, 0, 0, 0.56123)).reshape(3,3)

def ConvertToSkewSymmetric(x: np.ndarray):
    return np.array([[   0, -x[2],  x[1]],
                    [ x[2],     0, -x[0]],
                    [-x[1],  x[0],    0]], dtype=float)

class CentroidalModel():
    def __init__(self, mass, inertia):
        self.mass = mass
        self.inertia = inertia

        self.inv_inertia = np.linalg.inv(self.inertia)
        self.inv_mass = 1.0 / self.mass

        self.g = -9.81
        self.gravity_vector = np.array([0, 0, self.g])

    def prepareModel(self,
                     com_position, com_velocity, com_roll_pitch_yaw, com_angular_velocity,
                     foot_contact_states, foot_positions_body_frame):

        # Compute the foot positions in the world frame.
        self.R_body2world = pinocchio.rpy.rpyToMatrix(*com_roll_pitch_yaw)
        self.foot_positions_world_frame = np.zeros_like(foot_positions_body_frame)

        for i in range(4):
            self.foot_positions_world_frame[i] = self.R_body2world @ foot_positions_body_frame[i]

        self.contact_condition = (foot_contact_states > 0).astype(int)
        self.foot_positions_body_frame = foot_positions_body_frame

        self.calc_A_mat(com_roll_pitch_yaw)
        self.calc_B_mat()

    def dicrete_model(self):
        '''
        Generates A_hat and B-hat matrices s.t
        x_{k+1} = A_hat x_{k} + B_hat u_{k}
        According to Euler's integration scheme
        A_hat = I + A.T
        B_hat = B.T
        where T is the sampling interval (1/sampling_frequency)
        :return: A_hat and B-hat
        '''
        self.A_mat = np.eye(13) + self.A_mat * dt
        self.B_mat = self.B_mat * dt

    def calc_A_mat(self, com_roll_pitch_yaw):
        A = np.zeros((13, 13))
        cos_yaw = np.cos(com_roll_pitch_yaw[2])
        sin_yaw = np.sin(com_roll_pitch_yaw[2])
        cos_pitch = np.cos(com_roll_pitch_yaw[1])
        tan_pitch = np.tan(com_roll_pitch_yaw[1])
        angular_velocity_to_rpy_rate = np.array([
            [cos_yaw / cos_pitch, sin_yaw / cos_pitch, 0],
            [-sin_yaw, cos_yaw, 0],
            [cos_yaw * tan_pitch, sin_yaw * tan_pitch, 1]])
        A[0:3, 6:6 + 3] = angular_velocity_to_rpy_rate
        A[3, 9] = 1
        A[4, 10] = 1
        A[5, 11] = 1
        A[11, 12] = 1
        self.A_mat = A

    def calc_B_mat(self):
        B = np.zeros((13, 12))
        inertia_world = self.R_body2world @ self.inertia @ self.R_body2world.T
        inv_inertia_world = self.R_body2world @ self.inv_inertia @ self.R_body2world.T
        for i in range(4):
            B[6:6 + 3, i * 3:i * 3 + 3] = self.contact_condition[i] * inv_inertia_world @ ConvertToSkewSymmetric(
                self.foot_positions_world_frame[i])
            B[9, i * 3] = self.contact_condition[i] * self.inv_mass
            B[10, i * 3 + 1] = self.contact_condition[i] * self.inv_mass
            B[11, i * 3 + 2] = self.contact_condition[i] * self.inv_mass
        self.B_mat = B

data = pickle.load(open('data.pkl', 'rb'))

com_pos = data[0]['p']
com_vel = data[0]['p_dot']
com_roll_pitch_yaw = data[0]['theta']
com_ang_vel = data[0]['ang_vel']
foot_pos = data[0]['foot_pos']
foot_contacts = data[0]['foot_contact']

SRB_model = CentroidalModel(mass=MPC_BODY_MASS, inertia=MPC_BODY_INERTIA)
SRB_model.prepareModel(com_position = com_pos, com_velocity = com_vel,
                       com_roll_pitch_yaw = com_roll_pitch_yaw,
                       com_angular_velocity = com_ang_vel,
                       foot_contact_states = np.array(foot_contacts),
                       foot_positions_body_frame = foot_pos)
SRB_model.dicrete_model()

A, B = SRB_model.A_mat, SRB_model.B_mat
print(A.shape, B.shape)

