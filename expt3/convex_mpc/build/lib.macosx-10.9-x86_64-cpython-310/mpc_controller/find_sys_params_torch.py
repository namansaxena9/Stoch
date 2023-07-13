import numpy as np
import pickle
import torch
from scipy.spatial.transform import Rotation as R

dt = 0.025
g = -9.81
MPC_BODY_MASS = 25
MPC_BODY_INERTIA = np.array((0.05922, 0, 0, 0, 0.06835, 0, 0, 0, 0.56123)).reshape((3, 3))
L = np.linalg.cholesky(MPC_BODY_INERTIA)


def ConvertToSkewSymmetric(x: torch.tensor):
    return torch.tensor([[0, -x[2],  x[1]],
                    [ x[2],     0, -x[0]],
                    [-x[1],  x[0],    0]], dtype=torch.float32)


def calc_A_mat(com_roll_pitch_yaw: torch.tensor):
    A = np.zeros((13,13), dtype=np.float32)
    cos_yaw = np.cos(com_roll_pitch_yaw[2])
    sin_yaw = np.sin(com_roll_pitch_yaw[2])
    cos_pitch = np.cos(com_roll_pitch_yaw[1])
    tan_pitch = np.tan(com_roll_pitch_yaw[1])
    angular_velocity_to_rpy_rate = np.array([
        [cos_yaw / cos_pitch, sin_yaw / cos_pitch, 0],
        [-sin_yaw, cos_yaw, 0],
        [cos_yaw * tan_pitch, sin_yaw * tan_pitch, 1]])
    A[0:3, 6:6 + 3] = angular_velocity_to_rpy_rate
    A[3:6, 9:12] = np.eye(3)
    A[11, 12] = 1
    return torch.tensor(A, dtype=torch.float32)


def calc_B_mat(inv_mass, foot_contacts, foot_pos_world_frame, inv_inertia_world):
    B = torch.zeros((13, 12))
    for i in range(4):
        r_hat = ConvertToSkewSymmetric(foot_pos_world_frame[i])
        B[6:6 + 3, i * 3:i * 3 + 3] = foot_contacts[i] * torch.matmul(inv_inertia_world, r_hat)
        B[9, i * 3] = foot_contacts[i] * inv_mass
        B[10, i * 3 + 1] = foot_contacts[i] * inv_mass
        B[11, i * 3 + 2] = foot_contacts[i] * inv_mass
    return B

def get_next_state(st, u, params, foot_pos_body_frame, foot_contacts):
    '''

    :param st: current_state = (rpy, p, ang_vel, vel, gravity_term)
    :param u: contact_force = (f_FL, f_FR, f_BL, f_BR)
    :param params: (M, pararms of cholesky decomposition of Inertia matrix,
                    foot_pos_body_frame, foot_contacts)
    :param foot_contacts: list of foot contacts
    :param foot_pos_body_frame: 4 x 3 matrix of foot positions in body frame
    :return: Next state
    '''

    inv_mass = 1/params[0]

    L = torch.zeros((3, 3), requires_grad=True)
    indices = torch.tril(torch.ones((3, 3), dtype=torch.bool))
    L = L.masked_scatter(indices, params[1:7])

    inertia = torch.matmul(L, L.T) + 1e-5
    inv_inertia = torch.inverse(inertia)

    com_roll_pitch_yaw = st[0:3, 0]
    com_yaw_pitch_roll = [st[2, 0], st[1, 0], st[0, 0]]
    R_body2world = torch.tensor(R.from_euler('zyx', com_yaw_pitch_roll, degrees=False).as_matrix(), dtype=torch.float32)

    foot_pos_world_frame = torch.zeros_like(foot_pos_body_frame)

    for i in range(4):
        foot_pos_world_frame[i] = torch.matmul(R_body2world, foot_pos_body_frame[i])

    inv_inertia_world = torch.matmul(R_body2world, torch.matmul(inv_inertia, R_body2world.T))

    A = calc_A_mat(com_roll_pitch_yaw)
    B = calc_B_mat(inv_mass, foot_contacts, foot_pos_world_frame, inv_inertia_world)

    # According to convex-mpc code
    # AB = torch.concatenate([A * dt, B * dt], dim=1)
    # zeros = torch.zeros(size=(12, 25), dtype=torch.float32)
    # AB = torch.concatenate([AB, zeros])
    # AB = torch.matrix_exp(AB)[:13, :]
    # next_state = AB @ torch.concatenate([st, u])

    # Simple Euler integration scheme
    A = torch.eye(13) + A * dt
    B = B * dt
    next_state = A @ st + B @ u

    return next_state


def get_data(data: list):
    '''

    :param data: list of dictionaries consisting of com position, velocities along with foot contacts
                 and foot positions
    :return: states, actions, foot_contacts, foot_positions
    '''
    states = []
    actions = []
    foot_contacts = []
    foot_positions = []

    for i in range(len(data)):
        com_pos = data[i]['p']
        com_vel = data[i]['p_dot']
        com_roll_pitch_yaw = data[i]['theta']
        com_ang_vel = data[i]['ang_vel']
        foot_pos = data[i]['foot_pos']
        foot_contact = list(data[i]['foot_contact'])

        st = list(com_roll_pitch_yaw) + list(com_pos) + list(com_ang_vel) + list(com_vel) + [g]
        u = data[i]['contact_force']
        u = u.flatten(order='C')
        foot_pos = foot_pos.flatten(order='C')
        states.append(st)
        actions.append(u)
        foot_contacts.append(foot_contact)
        foot_positions.append(foot_pos)

    return states, actions, foot_contacts, foot_positions


data = pickle.load(open('data.pkl', 'rb'))[:1000]
states, actions, foot_contacts, foot_positions = get_data(data)

params = torch.tensor([25.0, L[0, 0], L[1, 0], L[1, 1], L[2, 0], L[2, 1], L[2, 2]],
                      dtype=torch.float32, requires_grad=True)

next_states = torch.zeros((len(states), 13), dtype=torch.float32, requires_grad=False)

for i in range(len(states)-1):
    st = torch.tensor(np.reshape(states[i], newshape=(-1, 1)), dtype=torch.float32, requires_grad=False)
    u = torch.tensor(np.reshape(actions[i], newshape=(-1, 1)), dtype=torch.float32, requires_grad=False)
    foot_pos = torch.tensor(np.reshape(foot_positions[i], newshape=(4, 3), order='C'),
                            dtype=torch.float32, requires_grad=False)
    foot_contact = torch.tensor(np.array(foot_contacts[i]), requires_grad=False)
    next_state = get_next_state(st, u, params, foot_pos, foot_contact)
    next_states[i, :] = next_state.T

print(next_states.shape)
print(next_states[:20])












