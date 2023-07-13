import numpy
import numpy as np
import pinocchio
import pickle
from autograd import grad, jacobian
import autograd.numpy as np

dt = 0.025
g = -9.81
MPC_BODY_MASS = 25
MPC_BODY_INERTIA = np.array((0.05922, 0, 0, 0, 0.06835, 0, 0, 0, 0.56123)).reshape(3, 3)
L = np.linalg.cholesky(MPC_BODY_INERTIA)


def ConvertToSkewSymmetric(x: np.ndarray):
    return np.array([[0, -x[2],  x[1]],
                    [ x[2],     0, -x[0]],
                    [-x[1],  x[0],    0]], dtype=float)


def calc_A_mat(com_roll_pitch_yaw):
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
    return A


def calc_B_mat(mass, foot_contacts, foot_pos_world_frame, inv_inertia_world, get_gradient):
    B = np.zeros((13, 12))
    if get_gradient:
        inv_inertia_world = inv_inertia_world._value
        mass = mass._value
    for i in range(4):
        B[6:6 + 3, i * 3:i * 3 + 3] = foot_contacts[i] * inv_inertia_world @ ConvertToSkewSymmetric(
            foot_pos_world_frame[i])
        B[9, i * 3] = foot_contacts[i] / mass
        B[10, i * 3 + 1] = foot_contacts[i] / mass
        B[11, i * 3 + 2] = foot_contacts[i] / mass
    return B


def get_next_state(st, u, params, foot_pos_body_frame, foot_contacts, get_gradient=False):
    '''

    :param st: current_state = (rpy, p, ang_vel, vel, gravity_term)
    :param u: contact_force = (f_FL, f_FR, f_BL, f_BR)
    :param params: (M, pararms of cholesky decomposition of Inertia matrix,
                    foot_pos_body_frame, foot_contacts)
    :param foot_contacts: list of foot contacts
    :param foot_pos_body_frame: 4 x 3 matrix of foot positions in body frame
    :return: Next state
    '''

    mass = params[0]

    l_11 = params[1]
    l_21 = params[2]
    l_22 = params[3]
    l_31 = params[4]
    l_32 = params[5]
    l_33 = params[6]

    L = np.array([[l_11, 0, 0], [l_21, l_22, 0], [l_31, l_32, l_33]])

    inertia = L @ L.T + 1e-5
    inv_inertia = np.linalg.inv(inertia)

    com_roll_pitch_yaw = st[0:3, 0]
    R_body2world = pinocchio.rpy.rpyToMatrix(*com_roll_pitch_yaw)

    foot_pos_world_frame = np.zeros_like(foot_pos_body_frame)

    for i in range(4):
        foot_pos_world_frame[i] = R_body2world @ foot_pos_body_frame[i]

    inertia_world = R_body2world @ inertia @ R_body2world.T
    inv_inertia_world = R_body2world @ inv_inertia @ R_body2world.T

    A = calc_A_mat(com_roll_pitch_yaw)
    B = calc_B_mat(mass, foot_contacts, foot_pos_world_frame, inv_inertia_world, get_gradient)

    A = np.eye(A.shape[0]) + A * dt
    B = B * dt

    next_state = A @ st + B @ u

    return next_state


def get_grad_fn(dynamics, st, u, params,foot_pos_body_frame, foot_contacts):
    grad_fn = jacobian(dynamics, 2)
    grad_params = grad_fn(st, u, params, foot_pos_body_frame, foot_contacts)
    return np.squeeze(grad_params)


data = pickle.load(open('data.pkl', 'rb'))


def get_data(data):
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


states, actions, foot_contacts, foot_positions = get_data(data)

params = [0] * 7

params[0] = MPC_BODY_MASS * 1.0
params[1] = L[0, 0]
params[2] = L[1, 0]
params[3] = L[1, 1]
params[4] = L[2, 0]
params[5] = L[2, 1]
params[6] = L[2, 2]

params = np.array(params)

foot_pos = np.reshape(foot_positions[0], newshape=(4,3), order='C')
foot_contact = foot_contacts[0]

st = np.array(states[0]).reshape(-1,1)
u = np.array(actions[0]).reshape(-1,1)

next_pred_state = get_next_state(st, u, params, foot_pos, foot_contact)
print(next_pred_state)

grad_params = get_grad_fn(lambda x, u, params, foot_position, foot_contact:
                          get_next_state(x, u, params, foot_position, foot_contact, get_gradient=True),
                          st, u, params, foot_pos, foot_contact)
print(grad_params.shape)