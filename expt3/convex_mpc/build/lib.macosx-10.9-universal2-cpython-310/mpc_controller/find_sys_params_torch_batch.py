import numpy as np
import pickle
import torch
from scipy.spatial.transform import Rotation as R

dt = 0.025
g = -9.81
MPC_BODY_MASS = 25
MPC_BODY_INERTIA = np.array((0.05922, 0, 0, 0, 0.06835, 0, 0, 0, 0.56123)).reshape((3, 3))
L = np.linalg.cholesky(MPC_BODY_INERTIA)

class GetNextStates(torch.nn.Module):
    def __init__(self, init_params):
        self.params = torch.tensor(init_params, dtype=torch.float32, requires_grad=True)

    def ConvertToSkewSymmetric(self, x):
        return torch.tensor([[0, -x[2], x[1]],
                             [x[2], 0, -x[0]],
                             [-x[1], x[0], 0]], dtype=torch.float32)

    def calc_A_mat(self, com_roll_pitch_yaw):
        batch_size = com_roll_pitch_yaw.shape[0]
        A = np.zeros((batch_size, 13, 13), dtype=np.float32)
        for i in range(batch_size):
            cos_yaw = np.cos(com_roll_pitch_yaw[i, 2])
            sin_yaw = np.sin(com_roll_pitch_yaw[i, 2])
            cos_pitch = np.cos(com_roll_pitch_yaw[i, 1])
            tan_pitch = np.tan(com_roll_pitch_yaw[i, 1])
            angular_velocity_to_rpy_rate = np.array([
                [cos_yaw / cos_pitch, sin_yaw / cos_pitch, 0],
                [-sin_yaw, cos_yaw, 0],
                [cos_yaw * tan_pitch, sin_yaw * tan_pitch, 1]])
            A[i, 0:3, 6:6 + 3] = angular_velocity_to_rpy_rate
            A[i, 3:6, 9:12] = np.eye(3)
            A[i, 11, 12] = 1
        return torch.tensor(A, dtype=torch.float32)

    def calc_B_mat(self, inv_mass, foot_contacts, foot_pos_world_frame, inv_inertia_world):
        batch_size = foot_contacts.shape[0]
        B = torch.zeros((batch_size, 13, 12), dtype=torch.float32, requires_grad=False)
        for i in range(batch_size):
            for j in range(4):
                r_hat = self.ConvertToSkewSymmetric(foot_pos_world_frame[i, j])
                B[i, 6:6 + 3, j * 3:j * 3 + 3] = foot_contacts[i, j] * torch.matmul(inv_inertia_world[i], r_hat)
                B[i, 9, j * 3] = foot_contacts[i, j] * inv_mass
                B[i, 10, j * 3 + 1] = foot_contacts[i, j] * inv_mass
                B[i, 11, j * 3 + 2] = foot_contacts[i, j] * inv_mass
        return B

    def forward(self, states, actions, foot_positions_body_frame, foot_contacts):
        '''

        :param states: (batch_size, state_dim)
        :param actions: (batch_size, action_dim)
        :param foot_positions_body_frame: (batch_size, 4, 3)
        :param foot_contacts: (batch_size, 4)
        :return: next_states: (batch_size, state_dim)
        '''

        batch_size = states.shape[0]
        inv_mass = 1 / self.params[0]

        L = torch.zeros((3, 3), requires_grad=True)
        indices = torch.tril(torch.ones((3, 3), dtype=torch.bool))
        L = L.masked_scatter(indices, self.params[1:7])

        inertia = torch.matmul(L, L.T) + 1e-5
        inv_inertia = torch.inverse(inertia)

        com_roll_pitch_yaw = states[:, 0:3]
        R_body2world = torch.zeros((batch_size, 3, 3), dtype=torch.float32)
        foot_pos_world_frame = torch.zeros_like(foot_positions_body_frame)

        for i in range(batch_size):
            R_body2world[i, :, :] = torch.tensor(R.from_euler('zyx', reversed(com_roll_pitch_yaw[i, :]),
                                                              degrees=False).as_matrix(), dtype=torch.float32)
            foot_pos_world_frame[i, :, :] = torch.transpose(torch.matmul(R_body2world[i, :, :],
                                                                         foot_positions_body_frame[i, :, :].T), 0, 1)

        inv_inertia_world = torch.matmul(torch.matmul(R_body2world, inv_inertia), torch.transpose(R_body2world, 1, 2))
        self.A = self.calc_A_mat(com_roll_pitch_yaw)
        self.B = self.calc_B_mat(inv_mass, foot_contacts, foot_pos_world_frame, inv_inertia_world)

        # Euler integration scheme
        self.A = torch.eye(13) + self.A * dt
        self.B = self.B * dt

        next_states = torch.matmul(self.A, states.unsqueeze(2)) \
                      + torch.matmul(self.B, actions.unsqueeze(2))
        return next_states.squeeze()

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
        # foot_pos = foot_pos.flatten(order='C')
        states.append(st)
        actions.append(u)
        foot_contacts.append(foot_contact)
        foot_positions.append(foot_pos)

    return states, actions, foot_contacts, foot_positions

start = 1000
size = 101
data = pickle.load(open('data.pkl', 'rb'))[start:start+size]
states, actions, foot_contacts, foot_positions = get_data(data)

states = torch.tensor(states, dtype=torch.float32)
actions = torch.tensor(np.array(actions), dtype=torch.float32)
foot_contacts = torch.tensor(np.array(foot_contacts), dtype=torch.bool)
foot_positions = torch.tensor(np.array(foot_positions), dtype=torch.float32)

params = [20.0, L[0, 0], L[1, 0], L[1, 1], L[2, 0], L[2, 1], L[2, 2]]

model = GetNextStates(params)
next_true_states = states[1:size]

learning_rate = 0.9
n_iters = 500
loss = torch.nn.MSELoss()

alpha = torch.tensor([0.9, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1], dtype=torch.float32)

for epoch in range(n_iters):
    next_pred_states = model.forward(states, actions, foot_positions, foot_contacts)[:size-1]
    L = torch.sqrt(loss(next_pred_states, next_true_states))
    L.backward(retain_graph=True)

    print('epoch: ', epoch)
    print('loss: ', L.detach().numpy())
    print('grad: ', model.params.grad.detach().numpy())
    print('params: ', model.params.detach().numpy())

    with torch.no_grad():
        model.params -= torch.mul(alpha, model.params.grad)
        model.params.grad.zero_()

parms_star = model.params.detach().numpy()
print('MASS: ', parms_star[0])

L = np.array([[parms_star[1],           0.0,            0.0],
              [parms_star[2], parms_star[3],            0.0],
              [parms_star[4], parms_star[5], parms_star[6]]])

INERTIA = L @ L.T + 1e-6
print('INERTIA: ', INERTIA)