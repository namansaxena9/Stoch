import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

import datagen

LOG_SIG_MAX = 2
LOG_SIG_MIN = -20
epsilon = 1e-6
GAMMA = 0.99

dg = datagen.Datagen()

policy_fn1 = "GaussianPolicy.pt1"
value1_fn1 = "QNetwork.pt1"
value2_fn1 = "QNetwork_copy.pt1"


def build_mlp(n_in, hidden, n_out, act_fn):
    hidden.append(n_out)
    li = []
    li.append(nn.Linear(n_in, hidden[0]))
    for i in range(1, len(hidden)):
        li.append(act_fn())
        li.append(nn.Linear(hidden[i - 1], hidden[i]))

    return nn.Sequential(*nn.ModuleList(li))


# Initialize Policy weights
def weights_init_(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=1)
        torch.nn.init.constant_(m.bias, 0)


class QNetwork(nn.Module):
    def __init__(self, num_actions,):
        super(QNetwork, self).__init__()

        # Q1 architecture
        self.q1_x = build_mlp(71, [72], 64, nn.Tanh)
        self.q1 = build_mlp(194 + num_actions, [256, 128, 64], 1, nn.Tanh)

        self.apply(weights_init_)

    def forward(self, state, action):
        # xu = torch.cat([state, action], 1)
        x1 = self.q1(torch.concat((state[:, :130].to(torch.float32), F.tanh(self.q1_x(state[:, 130:].to(torch.float32))), action.to(torch.float32)), 1))

        return x1


class QNetwork_copy(nn.Module):
    def __init__(self, num_actions,):
        super(QNetwork_copy, self).__init__()

        # Q1 architecture
        self.q1_x = build_mlp(71, [72], 64, nn.Tanh)
        self.q1 = build_mlp(194 + num_actions, [256, 128, 64], 1, nn.Tanh)

        self.apply(weights_init_)

    def forward(self, state, action):
        # xu = torch.cat([state, action], 1)

        x1 = self.q1(torch.concat((state[:, :130].to(torch.float32), F.tanh(self.q1_x(state[:, 130:].to(torch.float32))), action.to(torch.float32)), 1))

        return x1


class GaussianPolicy(nn.Module):
    def __init__(self, num_actions, action_space=None):
        super(GaussianPolicy, self).__init__()

        self.policy_net_x = build_mlp(71, [72], 64, nn.Tanh)
        self.policy_net = build_mlp(194, [256, 128], 64, nn.Tanh)

        self.mean_linear = nn.Linear(64, num_actions)
        self.log_std_linear = nn.Linear(64, num_actions)

        self.apply(weights_init_)

        # action rescaling
        if action_space is None:
            self.action_scale = torch.tensor(1.)
            self.action_bias = torch.tensor(0.)
        else:
            self.action_scale = torch.FloatTensor(
                (action_space.high - action_space.low) / 2.)
            self.action_bias = torch.FloatTensor(
                (action_space.high + action_space.low) / 2.)

    def forward(self, state):
        x = self.policy_net(torch.concat((state[:, :130].to(torch.float32), F.tanh(self.policy_net_x(state[:, 130:].to(torch.float32)))), 1))
        mean = self.mean_linear(x)
        log_std = self.log_std_linear(x)
        log_std = torch.clamp(log_std, min=LOG_SIG_MIN, max=LOG_SIG_MAX)
        return mean, log_std

    def sample(self, state):
        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = Normal(mean, std)
        x_t = normal.rsample()  # for reparameterization trick (mean + std * N(0,1))
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        log_prob = normal.log_prob(x_t)
        # Enforcing Action Bound
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + epsilon)
        log_prob = log_prob.sum(1, keepdim=True)
        mean = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, mean

    def to(self, device):
        self.action_scale = self.action_scale.to(device)
        self.action_bias = self.action_bias.to(device)
        return super(GaussianPolicy, self).to(device)


def train(policy_model, q_val_model1, q_val_model2, epochs, BATCH_SIZE=32):
    criterion = nn.MSELoss()
    optimiser1 = torch.optim.Adam(policy_model.parameters(), lr=0.0003)
    optimiser2 = torch.optim.Adam(q_val_model1.parameters(), lr=0.0003)
    cum_loss_policy = 0
    cum_loss_q_net = 0

    for ep in range(epochs):
        states, actions, rewards, next_states, next_actions = dg.get_batch(BATCH_SIZE)
        policy_model.zero_grad()
        q_val_model1.zero_grad()
        q_val_model2.zero_grad()
        preds, log_prob, mean = policy_model.sample(states.to(torch.float32))
        policy_loss = criterion(preds.to(torch.float32), actions.to(torch.float32))

        qvals = q_val_model1(states.to(torch.float32), actions.to(torch.float32))
        qvals_next = q_val_model2(next_states.to(torch.float32), next_actions.to(torch.float32))
        qvals_next = rewards + GAMMA * qvals_next

        q_loss = criterion(qvals.to(torch.float32), qvals_next.to(torch.float32))

        policy_loss.backward()
        optimiser1.step()

        q_loss.backward()
        optimiser2.step()

        # print("policy loss", policy_loss)
        # print("value loss", q_loss)

        cum_loss_policy += policy_loss
        cum_loss_q_net += q_loss

        if ep % 50 == 0:
            torch.save(policy_model.state_dict(), policy_fn1)
            torch.save(q_val_model1.state_dict(), value1_fn1)
            print("avg loss policy: ", cum_loss_policy / 50)
            print("avg loss qnet: ", cum_loss_q_net / 50)
            cum_loss_policy = 0
            cum_loss_q_net = 0

        if ep % 100 == 0:
            torch.save(q_val_model1.state_dict(), value2_fn1)

    return policy_model, q_val_model1, q_val_model2


if __name__ == "__main__":
    dg = datagen.Datagen()
    policy_model = GaussianPolicy(16)
    value_model1 = QNetwork(16)
    value_model2 = QNetwork_copy(16)

    policy_model.load_state_dict(torch.load(policy_fn1))
    value_model1.load_state_dict(torch.load(value1_fn1))
    value_model2.load_state_dict(torch.load(value2_fn1))
    train(policy_model=policy_model, q_val_model1=value_model1, q_val_model2=value_model2, epochs=4000, BATCH_SIZE=32)

