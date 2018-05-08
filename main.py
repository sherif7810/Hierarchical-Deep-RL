import random
from collections import deque

import gym

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

env = gym.make('SpaceInvaders-v0')
env.render()

alpha = torch.tensor(0.625)
gamma = torch.tensor(0.7)
epsilon = 0.925
N = 6


def wrap_state(state):
    """Wrap state in a tensor."""
    return torch.tensor(state).view(3, 210, 160).unsqueeze(0).float()


class DQN(nn.Module):
    """A NN from state to actions."""

    def __init__(self, num_actions, g_size, ram_size):
        super(DQN, self).__init__()
        self.g_size = g_size
        self.num_actions = num_actions

        self.conv1 = nn.Conv2d(3, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.sigmoid = nn.Sigmoid()

        self.fc = nn.Linear(22528 + 6, 1024)

        num_lstm_layers = 3
        self.lstm_hidden = (torch.rand(num_lstm_layers, 1, self.num_actions),
                            torch.rand(num_lstm_layers, 1, self.num_actions))
        self.lstm = nn.LSTM(1024, self.num_actions, num_lstm_layers)


        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.criterion = nn.MSELoss()

        t1 = wrap_state(env.reset())
        t2 = torch.zeros(1, self.g_size)
        self.D = deque(8 * [(t1, t2, 0, t1)], ram_size)

    def forward(self, x, g):
        batch_size = x.shape[0]
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.sigmoid(self.conv3(x))

        x = torch.cat((x.view(batch_size, -1), g), 1)
        x = self.sigmoid(self.fc(x))

        x, self.lstm_hidden = self.lstm(x.view(batch_size, 1, -1),
                                        self.lstm_hidden)
        self.lstm_hidden = (self.lstm_hidden[0].detach(),
                            self.lstm_hidden[1].detach())
        return x.view(batch_size, self.num_actions)

    def epsilon_greedy(self, state, g):
        action = 0
        if torch.rand(1)[0] > epsilon:
            action = env.action_space.sample()
        else:
            Q = self(state, g)
            action = Q.max(1)[1].item()
        return action

    def optimize(self, batch_size):
        batch = random.sample(self.D, batch_size)
        state1, g, reward, state2 = ([], [], [], [])
        for state1_, g_, reward_, state2_ in batch:
            state1.append(state1_)
            g.append(g_)
            reward.append(reward_)
            state2.append(state2_)
        state1 = torch.cat(state1)
        g = torch.cat(g).detach()  # It seems it keeps grad from meta-controller.
        reward = torch.tensor(reward).view(-1, 1)
        state2 = torch.cat(state2)

        target = reward.float() + gamma * self(state2, g).max(1)[0].view(-1, 1)
        target = target.repeat(1, self.num_actions)

        Q = self(state1, g)
        self.optimizer.zero_grad()
        loss = self.criterion(Q, target.detach())
        loss.backward()
        self.optimizer.step()


class MetaController(nn.Module):
    """Meta-controller that gives policy for critic."""

    def __init__(self, g_size, ram_size):
        super(MetaController, self).__init__()
        self.g_size = g_size

        self.conv1 = nn.Conv2d(3, 16, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(16, 16, kernel_size=4)

        num_lstm_layers = 2
        self.lstm_hidden = (torch.rand(num_lstm_layers, 1, self.g_size),
                            torch.rand(num_lstm_layers, 1, self.g_size))
        self.lstm = nn.LSTM(27648, self.g_size, num_lstm_layers)

        self.sigmoid = nn.Sigmoid()

        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.criterion = nn.MSELoss()

        t = wrap_state(env.reset())
        self.D = deque(8 * [(t, 0, t)], ram_size)

    def forward(self, x):
        batch_size = x.shape[0]
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))

        x, self.lstm_hidden = self.lstm(x.view(batch_size, 1, -1),
                                        self.lstm_hidden)
        self.lstm_hidden = (self.lstm_hidden[0].detach(),
                            self.lstm_hidden[1].detach())

        x = self.sigmoid(x.view(batch_size, self.g_size))
        return x

    def epsilon_greedy(self, state):
        g = 0
        if torch.rand(1)[0] > epsilon:
            g = torch.rand(1, self.g_size)
        else:
            g = self(state)
        return g

    def optimize(self, batch_size):
        batch = random.sample(self.D, batch_size)
        state1, reward, state2 = ([], [], [])
        for state1_, reward_, state2_ in batch:
            state1.append(state1_)
            reward.append(reward_)
            state2.append(state2_)
        state1 = torch.cat(state1, 0)
        reward = torch.tensor(reward).view(-1, 1)
        state2 = torch.cat(state2, 0)

        target = reward.float() + gamma * self(state2).max(1)[0].view(-1, 1)
        target = target.repeat(1, self.g_size)

        Q = self(state1)

        self.optimizer.zero_grad()
        loss = self.criterion(Q, target.detach())
        loss.backward()
        self.optimizer.step()


class Agent:
    """Hierarchical DQN agent."""

    def __init__(self, num_actions, g_size, ram_size):
        self.num_actions = num_actions
        self.g_size = g_size

        self.meta_controller = MetaController(self.g_size, ram_size)
        self.critic = DQN(num_actions, self.g_size, ram_size)

    def update(self):
        """Update the meta-controller and the critic."""
        self.critic.optimize(6)
        self.meta_controller.optimize(6)


if __name__ == '__main__':
    agent = Agent(env.action_space.n, 6, 20)

    for episode in range(100):
        done = False
        G = 0

        state0 = wrap_state(env.reset())
        state1 = state0

        g = agent.meta_controller.epsilon_greedy(state1)
        while done is not True:
            extrinsic_reward = 0
            n = 0

            while not done and n < N:
                action = agent.critic.epsilon_greedy(state1, g)
                state2, f, done, info = env.step(action)

                state2 = wrap_state(state2)
                agent.critic.D.append((state1, g, f, state2))

                agent.update()

                extrinsic_reward += f
                state1 = state2

                n += 1

                G += f
                env.render()
            agent.meta_controller.D.append((state0, extrinsic_reward, state1))
            if not done:
                g = agent.meta_controller.epsilon_greedy(state1)

        # Reward display:
        episode_number = episode + 1
        if episode_number % 10 == 0:
            print("Episode {}: Total reward = {}.".format(episode_number, G))
