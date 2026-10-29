import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque

class DQNScheduler(nn.Module):
    """Deep Q-Network Model Architecture."""
    def __init__(self, state_dim=12, action_dim=9):
        super(DQNScheduler, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Linear(128, action_dim)
        )

    def forward(self, x):
        return self.net(x)


class ReplayBuffer:
    """Experience Replay Memory."""
    def __init__(self, capacity=20000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        state, action, reward, next_state, done = zip(*random.sample(self.buffer, batch_size))
        return (
            torch.FloatTensor(np.array(state)),
            torch.LongTensor(action),
            torch.FloatTensor(reward),
            torch.FloatTensor(np.array(next_state)),
            torch.FloatTensor(done)
        )

    def __len__(self):
        return len(self.buffer)


class DQNAgent:
    """DQN Learning Agent with Target Network & Epsilon-Greedy Exploration."""
    def __init__(self, state_dim=12, action_dim=9, lr=1e-3, gamma=0.99):
        self.action_dim = action_dim
        self.gamma = gamma
        self.policy_net = DQNScheduler(state_dim, action_dim)
        self.target_net = DQNScheduler(state_dim, action_dim)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.memory = ReplayBuffer()
        
        self.epsilon = 1.0
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.993

    def select_action(self, state, eval_mode=False):
        if not eval_mode and random.random() < self.epsilon:
            return random.randint(0, self.action_dim - 1)
        
        with torch.no_grad():
            state_t = torch.FloatTensor(state).unsqueeze(0)
            q_values = self.policy_net(state_t)
            return q_values.argmax(dim=1).item()

    def train_step(self, batch_size=64):
        if len(self.memory) < batch_size:
            return 0.0
            
        states, actions, rewards, next_states, dones = self.memory.sample(batch_size)
        
        q_values = self.policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        
        with torch.no_grad():
            max_next_q = self.target_net(next_states).max(1)[0]
            target_q = rewards + (1.0 - dones) * self.gamma * max_next_q
            
        loss = nn.MSELoss()(q_values, target_q)
        
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=1.0)
        self.optimizer.step()
        
        return loss.item()

    def update_target_network(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def save(self, filepath):
        torch.save(self.policy_net.state_dict(), filepath)

    def load(self, filepath):
        state_dict = torch.load(filepath, map_location="cpu", weights_only=True)
        self.policy_net.load_state_dict(state_dict)
        self.target_net.load_state_dict(self.policy_net.state_dict())