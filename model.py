import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np # everything should be in torch imo


class PrioritizedReplayBuffer:
    def __init__(self, capacity, alpha=0.6):
        self.capacity = capacity
        self.alpha = alpha
        self.buffer = []
        self.priorities = np.zeros(capacity, dtype=np.float32)
        self.pos = 0

    def add(self, transition, priority=1.0):
        if len(self.buffer) < self.capacity:
            self.buffer.append(transition)
        else:
            self.buffer[self.pos] = transition
        self.priorities[self.pos] = priority
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size):
        n = len(self.buffer)
        priorities = self.priorities[:n] ** self.alpha
        probs = priorities / priorities.sum()
        indices = np.random.choice(n, batch_size, p=probs, replace=False)
        transitions = [self.buffer[i] for i in indices]
        return transitions, indices

    def update_priorities(self, indices, losses):
        for idx, loss in zip(indices, losses):
            self.priorities[idx] = abs(loss) + 1e-6

    def __len__(self):
        return len(self.buffer)


class ActorCritic(nn.Module):
    def __init__(self, feature_dim, N=128, lr=1e-3, gamma=0.99):
        super().__init__()
        self.gamma = gamma
        self.N = N

        self.features = nn.Sequential(
            # CNN goes here
        )
        self.actor  = nn.Linear(feature_dim, 5)
        self.critic = nn.Linear(feature_dim, 1)

        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        self.rollout = []  # stores (state, action, log_prob, value, reward, done)

    def forward(self, state):
        f = self.features(state)
        return F.softmax(self.actor(f), dim=-1), self.critic(f)

    def get_action(self, state):
        probs, value = self.forward(state)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        return action.item(), dist.log_prob(action), value

    def store(self, state, action, log_prob, value, reward, done):
        self.rollout.append((state, action, log_prob, value, reward, done))

    def update(self):
        if len(self.rollout) < self.N:
            return

        _, _, log_probs, values, rewards, dones = zip(*self.rollout)

        rewards   = torch.tensor(rewards,   dtype=torch.float32)
        dones     = torch.tensor(dones,     dtype=torch.float32)
        values    = torch.stack(values).squeeze()
        log_probs = torch.stack(log_probs)

        td_errors = rewards + self.gamma * torch.cat([values[1:], torch.zeros(1)]) * (1 - dones) - values

        critic_loss = td_errors.pow(2).mean()
        actor_loss  = -(log_probs * td_errors.detach()).mean()

        self.optimizer.zero_grad()
        (actor_loss + critic_loss).backward()
        self.optimizer.step()

        self.rollout = []
