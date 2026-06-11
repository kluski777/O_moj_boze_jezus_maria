import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from abstract_car import AbstractCar

ACTIONS = ["forward", "backward", "left", "right", "stop"]

# TODO
# * ToCheck Critic inputs are CONSTANT
# *


def compute_gae(rewards, values, last_value, dones, gamma, lam):
    N = len(rewards)
    advantages = torch.zeros(N)
    gae = 0.0
    for t in reversed(range(N)):
        next_val = values[t + 1] if t + 1 < N else last_value
        delta = rewards[t] + gamma * next_val * (1 - dones[t]) - values[t]
        gae = delta + gamma * lam * (1 - dones[t]) * gae
        advantages[t] = gae
    return advantages


def _make_backbone():
    return nn.Sequential(
        nn.Conv2d(4, 32, kernel_size=5, stride=4),
        nn.GroupNorm(1, 32),  # GroupNorm(1, C) == LayerNorm over (C,H,W)
        nn.GELU(),
        nn.Conv2d(32, 64, kernel_size=5, stride=2),
        nn.GroupNorm(1, 64),
        nn.GELU(),
        nn.Conv2d(64, 96, kernel_size=5, stride=1),
        nn.GroupNorm(1, 96),
        nn.GELU(),
        nn.Conv2d(96, 96, kernel_size=3, stride=2),
        nn.GroupNorm(1, 96),
        nn.GELU(),
        nn.Flatten(),
    )


class ActorCritic(nn.Module, AbstractCar):
    def __init__(
        self,
        name,
        feature_dim,
        N=2048,
        batch_size=24,
        K=8,
        lr=2e-3,
        gamma=0.99,
        lam=0.95,
        eps=0.2,
        entropy_coef=0.05,
    ):
        nn.Module.__init__(self)
        AbstractCar.__init__(self, name)
        self.gamma = gamma
        self.lam = lam
        self.eps = eps
        self.N = N
        self.batch_size = batch_size
        self.K = K
        self.entropy_coef = entropy_coef

        self.actor = nn.Sequential(
            *_make_backbone(),
            nn.LazyLinear(feature_dim // 2),
            nn.GELU(),
            nn.Linear(feature_dim // 2, feature_dim // 4),
            nn.GELU(),
            nn.Linear(feature_dim // 4, 5),
        )
        self.critic = nn.Sequential(
            *_make_backbone(),
            nn.LazyLinear(feature_dim // 2),
            nn.GELU(),
            nn.Linear(feature_dim // 2, feature_dim // 4),
            nn.GELU(),
            nn.Linear(feature_dim // 4, 1),
        )

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)

        # materialize LazyLinear then apply orthogonal init
        with torch.no_grad():
            self.forward(torch.zeros(1, 4, 256, 256, device=self.device))
        self._init_weights()

        self.actor_optimizer  = torch.optim.Adam(self.actor.parameters(),  lr=lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr)
        self.rollout = []  # stores (state, action, old_log_prob, reward, done) — all on CPU

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.orthogonal_(m.weight, gain=math.sqrt(2))
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        nn.init.orthogonal_(self.actor[-1].weight, gain=0.01)
        nn.init.orthogonal_(self.critic[-1].weight, gain=1.0)

    def forward(self, state):
        return F.softmax(self.actor(state), dim=-1), self.critic(state)

    def choose_action(self, state):
        action_idx, log_prob, _ = self.get_action(state)
        self._last_action = action_idx
        self._last_log_prob = log_prob
        return ACTIONS[action_idx]

    def get_action(self, state):
        with torch.no_grad():
            probs, value = self.forward(state.to(self.device))
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        return action.item(), dist.log_prob(action).item(), value.item()

    def store(self, state, action, old_log_prob, reward, done):
        self.rollout.append((state, action, old_log_prob, reward, done))

    def update(self, last_next_state):
        if len(self.rollout) < self.N:
            return None, None, None, None, None, None, None, None, None, None

        states, actions, old_log_probs, rewards, dones = zip(*self.rollout)

        states        = torch.stack(states).to(self.device)
        actions       = torch.tensor(actions).to(self.device)
        old_log_probs = torch.tensor(old_log_probs, dtype=torch.float32).to(self.device)
        rewards       = torch.tensor(rewards,       dtype=torch.float32).to(self.device)
        dones         = torch.tensor(dones,         dtype=torch.float32).to(self.device)

        with torch.no_grad():
            probs_all, values = self.forward(states)
        values = values.squeeze()

        mean_max_prob = probs_all.max(dim=-1).values.mean().item()
        action_freq = torch.zeros(len(ACTIONS))
        for a in range(len(ACTIONS)):
            action_freq[a] = (actions == a).float().mean()
        action_freq = action_freq.cpu().numpy()

        last_value = 0.0 if dones[-1] else self.bootstrap(last_next_state)

        advantages = compute_gae(
            rewards, values.detach(), last_value, dones, self.gamma, self.lam
        ).to(self.device)
        returns = advantages + values.detach()

        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        total_actor   = 0.0
        total_critic  = 0.0
        total_entropy = 0.0
        total_ratio   = 0.0
        total_kl      = 0.0
        layer_norm_accum = {}
        count = 0

        for _ in range(self.K):
            idx = torch.randperm(self.N)
            for start in range(0, self.N, self.batch_size):
                mb = idx[start : start + self.batch_size]

                mb_probs, mb_values = self.forward(states[mb])
                mb_values   = mb_values.squeeze()
                mb_log_probs = torch.distributions.Categorical(mb_probs).log_prob(actions[mb])

                ratio  = torch.exp(mb_log_probs - old_log_probs[mb])
                mb_adv = advantages[mb]

                dist        = torch.distributions.Categorical(mb_probs)
                actor_loss  = -torch.min(
                    ratio * mb_adv,
                    torch.clamp(ratio, 1 - self.eps, 1 + self.eps) * mb_adv,
                ).mean()
                critic_loss = F.mse_loss(mb_values, returns[mb])
                entropy_loss = -self.entropy_coef * dist.entropy().mean()

                self.actor_optimizer.zero_grad()
                self.critic_optimizer.zero_grad()
                (actor_loss + critic_loss + entropy_loss).backward()

                nn.utils.clip_grad_norm_(self.actor.parameters(),  3.0)
                nn.utils.clip_grad_norm_(self.critic.parameters(), 3.0)

                # snapshot weights before the step for Karpathy update/weight ratio
                before = {
                    n: p.detach().clone()
                    for n, p in self.named_parameters()
                    if p.requires_grad
                }

                self.actor_optimizer.step()
                self.critic_optimizer.step()

                # |Δw|/|w| per module (Adam-aware, scale-invariant)
                upd_sq = {}
                wgt_sq = {}
                for n, p in self.named_parameters():
                    if n in before:
                        key = ".".join(n.split(".")[:2])
                        upd_sq[key] = upd_sq.get(key, 0.0) + (p.detach() - before[n]).norm(2).item() ** 2
                        wgt_sq[key] = wgt_sq.get(key, 0.0) + before[n].norm(2).item() ** 2
                for k in upd_sq:
                    ratio_k = (upd_sq[k] ** 0.5) / (wgt_sq[k] ** 0.5 + 1e-12)
                    layer_norm_accum[k] = layer_norm_accum.get(k, 0.0) + ratio_k

                total_actor   += actor_loss.item()
                total_critic  += critic_loss.item()
                total_entropy += entropy_loss.item()
                total_ratio   += ratio.mean().item()
                total_kl      += (old_log_probs[mb] - mb_log_probs).mean().item()
                count += 1

        self.rollout = []

        update_ratios = {k: v / count for k, v in layer_norm_accum.items()}
        return (
            total_actor   / count,
            total_critic  / count,
            total_entropy / count,
            total_ratio   / count,
            update_ratios,
            total_kl      / count,
            mean_max_prob,
            action_freq,
            values.detach().cpu().numpy(),
            returns.detach().cpu().numpy(),
        )

    def bootstrap(self, state):
        with torch.no_grad():
            _, value = self.forward(state.to(self.device))
        return value.squeeze().item()
