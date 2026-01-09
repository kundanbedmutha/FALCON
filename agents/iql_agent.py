"""
agents/iql_agent.py
Independent Q-Learning agent — pure NumPy, no PyTorch.

Uses a simple 2-layer MLP with forward pass + manual SGD.
Gradient clipping applied to prevent explosion (fixes the loss=70T bug).
"""

import numpy as np
from collections import deque
import random


class NumpyMLP:
    """Lightweight 2-layer MLP with ReLU, implemented in NumPy."""

    def __init__(self, input_dim, hidden_dim, output_dim, seed=42):
        rng = np.random.RandomState(seed)
        # He initialization
        self.W1 = rng.randn(input_dim,  hidden_dim).astype(np.float32) * np.sqrt(2.0 / input_dim)
        self.b1 = np.zeros(hidden_dim, dtype=np.float32)
        self.W2 = rng.randn(hidden_dim, output_dim).astype(np.float32) * np.sqrt(2.0 / hidden_dim)
        self.b2 = np.zeros(output_dim, dtype=np.float32)

    def forward(self, x):
        """x: (batch, input_dim) → (batch, output_dim)"""
        self.x    = x
        self.z1   = x @ self.W1 + self.b1
        self.a1   = np.maximum(0, self.z1)          # ReLU
        self.out  = self.a1 @ self.W2 + self.b2
        return self.out

    def predict(self, x):
        """Single-sample prediction without storing activations."""
        a1  = np.maximum(0, x @ self.W1 + self.b1)
        return a1 @ self.W2 + self.b2

    def backward(self, dout, lr=1e-3, clip=1.0):
        """
        dout: (batch, output_dim) — gradient of loss w.r.t. output
        Returns loss-related gradients and updates weights in place.
        """
        batch = self.x.shape[0]

        # Layer 2
        dW2 = self.a1.T @ dout / batch
        db2 = dout.mean(axis=0)

        # Layer 1
        da1 = dout @ self.W2.T
        dz1 = da1 * (self.z1 > 0).astype(np.float32)    # ReLU grad
        dW1 = self.x.T @ dz1 / batch
        db1 = dz1.mean(axis=0)

        # Gradient clipping (prevents loss explosion)
        for g in [dW1, db1, dW2, db2]:
            np.clip(g, -clip, clip, out=g)

        self.W1 -= lr * dW1
        self.b1 -= lr * db1
        self.W2 -= lr * dW2
        self.b2 -= lr * db2

    def copy_from(self, other):
        self.W1 = other.W1.copy()
        self.b1 = other.b1.copy()
        self.W2 = other.W2.copy()
        self.b2 = other.b2.copy()


class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.buf = deque(maxlen=capacity)

    def push(self, obs, action, reward, next_obs, done):
        self.buf.append((obs, action, float(reward), next_obs, float(done)))

    def sample(self, batch_size):
        batch = random.sample(self.buf, batch_size)
        obs, acts, rews, nobs, dones = zip(*batch)
        return (np.array(obs,   dtype=np.float32),
                np.array(acts,  dtype=np.int32),
                np.array(rews,  dtype=np.float32),
                np.array(nobs,  dtype=np.float32),
                np.array(dones, dtype=np.float32))

    def __len__(self):
        return len(self.buf)


class IQLAgent:
    def __init__(self, agent_id, obs_dim, action_dim,
                 hidden=256, lr=1e-3, gamma=0.99,
                 batch_size=64, seed=42,
                 eps_start=1.0, eps_end=0.05, eps_decay=0.995,
                 target_update=50):
        self.agent_id   = agent_id
        self.obs_dim    = obs_dim
        self.action_dim = action_dim
        self.lr         = lr
        self.gamma      = gamma
        self.batch_size = batch_size
        self.epsilon    = eps_start
        self.eps_end    = eps_end
        self.eps_decay  = eps_decay
        self.target_update = target_update
        self.learn_steps   = 0
        self._losses       = []

        rng_seed = seed + agent_id * 37
        random.seed(rng_seed)
        np.random.seed(rng_seed)

        self.qnet    = NumpyMLP(obs_dim, hidden, action_dim, seed=rng_seed)
        self.target  = NumpyMLP(obs_dim, hidden, action_dim, seed=rng_seed)
        self.target.copy_from(self.qnet)
        self.buffer  = ReplayBuffer(capacity=10000)

    def select_action(self, obs, valid_actions=None, training=True):
        if valid_actions is None:
            valid_actions = list(range(self.action_dim))

        if training and np.random.rand() < self.epsilon:
            return int(np.random.choice(valid_actions))

        q = self.qnet.predict(np.array(obs, dtype=np.float32).reshape(1, -1))[0]
        # Mask invalid actions
        mask       = np.full(self.action_dim, -1e9, dtype=np.float32)
        mask[valid_actions] = 0.0
        q_masked   = q + mask
        return int(np.argmax(q_masked))

    def store(self, obs, action, reward, next_obs, done):
        self.buffer.push(obs, action, reward, next_obs, done)

    def learn(self):
        if len(self.buffer) < self.batch_size:
            return None

        obs, acts, rews, nobs, dones = self.buffer.sample(self.batch_size)

        # Current Q values
        q_all    = self.qnet.forward(obs)                    # (B, A)
        q_taken  = q_all[np.arange(self.batch_size), acts]  # (B,)

        # Target Q values (no-grad target network)
        q_next   = self.target.predict(nobs)                 # (B, A)
        q_target = rews + self.gamma * np.max(q_next, axis=1) * (1 - dones)

        # MSE loss gradient: dL/dQ = 2*(Q - target) / B
        td_error = q_taken - q_target                        # (B,)
        loss     = float(np.mean(td_error ** 2))

        # Backprop: only update the taken-action outputs
        dout = np.zeros_like(q_all)
        dout[np.arange(self.batch_size), acts] = 2.0 * td_error / self.batch_size

        self.qnet.backward(dout, lr=self.lr, clip=1.0)

        # Decay epsilon
        self.epsilon = max(self.eps_end, self.epsilon * self.eps_decay)

        # Soft-update target
        self.learn_steps += 1
        if self.learn_steps % self.target_update == 0:
            self.target.copy_from(self.qnet)

        self._losses.append(loss)
        return loss

    def get_metrics(self):
        return {
            "epsilon":    round(self.epsilon, 4),
            "buffer_len": len(self.buffer),
            "avg_loss":   float(np.mean(self._losses)) if self._losses else 0.0,
        }
