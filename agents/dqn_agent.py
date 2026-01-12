"""
agents/dqn_agent.py
Single-agent DQN — pure NumPy.
"""

import numpy as np
from collections import deque
import random
from agents.iql_agent import NumpyMLP, ReplayBuffer


class DQNAgent:
    def __init__(self, obs_size, action_size, seed=42,
                 lr=1e-3, gamma=0.99, batch_size=64,
                 eps_start=1.0, eps_end=0.05, eps_decay=0.995,
                 target_update=50):
        self.obs_size    = obs_size
        self.action_size = action_size
        self.lr          = lr
        self.gamma       = gamma
        self.batch_size  = batch_size
        self.epsilon     = eps_start
        self.eps_end     = eps_end
        self.eps_decay   = eps_decay
        self.target_update = target_update
        self.learn_steps   = 0

        random.seed(seed)
        np.random.seed(seed)

        self.qnet   = NumpyMLP(obs_size, 128, action_size, seed=seed)
        self.target = NumpyMLP(obs_size, 128, action_size, seed=seed)
        self.target.copy_from(self.qnet)
        self.buffer = ReplayBuffer(capacity=10000)

    def select_action(self, obs, deterministic=False):
        if not deterministic and np.random.rand() < self.epsilon:
            return int(np.random.randint(self.action_size))
        q = self.qnet.predict(np.array(obs, dtype=np.float32).reshape(1, -1))[0]
        return int(np.argmax(q))

    def store(self, obs, action, reward, next_obs, done):
        self.buffer.push(obs, action, reward, next_obs, done)

    def train(self):
        if len(self.buffer) < self.batch_size:
            return None
        obs, acts, rews, nobs, dones = self.buffer.sample(self.batch_size)

        q_all   = self.qnet.forward(obs)
        q_taken = q_all[np.arange(self.batch_size), acts]
        q_next  = self.target.predict(nobs)
        q_tgt   = rews + self.gamma * np.max(q_next, axis=1) * (1 - dones)

        td  = q_taken - q_tgt
        dout = np.zeros_like(q_all)
        dout[np.arange(self.batch_size), acts] = 2.0 * td / self.batch_size
        self.qnet.backward(dout, lr=self.lr, clip=1.0)

        self.epsilon = max(self.eps_end, self.epsilon * self.eps_decay)
        self.learn_steps += 1
        if self.learn_steps % self.target_update == 0:
            self.target.copy_from(self.qnet)
