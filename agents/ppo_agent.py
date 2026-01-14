"""
agents/ppo_agent.py
Single-agent PPO (clip variant) — pure NumPy.

Stores a rollout buffer, runs policy/value update every episode.
"""

import numpy as np
from agents.iql_agent import NumpyMLP


class PPOAgent:
    def __init__(self, obs_size, action_size, seed=42,
                 lr=3e-4, gamma=0.99, lam=0.95,
                 clip_eps=0.2, n_epochs=4, batch_size=64):
        self.obs_size    = obs_size
        self.action_size = action_size
        self.lr          = lr
        self.gamma       = gamma
        self.lam         = lam
        self.clip_eps    = clip_eps
        self.n_epochs    = n_epochs
        self.batch_size  = batch_size

        np.random.seed(seed)

        # Shared feature extractor → policy head + value head
        self.actor  = NumpyMLP(obs_size, 128, action_size, seed=seed)
        self.critic = NumpyMLP(obs_size, 128, 1,           seed=seed + 1)

        # Rollout buffer
        self._clear_buffer()

    def _clear_buffer(self):
        self._obs   = []
        self._acts  = []
        self._rews  = []
        self._vals  = []
        self._lps   = []
        self._dones = []

    def _softmax(self, x):
        x = x - x.max(axis=-1, keepdims=True)
        e = np.exp(x)
        return e / (e.sum(axis=-1, keepdims=True) + 1e-8)

    def select_action(self, obs, deterministic=False):
        obs_arr = np.array(obs, dtype=np.float32).reshape(1, -1)
        logits  = self.actor.predict(obs_arr)[0]
        val     = float(self.critic.predict(obs_arr)[0, 0])
        probs   = self._softmax(logits)

        if deterministic:
            action = int(np.argmax(probs))
        else:
            action = int(np.random.choice(self.action_size, p=probs))

        log_prob = float(np.log(probs[action] + 1e-8))
        return action, val, log_prob

    def store(self, obs, action, reward, val, log_prob, done):
        self._obs.append(obs)
        self._acts.append(action)
        self._rews.append(reward)
        self._vals.append(val)
        self._lps.append(log_prob)
        self._dones.append(done)

    def train(self):
        if len(self._obs) < self.batch_size:
            return

        obs   = np.array(self._obs,   dtype=np.float32)
        acts  = np.array(self._acts,  dtype=np.int32)
        rews  = np.array(self._rews,  dtype=np.float32)
        vals  = np.array(self._vals,  dtype=np.float32)
        lps   = np.array(self._lps,   dtype=np.float32)
        dones = np.array(self._dones, dtype=np.float32)

        # GAE returns
        T       = len(rews)
        advs    = np.zeros(T, dtype=np.float32)
        returns = np.zeros(T, dtype=np.float32)
        gae     = 0.0
        for t in reversed(range(T)):
            nv     = vals[t + 1] if t + 1 < T else 0.0
            delta  = rews[t] + self.gamma * nv * (1 - dones[t]) - vals[t]
            gae    = delta + self.gamma * self.lam * (1 - dones[t]) * gae
            advs[t]    = gae
            returns[t] = gae + vals[t]

        advs = (advs - advs.mean()) / (advs.std() + 1e-8)

        # Mini-batch updates
        idx = np.arange(T)
        for _ in range(self.n_epochs):
            np.random.shuffle(idx)
            for start in range(0, T, self.batch_size):
                mb = idx[start:start + self.batch_size]
                if len(mb) < 2:
                    continue
                ob  = obs[mb]; ac = acts[mb]; ret = returns[mb]
                adv = advs[mb]; lp_old = lps[mb]

                # Actor forward
                logits = self.actor.forward(ob)
                probs  = self._softmax(logits)
                lp_new = np.log(probs[np.arange(len(mb)), ac] + 1e-8)

                ratio  = np.exp(lp_new - lp_old)
                s1     = ratio * adv
                s2     = np.clip(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * adv
                actor_loss = -np.minimum(s1, s2).mean()

                # Actor backward
                d_actor = np.zeros_like(logits)
                clip_mask = ((ratio < 1 - self.clip_eps) | (ratio > 1 + self.clip_eps))
                eff_ratio = np.where(clip_mask, np.clip(ratio, 1-self.clip_eps, 1+self.clip_eps), ratio)
                for b_i in range(len(mb)):
                    p = probs[b_i]
                    d_lp = -adv[b_i] * eff_ratio[b_i]
                    d_log_p = np.zeros(self.action_size, dtype=np.float32)
                    d_log_p[ac[b_i]] = d_lp
                    d_actor[b_i] = d_log_p - p * d_log_p.sum()
                self.actor.backward(d_actor / len(mb), lr=self.lr, clip=0.5)

                # Critic update
                v_pred = self.critic.forward(ob)[:, 0]
                v_err  = v_pred - ret
                d_critic = (2.0 * v_err / len(mb)).reshape(-1, 1)
                self.critic.backward(d_critic, lr=self.lr, clip=0.5)

        self._clear_buffer()
